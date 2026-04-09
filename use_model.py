import os
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
from scipy import ndimage
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        return x

class CorrelationBlock(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = F.normalize(k, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        corr = torch.matmul(k.transpose(-2, -1), v) * self.scale
        corr = torch.tanh(corr)

        q = F.normalize(q, p=2, dim=-1)
        z = torch.matmul(q, corr)

        z = z.transpose(1, 2).contiguous().reshape(B, S, D)
        z = self.proj(z)

        return z

class ConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

    def forward(self, x):
        B, S, D = x.shape
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)

class Router(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.router(x)
        return weights[:, :, 0:1], weights[:, :, 1:2]

class HybridBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.corr_block = CorrelationBlock(d_model)
        self.conv_block = ConvBlock(d_model)
        self.router = Router(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)

        corr_out = self.corr_block(x)
        conv_out = self.conv_block(x)

        w_corr, w_conv = self.router(x)

        x = residual + self.dropout(w_corr * corr_out + w_conv * conv_out)
        x = self.norm2(x)

        return x

class CorrViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, d_model=256, depth=8, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)

        self.blocks = nn.ModuleList([
            HybridBlock(d_model, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x

class ObjectDetector:
    def __init__(self, classifier_model_path='model_weights_last4.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.classifier_model = CorrViT(
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=6,
            d_model=384,
            depth=4,
            dropout=0.1
        ).to(self.device)
        self.classifier_model.load_state_dict(torch.load(classifier_model_path, map_location=self.device)['model_state_dict'])
        self.classifier_model.eval()
        total_params = sum(p.numel() for p in self.classifier_model.parameters())
        print(f"Всего параметров: {total_params}")
        self.classifier_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
        ])

    def detect_objects(self, image_path):
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        gray = image.convert('L')
        arr = np.array(gray)

        smoothed = ndimage.gaussian_filter(arr, sigma=2)
        binary = smoothed < 128

        struct = ndimage.generate_binary_structure(2, 2)
        closed = ndimage.binary_closing(binary, structure=struct, iterations=2)
        labeled_im, num = ndimage.label(closed.astype(int))

        centers = []
        if num > 0:
            component_sizes = ndimage.sum(np.ones_like(labeled_im), labeled_im, index=np.arange(1, num + 1))
            min_area = 100
            valid_labels = [i for i, size in enumerate(component_sizes, start=1) if size >= min_area]
            if valid_labels:
                centers = ndimage.center_of_mass(np.ones_like(labeled_im), labeled_im, index=valid_labels)
                centers = [(int(round(cy)), int(round(cx))) for (cy, cx) in centers if not (np.isnan(cy) or np.isnan(cx))]

        detections = []
        for cy, cx in centers:
            crop_size = 180
            half = crop_size // 2
            left, top = cx - half, cy - half
            right, bottom = cx + half, cy + half

            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - orig_w)
            pad_bottom = max(0, bottom - orig_h)

            if any(p > 0 for p in [pad_left, pad_top, pad_right, pad_bottom]):
                new_w = orig_w + pad_left + pad_right
                new_h = orig_h + pad_top + pad_bottom
                padded = Image.new('RGB', (new_w, new_h), (255, 255, 255))
                padded.paste(image, (pad_left, pad_top))
                nx, ny = cx + pad_left, cy + pad_top
                crop = padded.crop((nx - half, ny - half, nx + half, ny + half))
            else:
                crop = image.crop((left, top, right, bottom))

            if crop.size[0] < 32 or crop.size[1] < 32:
                continue
            crop = crop.resize((32, 32), Image.BILINEAR)
            detections.append({'center': (cx, cy), 'crop': crop})

        return detections, image

    def classify_objects(self, detections):
        results = []
        for det in detections:
            crop_tensor = self.classifier_transform(det['crop']).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.classifier_model(crop_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            results.append({
                'center': det['center'],
                'class': predicted_class.item(),
                'confidence': confidence.item()
            })
        return results

    def process_image(self, image_path):
        detections, original_image = self.detect_objects(image_path)
        results = self.classify_objects(detections)
        return {
            'image_path': image_path,
            'original_size': original_image.size,
            'detections': results
        }

def main():
    detector = ObjectDetector(classifier_model_path='best_model.pth')
    image_path = "WIN_20260110_12_53_10_Pro.jpg"
    if not os.path.exists(image_path):
        print("Файл не найден!")
        return
    print(f"Обработка изображения: {image_path}")
    result = detector.process_image(image_path)
    print(f"Размер изображения: {result['original_size']}")
    print(f"Найдено объектов: {len(result['detections'])}")
    for i, det in enumerate(result['detections']):
        print(f"\nОбъект {i+1}:")
        print(f"Координаты центра: {det['center']}")
        print(f"Класс: {det['class']}")
        print(f"Уверенность: {det['confidence']:.4f}")

if __name__ == '__main__':
    main()