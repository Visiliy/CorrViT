import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


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


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    return total_loss / len(loader), 100. * correct / total


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    full_dataset = CustomDataset(root_dir='cropped_objects', transform=transform_train)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

    print(f'Number of classes: {len(full_dataset.classes)}')
    print(f'Classes: {full_dataset.classes}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')

    model = CorrViT(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=len(full_dataset.classes),
        d_model=384,
        depth=4,
        dropout=0.1
    ).to(device)

    print(f'Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999))

    total_steps = len(train_loader) * 100
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    num_epochs = 100
    best_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-' * 50)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        print(f'\nEpoch {epoch} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'classes': full_dataset.classes
            }, 'best_model2.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch} epochs')
                break

    print(f'\nTraining completed! Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()