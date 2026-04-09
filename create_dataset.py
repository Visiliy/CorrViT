import numpy as np
import cv2
from pathlib import Path
import imgaug.augmenters as iaa
import multiprocessing
import warnings

warnings.filterwarnings('ignore')

np.bool = bool


class ImageAugmenter:
    def __init__(self, dataset_path, target_size=(224, 224)):
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.classes = [0, 1, 2, 3, 4, 5]

        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            iaa.Affine(
                rotate=(-45, 45),
                scale=(0.8, 1.2),
                translate_percent=(-0.2, 0.2),
                shear=(-16, 16),
                mode='edge'
            ),
            iaa.SomeOf((2, 5), [
                iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.1 * 255)),
                iaa.AdditivePoissonNoise(lam=(0, 10)),
                iaa.Multiply((0.7, 1.3)),
                iaa.LinearContrast((0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0.0, 2.0)),
                iaa.AverageBlur(k=(2, 5)),
                iaa.MedianBlur(k=(3, 5)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0.5, 1.5)),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.MultiplyHueAndSaturation(mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)),
                iaa.Grayscale(alpha=(0.0, 0.5)),
                iaa.HistogramEqualization(),
                iaa.CLAHE(clip_limit=(1, 10)),
                iaa.ElasticTransformation(alpha=(0, 40.0), sigma=(4.0, 8.0)),
                iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                iaa.PerspectiveTransform(scale=(0.01, 0.1)),
                iaa.JpegCompression(compression=(50, 90)),
                iaa.SaltAndPepper(p=(0.01, 0.05)),
                iaa.CoarseSaltAndPepper(p=(0.01, 0.03), size_percent=(0.02, 0.1)),
                iaa.Dropout(p=(0.01, 0.05)),
                iaa.CoarseDropout(p=(0.01, 0.03), size_percent=(0.02, 0.1)),
                iaa.Invert(p=0.1),
                iaa.Solarize(p=0.1, threshold=(32, 128)),
                iaa.Posterize(nb_bits=(4, 8)),
                iaa.Cutout(nb_iterations=(1, 3), size=(0.05, 0.2), squared=False),
                iaa.Superpixels(p_replace=(0.3, 0.7), n_segments=(50, 100))
            ])
        ], random_order=True)

    def load_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
        return img, image_path

    def save_image(self, img, original_path, suffix):
        class_dir = original_path.parent
        stem = original_path.stem
        ext = original_path.suffix or '.jpg'
        new_filename = f"{stem}_{suffix}{ext}"
        new_path = class_dir / new_filename
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(new_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def process_image(self, args):
        img_path, class_id, num_augmentations = args
        try:
            img, original_path = self.load_image(img_path)
            if img is None:
                return 0
            images = np.array([img] * num_augmentations)
            augmented_images = self.augmenter(images=images)
            for i, aug_img in enumerate(augmented_images):
                self.save_image(aug_img, original_path, f"aug_{i:03d}")
            return len(augmented_images)
        except Exception:
            return 0

    def get_image_files(self, class_dir):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        all_extensions = set()
        for ext in valid_extensions:
            all_extensions.add(ext.lower())
            all_extensions.add(ext.upper())
        files = []
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in all_extensions and 'aug_' not in f.stem:
                files.append(f)
        return files

    def augment_class(self, class_id, num_augmentations=30):
        class_dir = self.dataset_path / str(class_id)
        if not class_dir.exists():
            return 0
        image_files = self.get_image_files(class_dir)
        if not image_files:
            return 0
        args_list = [(img_path, class_id, num_augmentations) for img_path in image_files]
        cpu_count = max(1, multiprocessing.cpu_count() - 1)
        total_augmented = 0
        for args in args_list:
            total_augmented += self.process_image(args)
        return total_augmented

    def augment_all(self, num_augmentations=30):
        total_images = 0
        for class_id in self.classes:
            count = self.augment_class(class_id, num_augmentations)
            total_images += count
            print(f"Class {class_id}: {count} augmented images created")
        return total_images


if __name__ == "__main__":
    dataset_path = input("Enter path to dataset folder: ").strip()
    augmenter = ImageAugmenter(dataset_path)
    print("Starting augmentation...")
    total = augmenter.augment_all(num_augmentations=30)
    print(f"Augmentation complete! Created {total} new images")