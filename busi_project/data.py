import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ("benign", "malignant", "normal")


def collect_images(dataset_dir: Path) -> List[Tuple[Path, str]]:
    """Collect real images and ignore BUSI segmentation masks."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    items = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS and "_mask" not in path.stem.lower():
                items.append((path, class_name))
    if not items:
        raise RuntimeError(f"No images found under {dataset_dir}")
    return items


def make_splits(items, test_size, validation_size, seed):
    labels = [label for _, label in items]
    train, temporary = train_test_split(
        items, test_size=test_size + validation_size,
        stratify=labels, random_state=seed
    )
    temporary_labels = [label for _, label in temporary]
    test_fraction = test_size / (test_size + validation_size)
    validation, test = train_test_split(
        temporary, test_size=test_fraction,
        stratify=temporary_labels, random_state=seed
    )
    return train, validation, test



def copy_split(items, target_dir: Path):
    for source, label in items:
        destination = target_dir / label
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)


def prepare_generators(config):
    items = collect_images(config.dataset_dir)
    train_items, validation_items, test_items = make_splits(
        items, config.test_size, config.validation_size, config.seed
    )
    if config.split_dir.exists():
        shutil.rmtree(config.split_dir)
    train_dir = config.split_dir / "train"
    validation_dir = config.split_dir / "validation"
    test_dir = config.split_dir / "test"
    copy_split(train_items, train_dir)
    copy_split(validation_items, validation_dir)
    copy_split(test_items, test_dir)

    size = (config.image_size, config.image_size)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest"
    )
    plain_datagen = ImageDataGenerator(rescale=1.0 / 255)
    common = dict(target_size=size, batch_size=config.batch_size,
                  class_mode="categorical")
    train_gen = train_datagen.flow_from_directory(train_dir, shuffle=True, **common)
    validation_gen = plain_datagen.flow_from_directory(validation_dir, shuffle=False, **common)
    test_gen = plain_datagen.flow_from_directory(test_dir, shuffle=False, **common)
    return train_gen, validation_gen, test_gen


def class_weights(generator):
    from sklearn.utils.class_weight import compute_class_weight
    labels = generator.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))
