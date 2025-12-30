import shutil
from pathlib import Path
import kagglehub

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_PetImages(dst_path: str):
    dst_path = Path(dst_path)
    # Download latest version
    src_path = kagglehub.dataset_download(handle='bhavikjikadara/dog-and-cat-classification-dataset')
    src_path = Path(src_path) / 'PetImages'

    print('Downlod dataset to path:', src_path)
    try:
        shutil.move(src_path, dst_path.parent)
        print('Move dataset to input data folder:', str(dst_path))
    except FileNotFoundError:
        print('Source directory not found:', str(src_path))
    except Exception as e:
        print('An unexpected error occurred:', e)

    return str(dst_path)


def get_dataloaders(
    data_dir,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15
    ):
    assert train_ratio + val_ratio < 1.0

    load_PetImages(data_dir)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(root=data_dir)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader
