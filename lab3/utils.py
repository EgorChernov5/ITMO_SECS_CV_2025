import os
import shutil
from pathlib import Path
import kagglehub
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
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
    val_ratio=0.2
    ):
    assert val_ratio < 1.0

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
    val_size = int(val_ratio*total_size)
    train_size = total_size - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size]
    )

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

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

    test_ds = datasets.ImageFolder(
        root=data_dir.replace('input', 'output'),
        transform=val_transform
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=5,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total


def train_model(
        model,
        train_loader, val_loader,
        epochs,
        criterion,
        optimizer,
        device='cpu'):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total


def visualize_losses_metrics(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    save_path,
    figsize=(10, 4),
    dpi=300,
    show=False
):
    """
    Сохраняет графики loss и accuracy в файл.

    Args:
        train_losses (list): потери на обучении
        val_losses (list): потери на валидации
        train_accs (list): accuracy на обучении
        val_accs (list): accuracy на валидации
        save_path (str): путь к файлу (например 'plots/metrics.png')
        figsize (tuple): размер фигуры
        dpi (int): разрешение картинки
        show (bool): показывать ли график
    """
    plt.figure(figsize=figsize)

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Изменение функции потерь')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train accuracy')
    plt.plot(val_accs, label='Val accuracy')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Изменение точности')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()
