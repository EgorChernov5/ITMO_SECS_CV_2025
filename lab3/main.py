import os
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dotenv import load_dotenv
load_dotenv()

from lab3.utils import *
from lab3.models import CustomResNet50, SimpleCNN


PATH_DATA = Path(os.getenv('PATH_DATA'))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/input/PetImages')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='./data/output')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(epochs.input)

    # model = CustomResNet50()
    model = SimpleCNN()

    # Parameters
    epochs = args.epochs
    lr = 1e-3

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    train_losses, val_losses, train_accs, val_accs = train_model(
        model,
        train_loader, val_loader,
        epochs,
        criterion,
        optimizer,
        DEVICE
    )

    # Evaluate and vis
    evaluate(model, test_loader, criterion, DEVICE)
    visualize_losses_metrics(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        str(Path(args.outdir) / 'graph_vis.png'),
        figsize=(10, 4),
        dpi=300,
        show=False
    )


if __name__ == '__main__':
    main()
