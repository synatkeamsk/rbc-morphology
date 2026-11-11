"""
src/train.py
Mixed-precision training loop with class-imbalance weighting and validation tracking.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from src.data import create_dataloaders
from src.model import get_model


class FocalLoss(nn.Module):
    """Focal Loss to counter extreme class imbalance in rare RBC variants."""
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc="Validating", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, macro_f1, all_targets, all_preds


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading metadata from {args.train_csv} and {args.val_csv}...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # Square-root smoothed class weights to prevent normal cell suppression
    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-5)
    class_weights = torch.FloatTensor(class_weights / class_weights.sum()).to(device)

    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"Building model: {args.backbone} with {args.num_classes} classes...")
    model = get_model(num_classes=args.num_classes, backbone_name=args.backbone, device=device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_macro_f1 = 0.0

    print("Starting SmearScope training loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, macro_f1, targets, preds = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_smearscope.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": macro_f1,
                "classes": args.num_classes
            }, checkpoint_path)
            print(f"[*] New best Macro-F1: {best_macro_f1:.4f} -> Saved to {checkpoint_path}")

    print("\n--- Final Best Model Evaluation Report ---")
    print(classification_report(targets, preds, digits=4, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmearScope RBC Training Pipeline")
    parser.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--backbone", type=str, default="convnext_nano")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
