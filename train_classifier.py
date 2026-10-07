"""
Stage 2 - train the tuber-condition classifier on cropped images.

--model lets you reproduce the base paper's architecture comparison table:
  alexnet | resnet50 | seresnet50 (ResNet+SE, the "combined" model) |
  efficientnet_b0
timm supplies all four with matching pretrained-weight loading, so swapping
architectures is a one-flag change - keep every other hyperparameter fixed
for a fair comparison, the way the base paper did.
"""
import argparse
import json
import time
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

MODEL_MAP = {
    "alexnet": "alexnet",
    "resnet50": "resnet50",
    "seresnet50": "seresnet50",       # ResNet + squeeze-and-excitation
    "efficientnet_b0": "efficientnet_b0",
}


def get_dataloaders(data_dir, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(data_dir) / "valid", transform=eval_tf)
    test_ds = datasets.ImageFolder(Path(data_dir) / "test", transform=eval_tf)

    num_workers = 0 if os.name == "nt" else 2
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers),
        train_ds.classes,
    )


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def evaluate_test(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nTest precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(class_names)
    print(cm)
    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="seresnet50", choices=MODEL_MAP.keys())
    parser.add_argument("--data", default="data/classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data, args.batch)

    model = timm.create_model(
        MODEL_MAP[args.model], pretrained=True, num_classes=len(class_names)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{args.model}] epoch {epoch+1}/{args.epochs} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"val_loss={val_loss:.4f} ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_{args.model}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(f"best_{args.model}.pt"))
    test_metrics = evaluate_test(model, test_loader, device, class_names)

    with open(f"results_{args.model}.json", "w") as f:
        json.dump(
            {"history": history, "test_metrics": test_metrics, "class_names": class_names},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
