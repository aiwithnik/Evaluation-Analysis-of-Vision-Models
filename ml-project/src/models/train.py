import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.models.dataset_loader import TensorDataset
from src.models.model import build_efficientnet
from src.models.registry import save_model


def train_model(
    train_dir,
    val_dir,
    num_classes,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    device,
):
    # Datasets
    train_ds = TensorDataset(train_dir)
    val_ds = TensorDataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = build_efficientnet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device).float()
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

    # Save model to registry
    metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }

    version = save_model(model, metadata)
    print(f"Model saved as registry version: {version}")

    return model
