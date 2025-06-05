import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import ImageReconstructionDataset, MSE_SSIM_Loss
from model import ConvAutoencoder
from utils import set_seeds
from transforms import all_transforms, transform_weights
import os
import pandas as pd

def train(
    image_dir,               # Path to the folder with original images (fill in)
    epochs=30,
    batch_size=16,
    learning_rate=1e-4,
    val_ratio=0.15,
    save_path="weights/"    # Folder to save weights and checkpoints (change if desired)
):
    set_seeds(42)

    # List and sort image files
    files = sorted(os.listdir(image_dir))
    indices = list(range(len(files)))

    # Split indices into training and validation
    val_size = int(val_ratio * len(indices))
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]

    train_transform = None  # Transformations applied inside dataset (ToTensor included)
    val_transform = None

    # Create datasets
    train_dataset = ImageReconstructionDataset(
        image_dir=image_dir,
        indices=train_indices,
        data_type='training',
        transform=train_transform,
        weights=transform_weights.copy(),
    )

    val_dataset = ImageReconstructionDataset(
        image_dir=image_dir,
        indices=val_indices,
        data_type='validation',
        transform=val_transform,
        weights=[1] * len(all_transforms),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder().to(device)
    criterion = MSE_SSIM_Loss(alpha=0.84)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    weights_history = []

    # Create folder for saving weights/checkpoints if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, "best_autoencoder.pth")
    best_val_loss = float("inf")
    start_epoch = 0

    # Load checkpoint if exists (path customizable)
    checkpoint_path = os.path.join(save_path, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Last epoch: {start_epoch}, Best val_loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []
        for inputs, targets, picked_idxs in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        # Dynamically update transform weights
        train_dataset.update_weights([avg_train_loss] * len(transform_weights))

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        weights_history.append(train_dataset.weights.copy())

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_loss = {best_val_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, checkpoint_path)

    # Save weights history (path customizable)
    df_weights = pd.DataFrame(weights_history, columns=[name for name, _ in all_transforms])
    df_weights.to_excel(os.path.join(save_path, "weights_history.xlsx"), index=False)
    print("Training completed. Best model saved at:", best_model_path)


if __name__ == "__main__":
    train(
        image_dir="PATH/TO/YOUR/IMAGES",      # <--- FILL IN with your local path to images
        epochs=30,
        batch_size=4,
        learning_rate=1e-4,
        val_ratio=0.15,
        save_path="weights"                   # <--- Change if you want to save elsewhere
    )
