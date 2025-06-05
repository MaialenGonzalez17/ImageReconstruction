import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import piq  # Perceptual Image Quality metrics library
from transforms import all_transforms, transform_weights  # Your custom transform list and weights
from utils import weighted_pick  # Your custom weighted sampling function

# Define a combined loss function that mixes MSE and SSIM losses for image reconstruction
class MSE_SSIM_Loss(nn.Module):
    def __init__(self, alpha=0.84):
        super(MSE_SSIM_Loss, self).__init__()
        self.alpha = alpha  # Weight factor to balance MSE and SSIM losses

    def forward(self, y_pred, y_true):
        # Clamp predictions and targets to [0, 1]
        y_pred = y_pred.clamp(0, 1)
        y_true = y_true.clamp(0, 1)

        mse_loss = F.mse_loss(y_pred, y_true)  # Mean Squared Error loss
        ssim_score = piq.ssim(y_pred, y_true, data_range=1.0)  # Structural Similarity Index Measure
        ssim_loss = 1.0 - ssim_score  # Convert similarity to loss

        # Combine the two losses weighted by alpha
        total_loss = self.alpha * mse_loss + (1.0 - self.alpha) * ssim_loss
        return total_loss

# Custom PyTorch Dataset for loading images and applying weighted random transformations
class ImageReconstructionDataset(Dataset):
    def __init__(
        self,
        image_dir: str,      # Path to the folder with input images (complete this)
        indices: list,       # List of indices to select specific files
        data_type: str = 'training',  # Dataset mode: 'training' or 'validation' etc.
        transform=None,      # Optional torchvision transforms applied after augmentations
        weights: list = None # Weights for weighted random picking of transformations
    ):
        self.epoch_counter = 0
        self.image_dir = image_dir
        self.indices = indices
        self.data_type = data_type
        self.transform = transform
        self.weights = weights or [1] * len(all_transforms)  # Default equal weights if none provided
        self.weights_history = []
        self.files = sorted(os.listdir(image_dir))  # List all files in the directory, sorted alphabetically

        self.all_transforms = all_transforms  # List of transformations to sample from
        self.stats = {name: 0 for name, _ in self.all_transforms}  # Track how often each transform is applied
        self.transform_losses = {name: [] for name, _ in self.all_transforms}  # Track losses per transform

        # Additional manual adjustments to specific transformation weights (indices depend on your list)
        transform_weights[-1] += 0.25
        transform_weights[-7] += 0.2
        transform_weights[-5] += 0.25
        transform_weights[-2] += 0.25
        transform_weights[-4] += 0.2

    def __len__(self):
        # Returns the number of samples in the dataset
        return len(self.indices)

    def update_weights(self, weight_dict: list):
        # Dynamically update transformation weights based on recent losses to focus on harder augmentations
        if not hasattr(self, 'weights_history'):
            self.weights_history = []

        self.weights_history.append(self.weights)
        avg_loss = sum(weight_dict) / len(weight_dict)

        initial_weight = 1
        max_weight = 1.8
        deltas = []
        total_delta = 0
        add_delta = [0] * len(weight_dict)

        for i, loss in enumerate(weight_dict):
            delta = max(loss - avg_loss, 0)
            deltas.append(delta)
            total_delta += delta

        if total_delta > 0:
            learning_rate = 0.5
            for i, delta in enumerate(deltas):
                if delta > 0:
                    adjustment = delta * learning_rate
                    add_delta[i] += adjustment

        updated_weights = [x + y for x, y in zip(self.weights, add_delta)]

        # Reset weights if any exceed max threshold, applying extra boosts to certain transforms
        if any(weight >= max_weight for weight in updated_weights):
            for i in range(len(updated_weights)):
                updated_weights[i] = initial_weight
            updated_weights[-1] += 0.15
            updated_weights[-1] += 0.25
            updated_weights[-7] += 0.2
            updated_weights[-5] += 0.25
            updated_weights[-2] += 0.25
            updated_weights[-4] += 0.2

        print(f"New weights: {updated_weights}")
        self.weights = updated_weights

    def __getitem__(self, idx):
        # Load image and apply weighted random transformations depending on dataset mode
        img_path = os.path.join(self.image_dir, self.files[self.indices[idx]])
        img = Image.open(img_path).convert('RGB').resize((320, 320))  # Load and resize image
        img_np = np.array(img)

        # Select transformations based on weights (2 for training, 1 for others)
        if self.data_type == 'training':
            k = 2
            selected, picked_idxs = weighted_pick(self.all_transforms, self.weights, k=k)
        else:
            k = 1
            selected, picked_idxs = weighted_pick(self.all_transforms, [1] * len(self.all_transforms), k=k)

        # Apply selected transformations (supporting albumentations or torchvision style)
        for name, tf in selected:
            if hasattr(tf, 'apply'):  # Albumentations style transform
                img_np = tf(image=img_np)['image']
            else:
                img_np = tf(img_np)

        img_pil = Image.fromarray(img_np)

        # Apply optional final transform or convert to tensor
        if self.transform:
            img_transformed = self.transform(img_pil)
            img_original = self.transform(img)
        else:
            img_transformed = transforms.ToTensor()(img_pil)
            img_original = transforms.ToTensor()(img)

        # Update stats on how many times each transform is used
        for name, _ in selected:
            self.stats[name] += 1

        # Return transformed image, original image, and indices of selected transforms
        return img_transformed, img_original, picked_idxs
