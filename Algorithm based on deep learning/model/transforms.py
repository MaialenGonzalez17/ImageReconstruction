import albumentations as A
import cv2
import random
import numpy as np
import torch

# Albumentations transforms with fixed parameters for reproducibility
alb_transforms = [
    ("RandomBrightnessContrast", A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=1.0)),
    ("GaussianBlur", A.Blur(blur_limit=(5,7), p=1.0)),
    ("GaussNoise", A.GaussNoise(std_range=(0.05, 0.1), p=1.0)),
    ("MotionBlur", A.MotionBlur(blur_limit=(5,7), p=1.0))
]

def add_uneven_illumination(image, intensity=None, center=None, sigma=None):
    """
    Adds uneven illumination to the image simulating non-uniform lighting conditions.
    Parameters can be optionally specified or randomized.
    """
    if intensity is None:
        intensity = random.uniform(0.5, 1.5)
    if center is None:
        h, w = image.shape[:2]
        center = (
            random.randint(0, w),
            random.randint(0, h)
        )
    if sigma is None:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        sigma = random.uniform(max_dim * 0.1, max_dim * 0.3)  # Spread of illumination

    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]

    # Create Gaussian illumination pattern
    gaussian = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * sigma ** 2))

    # Normalize and adjust intensity
    gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
    illumination = 1 + (gaussian * intensity)

    # Ensure minimum illumination level to avoid very dark regions
    min_illumination = 0.6
    illumination = min_illumination + (1 - min_illumination) * illumination

    # Apply illumination softly on the image
    result = image.copy().astype(float)
    result = result * illumination[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def add_random_shadows(image, num_shadows=None, shadow_intensity=None):
    """
    Adds random shadows to the image. The shadows can be linear or radial gradients.
    """
    if num_shadows is None:
        num_shadows = random.randint(1, 3)
    if shadow_intensity is None:
        shadow_intensity = random.uniform(0.4, 0.8)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_shadows):
        gradient_type = random.choice(['linear', 'radial'])

        if gradient_type == 'linear':
            # Create linear gradient shadow
            angle = random.uniform(0, 2 * np.pi)
            x = np.linspace(0, w - 1, w)
            y = np.linspace(0, h - 1, h)
            X, Y = np.meshgrid(x, y)

            gradient = (X * np.cos(angle) + Y * np.sin(angle)) / np.sqrt(w ** 2 + h ** 2)
            gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
            gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=w // 8)

        else:
            # Create radial gradient shadow
            center = (
                random.randint(-w // 2, w * 3 // 2),
                random.randint(-h // 2, h * 3 // 2)
            )
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

            max_dist = np.sqrt(w ** 2 + h ** 2)
            gradient = dist_from_center / max_dist
            gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=w // 8)

        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        shadow_mask = 1 - (gradient * (1 - shadow_intensity))

        # Apply shadow softly on all channels
        result = result * shadow_mask[:, :, np.newaxis]

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def add_elliptical_reflections(image, num_reflections=None, intensity=None):
    """
    Adds elliptical reflections simulating lens flares or light artifacts.
    """
    if num_reflections is None:
        num_reflections = random.randint(1, 3)
    if intensity is None:
        intensity = random.uniform(0.3, 0.7)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_reflections):
        reflection_mask = np.zeros((h, w), dtype=np.float32)
        center = (random.randint(0, w), random.randint(0, h))
        major_axis = random.randint(w // 20, w // 5)
        minor_axis = random.randint(w // 30, major_axis)
        angle = random.uniform(0, 180)

        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(temp_mask, center, (major_axis, minor_axis),
                    angle, 0, 360, 255, -1)

        reflection_mask = temp_mask.astype(float) / 255.0
        sigma = random.uniform(5, 15)
        reflection_mask = cv2.GaussianBlur(reflection_mask, (0, 0), sigma)
        if reflection_mask.max() > 0:
            reflection_mask /= reflection_mask.max()

        reflection_color = np.array([random.uniform(0.8, 1.0) for _ in range(3)])
        for c in range(3):
            channel = result[:, :, c]
            reflection_contribution = (255 - channel) * reflection_mask * intensity * reflection_color[c]
            channel += reflection_contribution
            result[:, :, c] = channel

    return np.clip(result, 0, 255).astype(np.uint8)

def add_streak_reflections(image, num_reflections=None, intensity=None):
    """
    Adds streak-like reflections simulating scratches or light streaks.
    """
    if num_reflections is None:
        num_reflections = random.randint(1, 3)
    if intensity is None:
        intensity = random.uniform(0.3, 0.7)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_reflections):
        reflection_mask = np.zeros((h, w), dtype=np.float32)

        # Define control points for a quadratic Bezier curve
        start_point = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))
        control_point = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))
        end_point = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))

        pts = np.array([start_point, control_point, end_point], dtype=np.float32)
        t_vals = np.linspace(0, 1, 100)
        curve_points = np.array([
            (1 - t)**2 * pts[0] + 2 * (1 - t) * t * pts[1] + t**2 * pts[2]
            for t in t_vals
        ], dtype=np.int32)

        temp_mask = np.zeros((h, w), dtype=np.uint8)
        thickness = random.randint(5, 20)
        cv2.polylines(temp_mask, [curve_points], False, 255, thickness)

        reflection_mask = temp_mask.astype(float) / 255.0
        sigma = random.uniform(5, 15)
        reflection_mask = cv2.GaussianBlur(reflection_mask, (0, 0), sigma)
        if reflection_mask.max() > 0:
            reflection_mask /= reflection_mask.max()

        reflection_color = np.array([random.uniform(0.8, 1.0) for _ in range(3)])
        for c in range(3):
            channel = result[:, :, c]
            reflection_contribution = (255 - channel) * reflection_mask * intensity * reflection_color[c]
            channel += reflection_contribution
            result[:, :, c] = channel

    return np.clip(result, 0, 255).astype(np.uint8)

# Custom transformations list combining Albumentations and custom ones
custom_transforms = [
    ("UnevenIllumination", add_uneven_illumination),
    ("RandomShadows", add_random_shadows),
    ("StreakReflections", add_streak_reflections),
    ("EllipticalReflections", add_elliptical_reflections),
]

# All transformations available for augmentation
all_transforms = alb_transforms + custom_transforms

# Initial weights for each transformation (can be updated dynamically here)
transform_weights = [1.0] * len(all_transforms)
# transform_weights[-1] += 0.2
