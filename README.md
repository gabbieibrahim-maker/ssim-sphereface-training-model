# SSIM SphereFace Training Model

# Overview of Project
Training a deep learning model in PyTorch, utilizing SphereFace (angular margin loss) and Focal Loss to predict structural similarity (SSIM) artifact classes for MRI imaging.

# Features
- SphereFace Architecture: Angular margin loss to map images on a high-dimensional hypersphere.
- Focal Loss Integration (gamma=2): Emphasizes focus on rare artifact classes by increasing loss penalty, while reducing focus on more common training examples.
- Custom Data Loading: Supports balancing options (`anatomy` / `anatomy_artifact`) to manage multi-organ imaging distributions. Supports importing a fixed number of samples or all samples in dataset.
- Other Parameters: Supports changing the encoder (ViTs/CNNs), learning rate, distribution of classes, number of batches, number of epochs, optimizer, angular margin, penalty multiplier, loRARank, and number of workers.

# Technical Details
- Python
- PyTorch
- Weights & Biases (wandb) for experiment tracking
