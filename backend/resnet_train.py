import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models # Import torchvision models
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import sys
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Configuration ---
# Hyperparameters for Transfer Learning (Head Tuning Phase)
IMG_SIZE = 224        # ResNet usually uses 224
BATCH_SIZE = 32
EPOCHS = 25           # Usually fewer epochs needed for head tuning vs full training
LEARNING_RATE = 0.001 # Can often start slightly higher for head tuning
WEIGHT_DECAY = 0.01   # Lower WD initially as most weights are frozen
LABEL_SMOOTHING = 0.1 # Optional, can keep or remove for head tuning
NUM_LOADER_WORKERS = 0
EARLY_STOPPING_PATIENCE = 7 # Can use shorter patience for head tuning

# Paths
script_dir = os.path.dirname(__file__)
model_dir = os.path.join(script_dir, "..", "model")
FULL_TRAIN_DATA_PATH = "/Users/home/Documents/minor/chest_xray/train"
# <<< Save ResNet model with a distinct name >>>
model_save_path = os.path.join(model_dir, "resnet50_pneumonia_best.pth")
VALIDATION_SPLIT = 0.15 # <<< Use the same split ratio as LeViT training
RANDOM_SEED = 42      # <<< Use the same seed for a comparable split

# --- Device Setup ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
     device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Transformations ---
# <<< MUST use ImageNet normalization for pre-trained ResNet >>>
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Moderate augmentation for training phase (similar to LeViT's successful run is fine)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize
])

# Simple validation transform
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize to 224x224
    transforms.ToTensor(),
    normalize
])

# --- Function Definitions ---
def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch, total_epochs):
    """Trains the model for one epoch (head only)."""
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    start_time = time.time()
    print(f"Epoch {epoch}/{total_epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # Gradients only calculated for unfrozen (head) params
        optimizer.step() # Updates only head params
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    scheduler.step()
    epoch_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
    epoch_train_acc = correct_train / total_train if total_train > 0 else 0.0
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch}/{total_epochs} Train Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_train_acc:.4f}, Time: {epoch_time:.2f}s")
    return epoch_loss, epoch_train_acc

@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluates the model."""
    model.eval()
    correct, total, total_val_loss = 0, 0, 0.0
    start_time = time.time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    avg_val_loss = total_val_loss / len(loader) if len(loader) > 0 else 0.0
    epoch_time = time.time() - start_time
    print(f"Validation Summary: Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}, Time: {epoch_time:.2f}s")
    return avg_val_loss, accuracy
# --- End Function Definitions ---


# --- Main Execution Block ---
if __name__ == '__main__':

    # Setup
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Data Loading and Splitting (using the same logic as LeViT script)
    if not os.path.isdir(FULL_TRAIN_DATA_PATH): sys.exit(f"ERROR: Full training data directory not found: {FULL_TRAIN_DATA_PATH}")
    print(f"Loading full dataset info from: {FULL_TRAIN_DATA_PATH}")
    full_dataset_info = datasets.ImageFolder(root=FULL_TRAIN_DATA_PATH)
    num_total = len(full_dataset_info)
    class_names = full_dataset_info.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    num_val = int(VALIDATION_SPLIT * num_total)
    num_train = num_total - num_val
    print(f"Splitting dataset: {num_train} training samples, {num_val} validation samples using seed {RANDOM_SEED}.")

    print("Loading datasets with appropriate transforms...")
    train_dataset_reloaded = datasets.ImageFolder(root=FULL_TRAIN_DATA_PATH, transform=train_transform)
    val_dataset_reloaded = datasets.ImageFolder(root=FULL_TRAIN_DATA_PATH, transform=val_transform)
    indices = list(range(num_total))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:num_train], indices[num_train:]
    train_data = torch.utils.data.Subset(train_dataset_reloaded, train_indices)
    val_data = torch.utils.data.Subset(val_dataset_reloaded, val_indices)
    print("Created training and validation subsets.")
    print(f"Training dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(val_data)}")

    # Calculate Class Weights
    print("Calculating class weights from training subset...")
    try:
        train_subset_labels = [full_dataset_info.targets[i] for i in train_indices]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_subset_labels), y=train_subset_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"Class weights calculated: {class_weights_tensor}")
    except Exception as e:
        print(f"Error calculating class weights: {e}. Using default unweighted loss.")
        class_weights_tensor = None

    # DataLoaders
    print(f"Using {NUM_LOADER_WORKERS} workers for DataLoaders.")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOADER_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_LOADER_WORKERS, pin_memory=False)

    # --- Model Initialization (Transfer Learning with ResNet50) ---
    print("Initializing pre-trained ResNet50 model...")
    weights = models.ResNet50_Weights.IMAGENET1K_V2 # Use modern weights API
    model = models.resnet50(weights=weights)

    # Freeze backbone layers
    print("Freezing backbone layers...")
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # New head is trainable by default
    print(f"Replaced final layer with new head for {num_classes} classes.")

    model = model.to(device)

    # --- Loss Function (Weighted CE + Smoothing) ---
    if class_weights_tensor is not None:
        print(f"Using Weighted CrossEntropyLoss with label smoothing: {LABEL_SMOOTHING}")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
    else:
        print(f"Using standard CrossEntropyLoss with label smoothing: {LABEL_SMOOTHING}")
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # --- Optimizer (Optimize ONLY the new head) ---
    print(f"Setting up AdamW optimizer for the classifier head (FC layer) only. LR: {LEARNING_RATE}, WD: {WEIGHT_DECAY}")
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Scheduler ---
    print(f"Using CosineAnnealingLR scheduler for {EPOCHS} epochs.")
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01)

    # --- Training Loop ---
    print(f"\n--- Starting Transfer Learning (Training Head) for up to {EPOCHS} epochs ---")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Train only the head
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, EPOCHS)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), model_save_path) # Save fine-tuned model
                print(f"*** Best ResNet50 model saved to {model_save_path} (Epoch: {epoch}, Val Acc: {best_val_acc:.4f}) ***")
            except Exception as e:
                 print(f"Error saving model checkpoint: {e}")
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print("--- Head Training Finished ---")
    if epochs_no_improve < EARLY_STOPPING_PATIENCE: print(f"Completed all {EPOCHS} epochs.")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.4f}")
    print(f"Best ResNet50 model state dict saved at: {model_save_path}")