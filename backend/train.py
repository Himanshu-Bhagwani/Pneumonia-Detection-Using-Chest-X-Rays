import torch
import torch.optim as optim
import torch.nn as nn
# import torch.nn.functional as F # Not strictly needed now
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
# from sklearn.utils.class_weight import compute_class_weight # Using manual weights now
import numpy as np
import os
import sys
import time
# Need CosineAnnealingLR and also LinearLR for warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))
from levit_model import LeViT # Assumes levit_model.py has dropout

# --- Configuration ---
# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 80           # Max epochs
LEARNING_RATE = 3e-4  # Keep intermediate LR
WARMUP_EPOCHS = 5     # Keep warmup
WEIGHT_DECAY = 0.05   # Keep moderate WD
# LABEL_SMOOTHING = 0.1 # <<< REMOVED Label Smoothing to isolate weight effect
NUM_LOADER_WORKERS = 0
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 20 # Keep increased patience
GRADIENT_CLIP_VAL = 1.0
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42
# <<< MANUAL CLASS WEIGHTS - INCREASED NORMAL WEIGHT SIGNIFICANTLY >>>
# Goal: Penalize misclassifying NORMAL (index 0) much more heavily to reduce False Positives.
# Previous attempt might have been [2.5, 1.0] or balanced weights. Now setting explicitly high weight for Normal.
# Ratio NORMAL:PNEUMONIA. Try 5:1 or potentially higher if FPs persist.
MANUAL_CLASS_WEIGHTS = [5.0, 1.0] # Weight for [NORMAL, PNEUMONIA]. Adjust NORMAL weight (first value) as needed.

# Paths
script_dir = os.path.dirname(__file__)
model_dir = os.path.join(script_dir, "..", "model")
FULL_TRAIN_DATA_PATH = "/Users/home/Documents/minor/chest_xray/train"
model_save_path = os.path.join(model_dir, "levit_pneumonia_best.pth") # Keep requested name

# --- Device Setup ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")
print(f"Using device: {device}")

# --- Transformations ---
# Keep ImageNet stats and refined augmentation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)), # Moderate affine
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(), normalize])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])

# --- Function Definitions (train_one_epoch, evaluate - unchanged structurally) ---
def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs, grad_clip_val):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    start_time = time.time()
    print(f"Epoch {epoch}/{total_epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) # Loss uses the heavily weighted criterion
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    # Scheduler stepped outside
    epoch_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
    epoch_train_acc = correct_train / total_train if total_train > 0 else 0.0
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch}/{total_epochs} Train Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_train_acc:.4f}, Time: {epoch_time:.2f}s")
    return epoch_loss, epoch_train_acc

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    correct, total, total_val_loss = 0, 0, 0.0
    start_time = time.time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels) # Evaluate loss with same weights for consistency
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

    # Data Loading and Splitting (Unchanged)
    if not os.path.isdir(FULL_TRAIN_DATA_PATH): sys.exit(f"ERROR: Full training data directory not found: {FULL_TRAIN_DATA_PATH}")
    print(f"Loading full dataset info from: {FULL_TRAIN_DATA_PATH}")
    full_dataset_info = datasets.ImageFolder(root=FULL_TRAIN_DATA_PATH)
    num_total = len(full_dataset_info); class_names = full_dataset_info.classes; num_classes = len(class_names)
    print(f"Classes: {class_names}") # Should be ['NORMAL', 'PNEUMONIA']
    num_val = int(VALIDATION_SPLIT * num_total); num_train = num_total - num_val
    print(f"Splitting dataset: {num_train} training samples, {num_val} validation samples.")
    print("Loading datasets with appropriate transforms...")
    train_dataset_reloaded = datasets.ImageFolder(root=FULL_TRAIN_DATA_PATH, transform=train_transform)
    val_dataset_reloaded = datasets.ImageFolder(root=FULL_TRAIN_DATA_PATH, transform=val_transform)
    indices = list(range(num_total)); np.random.shuffle(indices)
    train_indices, val_indices = indices[:num_train], indices[num_train:]
    train_data = torch.utils.data.Subset(train_dataset_reloaded, train_indices)
    val_data = torch.utils.data.Subset(val_dataset_reloaded, val_indices)
    print("Created training and validation subsets.")
    print(f"Training dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(val_data)}")

    # --- Use MANUAL Class Weights ---
    print(f"Using MANUAL class weights: {MANUAL_CLASS_WEIGHTS}")
    # Assign directly, assuming order matches ['NORMAL', 'PNEUMONIA']
    try:
        class_weights_tensor = torch.tensor(MANUAL_CLASS_WEIGHTS, dtype=torch.float).to(device)
    except Exception as e:
        print(f"Error creating weight tensor: {e}. Exiting.")
        sys.exit(1)

    # DataLoaders
    print(f"Using {NUM_LOADER_WORKERS} workers for DataLoaders.")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOADER_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_LOADER_WORKERS, pin_memory=False)

    # Model, Loss, Optimizer, Scheduler
    print(f"Initializing LeViT model with {num_classes} classes and Dropout: {DROPOUT_RATE}")
    model = LeViT(num_classes=num_classes, img_size=IMG_SIZE, dropout_rate=DROPOUT_RATE).to(device)

    # <<< Using Weighted CrossEntropyLoss WITHOUT Label Smoothing >>>
    if class_weights_tensor is not None:
        print(f"Using Weighted CrossEntropyLoss with MANUAL weights.")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # Pass manual weights
    else:
        # Fallback, should not happen with manual weights defined
        print(f"Warning: Manual weights not set. Using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    print(f"Using AdamW optimizer with LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Setup Warmup + Cosine Annealing Scheduler (Unchanged)
    print(f"Using Linear Warmup for {WARMUP_EPOCHS} epochs + Cosine Annealing for {EPOCHS - WARMUP_EPOCHS} epochs.")
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS])

    # --- Training Loop ---
    print(f"\n--- Starting Training for up to {EPOCHS} epochs (Patience: {EARLY_STOPPING_PATIENCE}, Grad Clip: {GRADIENT_CLIP_VAL}) ---")
    best_val_acc = 0.0
    best_val_loss = float('inf') # Track best loss for tie-breaking
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, EPOCHS, GRADIENT_CLIP_VAL)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step() # Step combined scheduler

        # Save based on best VALIDATION ACCURACY, tie-breaking with loss
        save_model = False
        if val_acc > best_val_acc:
            save_model = True
            print(f"Validation accuracy improved ({best_val_acc:.4f} --> {val_acc:.4f}).")
        elif val_acc == best_val_acc:
            if val_loss < best_val_loss:
                save_model = True
                print(f"Validation accuracy ({val_acc:.4f}) same as best, but loss improved ({best_val_loss:.4f} --> {val_loss:.4f}).")
            # else:
            #      print(f"Validation accuracy ({val_acc:.4f}) same as best, but loss did not improve ({best_val_loss:.4f} vs {val_loss:.4f}). Not saving.")

        if save_model:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), model_save_path)
                print(f"*** Best model saved to {model_save_path} (Epoch: {epoch}, Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f}) ***")
            except Exception as e:
                 print(f"Error saving model checkpoint: {e}")
        else:
            # Only increment counter if accuracy didn't strictly improve
            # This logic might be too strict if accuracy fluctuates slightly but loss improves over time.
            # Let's increment only if accuracy is strictly worse than best *or* same with worse loss (implicitly handled above)
             if val_acc < best_val_acc:
                 epochs_no_improve += 1
                 print(f"Validation accuracy did not improve for {epochs_no_improve} epoch(s).")
             elif val_acc == best_val_acc and val_loss > best_val_loss:
                  # Also count stagnation if acc is same but loss gets worse
                  epochs_no_improve += 1
                  print(f"Validation accuracy did not improve (stagnant loss) for {epochs_no_improve} epoch(s).")


        # Early stopping check
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print("--- Training Finished ---")
    if epochs_no_improve < EARLY_STOPPING_PATIENCE: print(f"Completed all {EPOCHS} epochs.")
    print(f"Best Validation Accuracy achieved (on new split): {best_val_acc:.4f} (corresponding loss: {best_val_loss:.4f})")
    print(f"Best model state dict saved at: {model_save_path}")