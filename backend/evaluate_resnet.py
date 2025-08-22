# backend/evaluate_resnet.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models # Import models
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_LOADER_WORKERS = 0

# Paths
script_dir = os.path.dirname(__file__)
# <<< Point to the saved ResNet50 model >>>
MODEL_PATH = os.path.join(script_dir, "..", "model", "resnet50_pneumonia_best.pth")
TEST_DATA_PATH = "/Users/home/Documents/minor/chest_xray/test"

# --- Device Setup ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
     device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Transformations (MUST match training validation/ResNet requirements) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

def evaluate_on_test_set(model_path, test_data_path, transform, batch_size, num_workers, device):
    """Loads a ResNet50 model and evaluates it on the test dataset."""

    # --- 1. Load Test Data ---
    if not os.path.isdir(test_data_path): sys.exit(f"ERROR: Test data directory not found: {test_data_path}")
    print(f"Loading test data from: {test_data_path}")
    test_data = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    class_names = test_data.classes
    num_classes = len(class_names)
    print(f"Test dataset classes: {class_names}")

    # --- 2. Load ResNet50 Model Structure ---
    if not os.path.exists(model_path): sys.exit(f"ERROR: Model file not found at {model_path}")

    print("Initializing ResNet50 model structure...")
    # Instantiate the ResNet50 architecture first
    model = models.resnet50(weights=None) # No pre-trained weights needed here
    # Modify the head to match the number of classes it was trained for
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device) # Move structure to device

    # --- Load Saved Weights ---
    print(f"Loading fine-tuned weights from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        sys.exit(f"Error loading model weights: {e}\nEnsure the saved weights are for a ResNet50 with {num_classes} output classes.")

    # --- 3. Evaluation Loop ---
    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()
    print("Starting evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    eval_time = time.time() - start_time
    print(f"Evaluation finished in {eval_time:.2f}s.")
    if not all_labels or not all_preds: sys.exit("Error: No labels or predictions collected.")
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)

    # --- 4. Calculate & Print Metrics ---
    print("\n--- Evaluation Metrics (ResNet50) ---")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # --- 5. Plot Confusion Matrix ---
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('ResNet50 Confusion Matrix on Test Set')
        plt.tight_layout()
        plt.show() # Display plot
        # plt.savefig('confusion_matrix_resnet50_test.png') # Optional save
    except Exception as e:
        print(f"\nWarning: Could not plot confusion matrix. Error: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    evaluate_on_test_set(
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        transform=eval_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_LOADER_WORKERS,
        device=device
    )