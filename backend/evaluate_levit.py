import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time

# Add backend directory to path to import the model
sys.path.append(os.path.dirname(__file__))
try:
    from levit_model import LeViT
except ImportError:
    print("Error: Could not import LeViT model.")
    print("Ensure levit_model.py is in the same directory or sys.path is set correctly.")
    sys.exit(1)

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32 # Adjust if needed, but usually fine for evaluation
NUM_LOADER_WORKERS = 0 # Typically 0 for evaluation

# Paths
script_dir = os.path.dirname(__file__)
# *** Path to the saved model state_dict ***
MODEL_PATH = os.path.join(script_dir, "..", "model", "levit_pneumonia_best.pth")
# *** Path to the TEST dataset ***
TEST_DATA_PATH = "/Users/home/Documents/minor/chest_xray/test"

# --- Device Setup ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
     device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Transformations (Should match validation transforms used during training) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

def evaluate_on_test_set(model_path, test_data_path, transform, batch_size, num_workers, device):
    """Loads a model and evaluates it on the test dataset."""

    # --- 1. Load Test Data ---
    if not os.path.isdir(test_data_path):
        print(f"ERROR: Test data directory not found at {test_data_path}")
        return
    print(f"Loading test data from: {test_data_path}")
    test_data = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    class_names = test_data.classes
    num_classes = len(class_names)
    print(f"Test dataset classes: {class_names}")

    # --- 2. Load Model ---
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    # Instantiate the model structure (must match the saved model's structure)
    # Assuming default dropout rate was used if not explicitly saved in checkpoint
    # If you saved dropout_rate in checkpoint, load it and pass it here.
    model = LeViT(num_classes=num_classes, img_size=IMG_SIZE).to(device)
    print(f"Loading model weights from: {model_path}")

    try:
        # Load the saved state dictionary
        checkpoint = torch.load(model_path, map_location=device)

        # Check if the loaded object is a dictionary (from saving state + extras)
        # or just the state_dict itself
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Loaded model state_dict from checkpoint dictionary.")
        else:
            # Assume the loaded object *is* the state_dict
            state_dict = checkpoint
            print("Loaded raw model state_dict.")

        # Load the state dictionary into the model
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")

    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- 3. Evaluation Loop ---
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    all_preds = []
    all_labels = []
    start_time = time.time()
    print("Starting evaluation on test set...")

    with torch.no_grad(): # Disable gradient calculations
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) # Get model predictions
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability

            all_preds.extend(predicted.cpu().numpy()) # Store predictions
            all_labels.extend(labels.cpu().numpy())   # Store true labels

    eval_time = time.time() - start_time
    print(f"Evaluation finished in {eval_time:.2f}s.")

    # Ensure predictions and labels were collected
    if not all_labels or not all_preds:
        print("Error: No labels or predictions were collected during evaluation.")
        return

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # --- 4. Calculate Metrics ---
    print("\n--- Evaluation Metrics ---")

    # Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Classification Report (Precision, Recall, F1-Score per class)
    print("\nClassification Report:")
    # Use target_names for readable labels in the report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # --- 5. Plot Confusion Matrix ---
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix on Test Set')
        # You can save the plot instead of showing it if running non-interactively
        # plt.savefig('confusion_matrix_test.png')
        # print("Confusion matrix saved as confusion_matrix_test.png")
        plt.show() # Display the plot
    except Exception as e:
        print(f"\nWarning: Could not plot confusion matrix. Error: {e}")
        print("Ensure matplotlib and seaborn are installed.")

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