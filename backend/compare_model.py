import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models # For ResNet50
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt # <<< Added for plotting
import seaborn as sns
import os
import sys
import time
import pandas as pd # For comparison table and easier plotting data prep

# Add backend directory to path to import the LeViT model
sys.path.append(os.path.dirname(__file__))
try:
    from levit_model import LeViT
except ImportError:
    print("Error: Could not import LeViT model.")
    print("Ensure levit_model.py is in the same directory or sys.path is set correctly.")
    sys.exit(1)

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 64 # Can increase batch size for evaluation if memory allows
NUM_LOADER_WORKERS = 0

# Paths
script_dir = os.path.dirname(__file__)
model_dir = os.path.join(script_dir, "..", "model")
LEVIT_MODEL_PATH = os.path.join(model_dir, "levit_pneumonia_best.pth")
RESNET_MODEL_PATH = os.path.join(model_dir, "resnet50_pneumonia_best.pth")
TEST_DATA_PATH = "/Users/home/Documents/minor/chest_xray/test"
# <<< Directory to save plots >>>
PLOT_SAVE_DIR = os.path.join(script_dir, "..", "comparison_plots")

# --- Device Setup ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
     device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Transformations ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

# --- Helper Functions ---

def load_model(model_type, model_path, num_classes, device):
    """Loads either LeViT or ResNet model structure and weights."""
    print(f"\n--- Loading {model_type} Model ---")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None
    if model_type.lower() == 'levit':
        print("Initializing LeViT model structure...")
        model = LeViT(num_classes=num_classes, img_size=IMG_SIZE, dropout_rate=0.5)
    elif model_type.lower() == 'resnet50':
        print("Initializing ResNet50 model structure...")
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        print(f"ERROR: Unknown model type '{model_type}'")
        return None
    model = model.to(device)
    print(f"Loading trained weights from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

@torch.no_grad()
def run_evaluation(model, loader, device):
    """Runs inference and returns predictions and labels."""
    model.eval()
    all_preds, all_labels = [], []
    print(f"Running evaluation for model: {model.__class__.__name__}...")
    start_time = time.time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    eval_time = time.time() - start_time
    print(f"Evaluation finished in {eval_time:.2f}s.")
    if not all_labels or not all_preds: return None, None
    return np.array(all_preds), np.array(all_labels)

def calculate_and_print_metrics(model_name, y_true, y_pred, class_names):
    """Calculates, prints, and returns key metrics."""
    print(f"\n--- Evaluation Metrics ({model_name}) ---")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = {'accuracy': accuracy}
    for i, name in enumerate(class_names):
         metrics[f'{name}_precision'] = report_dict[name]['precision']
         metrics[f'{name}_recall'] = report_dict[name]['recall']
         metrics[f'{name}_f1-score'] = report_dict[name]['f1-score']
    return cm, metrics

# --- Plotting Functions ---

def plot_metric_comparison_bars(levit_metrics, resnet_metrics, class_names, save_dir):
    """Creates a bar chart comparing key metrics."""
    if not levit_metrics or not resnet_metrics:
        print("Skipping metric comparison plot: Metrics missing for one or both models.")
        return

    metrics_to_plot = ['accuracy'] + \
                      [f'{name}_{m}' for name in class_names for m in ['precision', 'recall', 'f1-score']]
    labels = [m.replace('_', ' ').replace('f1-score', 'F1') for m in metrics_to_plot] # Nicer labels
    levit_values = [levit_metrics.get(m, 0) for m in metrics_to_plot] # Default to 0 if missing
    resnet_values = [resnet_metrics.get(m, 0) for m in metrics_to_plot]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, levit_values, width, label='LeViT', color='skyblue')
    rects2 = ax.bar(x + width/2, resnet_values, width, label='ResNet50', color='lightcoral')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05) # Scores are between 0 and 1

    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=8)

    fig.tight_layout()
    save_filename = os.path.join(save_dir, 'metric_comparison_bars.png')
    plt.savefig(save_filename)
    print(f"Metric comparison bar chart saved as {save_filename}")
    # plt.show()
    plt.close(fig) # Close the figure to free memory

def plot_side_by_side_cms(levit_cm, resnet_cm, class_names, save_dir):
    """Plots both confusion matrices side-by-side."""
    if levit_cm is None or resnet_cm is None:
        print("Skipping side-by-side confusion matrix plot: CM missing for one or both models.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # 1 row, 2 columns

    sns.heatmap(levit_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('LeViT Confusion Matrix')

    sns.heatmap(resnet_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_title('ResNet50 Confusion Matrix')

    plt.suptitle('Confusion Matrices on Test Set', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    save_filename = os.path.join(save_dir, 'confusion_matrices_comparison.png')
    plt.savefig(save_filename)
    print(f"Side-by-side confusion matrix plot saved as {save_filename}")
    # plt.show()
    plt.close(fig) # Close the figure


# --- Main Comparison Execution ---
if __name__ == '__main__':

    # Ensure plot directory exists
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    # Load Test Data
    if not os.path.isdir(TEST_DATA_PATH): sys.exit(f"ERROR: Test data directory not found: {TEST_DATA_PATH}")
    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_data = datasets.ImageFolder(root=TEST_DATA_PATH, transform=eval_transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_LOADER_WORKERS, pin_memory=False)
    class_names = test_data.classes
    num_classes = len(class_names)
    if num_classes != 2: sys.exit("Error: Expected 2 classes.")
    print(f"Test dataset classes: {class_names}")
    print(f"Test dataset size: {len(test_data)}")

    # Evaluate LeViT
    levit_model = load_model('levit', LEVIT_MODEL_PATH, num_classes, device)
    levit_preds, levit_labels = None, None
    levit_cm, levit_metrics = None, None
    if levit_model:
        levit_preds, levit_labels = run_evaluation(levit_model, test_loader, device)
        if levit_preds is not None:
            levit_cm, levit_metrics = calculate_and_print_metrics("LeViT", levit_labels, levit_preds, class_names)

    # Evaluate ResNet50
    resnet_model = load_model('resnet50', RESNET_MODEL_PATH, num_classes, device)
    resnet_preds, resnet_labels = None, None
    resnet_cm, resnet_metrics = None, None
    if resnet_model:
        resnet_preds, resnet_labels = run_evaluation(resnet_model, test_loader, device)
        if resnet_preds is not None:
            if levit_labels is not None and not np.array_equal(levit_labels, resnet_labels):
                print("WARNING: Labels obtained from test loader differ between runs!") # Should not happen with shuffle=False
            # Use resnet_labels for consistency in case LeViT failed
            eval_labels = resnet_labels if resnet_labels is not None else levit_labels
            if eval_labels is not None:
                 resnet_cm, resnet_metrics = calculate_and_print_metrics("ResNet50", eval_labels, resnet_preds, class_names)
            else:
                 print("Could not evaluate ResNet due to missing labels.")

    # --- Direct Comparison Summary & Plots ---
    print("\n--- Comparison Summary ---")
    if levit_metrics and resnet_metrics:
        # Create DataFrame comparison table
        comparison_data = { 'Metric': [], 'LeViT': [], 'ResNet50': [] }
        metric_order = ['accuracy'] + [f'{name}_{m}' for name in class_names for m in ['precision', 'recall', 'f1-score']]
        metric_labels = ['Accuracy'] + [f'{name} {m.replace("f1-score", "F1").capitalize()}' for name in class_names for m in ['precision', 'recall', 'f1-score']]

        for m, label in zip(metric_order, metric_labels):
             comparison_data['Metric'].append(label)
             comparison_data['LeViT'].append(levit_metrics.get(m, np.nan)) # Use NaN if missing
             comparison_data['ResNet50'].append(resnet_metrics.get(m, np.nan))

        df_compare = pd.DataFrame(comparison_data)
        print(df_compare.round(4).to_string(index=False)) # Print formatted table

        # Generate Plots
        plot_metric_comparison_bars(levit_metrics, resnet_metrics, class_names, PLOT_SAVE_DIR)
        plot_side_by_side_cms(levit_cm, resnet_cm, class_names, PLOT_SAVE_DIR)

    elif levit_metrics: print("\nComparison plots skipped: Only LeViT model evaluated.")
    elif resnet_metrics: print("\nComparison plots skipped: Only ResNet50 model evaluated.")
    else: print("\nComparison plots skipped: Neither model could be fully evaluated.")

    print("\nComparison Complete.")