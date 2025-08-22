# Pneumonia Detection from Chest X-rays using LeViT and ResNet50

This project implements and evaluates a LeViT (Vision Transformer) model for detecting pneumonia from chest X-ray images. For comparison, a ResNet50 model using transfer learning is also trained and evaluated on the same dataset. The project includes a Streamlit web application for users to upload X-ray images and get predictions with attention map visualizations from the LeViT model.

A key finding is that while the pre-trained ResNet50 achieved higher overall accuracy, the custom LeViT model demonstrated superior recall (sensitivity) for detecting pneumonia cases, a critical metric in medical diagnosis.

## Project Structure
minor/
├── backend/
│ ├── levit_model.py # LeViT model architecture definition
│ ├── train.py # Script for training the LeViT model
│ ├── train_resnet.py # Script for training the ResNet50 model
│ ├── inference.py # Script for LeViT model inference and attention map generation
│ ├── evaluate_model.py # Script to evaluate the trained LeViT model on the test set
│ ├── evaluate_resnet.py # Script to evaluate the trained ResNet50 model on the test set
│ └── compare_models.py # Script to compare metrics and generate plots for both models
├── model/
│ ├── levit_pneumonia_best.pth # (Generated) Best weights for the LeViT model
│ └── resnet50_pneumonia_best.pth # (Generated) Best weights for the ResNet50 model
├── comparison_plots/ # (Generated) Directory for saved comparison plots
│ ├── metric_comparison_bars.png
│ └── confusion_matrices_comparison.png
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
└── README.md # This file

## Dataset

The project uses the **"Chest X-Ray Images (Pneumonia)"** dataset available on Kaggle:
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

This dataset contains 5,863 chest X-ray images categorized into 'NORMAL' and 'PNEUMONIA' classes, pre-split into training and test sets. For robust training and evaluation, the training scripts (`train.py` and `train_resnet.py`) create a new validation split (15%) from the original training data, ensuring reproducibility using a fixed random seed (42).

**Local Data Setup:**
It is assumed that the dataset is downloaded and structured locally as follows:
/Users/home/Documents/minor/chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── val/ (Original small validation set from Kaggle, not directly used by our training scripts)

Update `FULL_TRAIN_DATA_PATH` and `TEST_DATA_PATH` variables in the Python scripts if your dataset is located elsewhere.

## Models Implemented

### 1. LeViT (Vision Transformer)
A custom implementation of a LeViT-like model is provided in `backend/levit_model.py`. Key features include:
*   **Convolutional Patch Embedding:** Uses a `Conv2d` layer to efficiently convert image patches into token embeddings.
*   **Transformer Blocks:** Comprises standard Multi-Head Self-Attention (MHSA) and Feed-Forward Network (FFN) layers with Pre-Layer Normalization and residual connections.
*   **Global Average Pooling:** Applied after the Transformer blocks to get a fixed-size representation for classification.
*   **Dropout:** Included before the final classifier for regularization (dropout rate 0.5).
*   **Attention Map Output:** The model can output attention maps from the last Transformer block for visualization, providing insight into its decision-making process.

### 2. ResNet50 (Benchmark)
A pre-trained ResNet50 model from `torchvision.models` (using `IMAGENET1K_V2` weights) is used as a benchmark. Transfer learning is applied:
*   The backbone convolutional layers are frozen.
*   The final fully connected layer is replaced with a new one suited for 2-class (Normal/Pneumonia) classification.
*   Only this new head is trained initially.

## Setup and Installation

1.  **Clone the repository (if applicable) or create the project structure.**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, and `streamlit`.

## Training the Models

The training scripts utilize a reproducible validation split from the main training data, weighted loss (to handle class imbalance), label smoothing, Cosine Annealing learning rate schedule, AdamW optimizer, and early stopping. LeViT training also includes gradient clipping.

1.  **Train LeViT Model:**
    *   Ensure `FULL_TRAIN_DATA_PATH` in `backend/train.py` points to your training data.
    *   Run the script from the `minor/` directory:
        ```bash
        python backend/train.py
        ```
    *   The best model based on validation accuracy will be saved to `model/levit_pneumonia_best.pth`.

2.  **Train ResNet50 Model:**
    *   Ensure `FULL_TRAIN_DATA_PATH` in `backend/train_resnet.py` points to your training data.
    *   Run the script from the `minor/` directory:
        ```bash
        python backend/train_resnet.py
        ```
    *   The best model will be saved to `model/resnet50_pneumonia_best.pth`.

**Note:** Training was performed using **[Specify your training environment, e.g., Kaggle P100 GPU, MacBook Pro M1 with MPS]**. Using a GPU is highly recommended for practical training times.

## Evaluating and Comparing Models

After training, the models are evaluated on the unseen test set.

1.  **Evaluate LeViT Model:**
    ```bash
    python backend/evaluate_model.py
    ```
2.  **Evaluate ResNet50 Model:**
    ```bash
    python backend/evaluate_resnet.py
    ```
3.  **Compare Both Models:**
    ```bash
    python backend/compare_models.py
    ```
    This script generates detailed metrics and saves comparison plots to the `comparison_plots/` directory.

## Results and Discussion

**Training Environment:** **[Specify your training environment, e.g., Kaggle P100 GPU, MacBook Pro M1 with MPS]**

**LeViT Model Performance (Test Set):**
*   Best Validation Accuracy achieved during training: **[Your LeViT's Best Validation Accuracy, e.g., 90.00%]**
*   Test Set Overall Accuracy: **87.34%**
*   Test Set PNEUMONIA Precision: **0.8479**
*   Test Set PNEUMONIA Recall (Sensitivity): **0.9718**
*   Test Set PNEUMONIA F1-Score: **0.9056**
*   Test Set NORMAL Precision: **0.9379**
*   Test Set NORMAL Recall (Specificity): **0.7094**
*   Test Set NORMAL F1-Score: **0.8078**
*   *Discussion:* The LeViT model, trained from scratch with aggressive augmentation and regularization, achieved a test accuracy of 87.34%. A standout result is its exceptional recall (sensitivity) for pneumonia cases at 97.18%, indicating it correctly identified a very high percentage of actual pneumonia instances. This high sensitivity came at the cost of lower precision for pneumonia (84.79%) and lower recall for normal cases (specificity of 70.94%), suggesting a tendency to classify borderline cases as pneumonia, thus minimizing false negatives for the critical class.
*   *Attention Maps:* [Describe what the attention maps typically highlight. E.g., "Attention map visualizations from the LeViT model often highlighted areas within the lung fields, particularly regions with consolidations or opacities characteristic of pneumonia, suggesting the model was focusing on relevant pathological features for its predictions."]

**ResNet50 Model Performance (Benchmark - Test Set):**
*   Best Validation Accuracy achieved during training: **[Your ResNet50's Best Validation Accuracy]%**
*   Test Set Overall Accuracy: **90.54%**
*   Test Set PNEUMONIA Precision: **0.9190**
*   Test Set PNEUMONIA Recall (Sensitivity): **0.9308**
*   Test Set PNEUMONIA F1-Score: **0.9248**
*   Test Set NORMAL Precision: **0.8821**
*   Test Set NORMAL Recall (Specificity): **0.8632**
*   Test Set NORMAL F1-Score: **0.8726**
*   *Discussion:* The ResNet50 model, leveraging transfer learning from ImageNet, achieved a higher overall test accuracy of 90.54%. It demonstrated strong and more balanced performance across precision and recall for both classes compared to the LeViT model in this specific training run.

**Comparison:**
*   Overall, the pre-trained ResNet50 achieved a higher test accuracy (90.54%) compared to the LeViT model trained from scratch (87.34%). ResNet50 also demonstrated higher precision for pneumonia (0.9190 vs 0.8479) and a better balance in classifying normal cases (higher recall/specificity for normal at 0.8632 vs LeViT's 0.7094).
*   **Crucially, the custom LeViT model achieved a higher Pneumonia Recall (Sensitivity) of 0.9718 compared to ResNet50's 0.9308.** This is a significant finding. It indicates that the LeViT model, despite its lower overall accuracy and lower pneumonia precision in this instance, was more effective at identifying true pneumonia cases and had fewer false negatives for pneumonia.
*   *Discussion:* The ResNet50's overall superior accuracy and balanced metrics likely stem from the powerful, general-purpose features learned during its ImageNet pre-training. Training Vision Transformers like LeViT effectively from scratch is known to be data-intensive. However, the LeViT model's superior recall for pneumonia is noteworthy. This suggests that the combination of its architecture and the applied training strategies (weighted loss, aggressive augmentation, dropout) successfully biased the model towards correctly identifying the positive (Pneumonia) class, a characteristic often highly desirable in medical screening scenarios where minimizing missed positive cases is paramount. The trade-off was a higher rate of false positives for pneumonia (lower precision for pneumonia and lower recall for normal cases).

## Conclusion

This project successfully implemented a LeViT-like Vision Transformer model from scratch for pneumonia detection from chest X-rays. When trained with appropriate data augmentation, regularization, and class imbalance handling techniques, the LeViT model achieved a commendable test accuracy of 87.34%. More significantly, it demonstrated a superior Pneumonia Recall (Sensitivity) of 97.18% compared to a pre-trained ResNet50 benchmark (93.08%), highlighting its potential for applications where minimizing false negatives for the critical class is paramount.

While the ResNet50 baseline achieved higher overall accuracy due to the benefits of pre-training, this project effectively showcased the implementation of a LeViT model and demonstrated its capability to learn relevant features for this medical imaging task, particularly excelling in sensitivity for the target condition. The developed Streamlit application further provides a means for interactive prediction and visualization of the LeViT model's attention mechanisms.

## Future Work

*   Further hyperparameter tuning for the LeViT model, potentially using techniques like Bayesian optimization or automated hyperparameter search, specifically aiming to improve pneumonia precision while maintaining high recall.
*   Implement and evaluate decision threshold adjustments for the LeViT model to explore the precision-recall trade-off explicitly.
*   Experiment with different LeViT architectural variants (e.g., increasing model depth/width, incorporating downsampling stages akin to canonical LeViT designs).
*   Investigate fine-tuning the pre-trained ResNet50 backbone layers after initial head training for potential further improvement in its performance.
*   Explore more sophisticated attention visualization techniques for LeViT to gain deeper insights into its decision-making process.
*   Train the LeViT model on a significantly larger X-ray dataset to assess its scalability and potential to close the gap with pre-trained CNNs.
