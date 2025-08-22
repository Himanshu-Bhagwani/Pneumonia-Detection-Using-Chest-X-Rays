# backend/inference.py
import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F # Needed for interpolate
import numpy as np # For processing heatmap
import traceback

# Need overlay_mask from torchcam (can install torchcam just for this)
# Or implement a simple overlay function if you want to avoid the dependency
try:
    from torchcam.utils import overlay_mask
except ImportError:
    print("Warning: torchcam not found. Heatmap overlay might not work.")
    print("Install torchcam (`pip install torchcam`) or implement overlay function.")
    def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.5) -> Image.Image:
        """Minimal overlay function if torchcam is not installed."""
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        mask_np = np.array(mask.convert('L')) # Ensure grayscale
        mask_norm = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-6) # Normalize 0-1
        heatmap_colored = (cmap(mask_norm)[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_colored)
        if heatmap_pil.size != img.size:
            heatmap_pil = heatmap_pil.resize(img.size, Image.BILINEAR)
        # Blend
        overlay = Image.blend(img.convert("RGB"), heatmap_pil.convert("RGB"), alpha)
        return overlay


# Assuming levit_model defines the LeViT class correctly
from levit_model import LeViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
model = None
try:
    model = LeViT(num_classes=2).to(device)
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "..", "model", "levit_pneumonia_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Transformations ---
# Ensure image size matches model training/init (default 224)
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Visualization Function ---
def visualize_attention(image_path):
    """
    Generates prediction and attention heatmap.
    """
    if not model:
        print("Error: Model not initialized.")
        return None, None, None

    if not os.path.isfile(image_path):
        print(f"Error: Invalid image path provided: {image_path}")
        return None, None, None

    try:
        original_img = Image.open(image_path).convert("RGB")
        input_tensor = transform(original_img).unsqueeze(0).to(device)

        # --- Model Forward Pass - Request Attention ---
        model.eval()
        with torch.no_grad():
            # Call model asking for attention map from the last block
            outputs, attention_map = model(input_tensor, return_attention=True)

            # --- Prediction Calculation ---
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            prediction_idx = predicted_class.item()
            prediction_label = "Pneumonia" if prediction_idx == 1 else "Normal"
            confidence_score = confidence.item()

        # --- Process Attention Map ---
        heatmap_overlay = original_img # Default to original if attention fails
        if attention_map is not None:
            print(f"Raw attention map shape: {attention_map.shape}") # Should be [1, num_heads, N, N]
            num_patches = model.num_patches
            num_heads = attention_map.shape[1]

            # 1. Average across heads
            avg_attention = attention_map.mean(dim=1).squeeze(0) # Shape [N, N]
            print(f"Average attention map shape: {avg_attention.shape}")

            # 2. Aggregate attention received by each patch
            # We average the attention scores for each destination patch (rows) across all source patches (columns)
            # Or simply use the diagonal (attention to self) - let's try averaging received attention
            # avg_attention_received = avg_attention.mean(dim=1) # Shape [N] -> represents importance of each patch

            # Alternative: Use attention *from* the mean representation (if it existed)
            # Alternative: Average attention *from* each patch to all others
            # Let's average the attention *received* by each patch:
            patch_importance = avg_attention.sum(dim=1) # Sum attention scores going *to* each patch (dim 1)
            # Or take the mean instead of sum: patch_importance = avg_attention.mean(dim=1)
            print(f"Patch importance vector shape: {patch_importance.shape}") # Should be [N]

            # 3. Reshape to 2D grid
            h_feat = w_feat = int(num_patches ** 0.5) # Calculate feature map H/W (e.g., 14 for 224/16)
            if patch_importance.shape[0] != h_feat * w_feat:
                 print(f"Error: Patch importance size {patch_importance.shape[0]} doesn't match expected {h_feat*w_feat}")
                 heatmap_2d = None
            else:
                heatmap_2d = patch_importance.reshape(1, 1, h_feat, w_feat) # Reshape to [1, 1, H_feat, W_feat]
                print(f"Reshaped heatmap shape: {heatmap_2d.shape}")

                # 4. Upscale to original image size
                heatmap_resized = F.interpolate(
                    heatmap_2d,
                    size=(IMG_SIZE, IMG_SIZE),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().cpu().numpy() # Remove batch/channel dims, move to CPU, convert to numpy
                print(f"Resized heatmap shape: {heatmap_resized.shape}")

                # 5. Normalize and convert to PIL Image
                heatmap_norm = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-6)
                heatmap_pil = Image.fromarray((heatmap_norm * 255).astype(np.uint8))

                # 6. Overlay heatmap
                heatmap_overlay = overlay_mask(original_img, heatmap_pil, alpha=0.6) # Adjust alpha

        print(f"Prediction: {prediction_label} (Confidence: {confidence_score:.4f})")
        return prediction_label, confidence_score, heatmap_overlay

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during prediction or visualization: {e}")
        traceback.print_exc()
        return None, None, None

# --- Example Usage ---
if __name__ == "__main__":
    test_image = "/Users/home/Documents/minor/chest_xray/test/PNEUMONIA/person25_virus_59.jpeg"

    if os.path.exists(test_image):
        label, confidence, heatmap_img = visualize_attention(test_image)
        if label is not None and heatmap_img is not None:
            print(f"Final Prediction: {label}")
            print(f"Confidence: {confidence:.4f}")
            if isinstance(heatmap_img, Image.Image):
                try:
                    heatmap_img.save("attention_heatmap_output.png")
                    print("Attention heatmap saved as attention_heatmap_output.png")
                except Exception as e:
                    print(f"Could not save heatmap image: {e}")
            else:
                print("Heatmap generation returned invalid object.")
        else:
            print("Prediction or visualization failed.")
    else:
        print(f"Test image not found at {test_image}. Please update the path for testing.")      