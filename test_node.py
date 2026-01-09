
import torch
import numpy as np
from PIL import Image
try:
    from nodes import TK_Image2Mask
except ImportError:
    # If running from root, adjust path
    import sys
    import os
    sys.path.append(os.getcwd())
    from nodes import TK_Image2Mask

def load_image_to_tensor(path):
    img = Image.open(path)
    # Normalize to 0-1
    img_np = np.array(img).astype(np.float32) / 255.0
    # Add batch dimension: (1, H, W, C)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    return img_tensor, img.mode

def save_tensor_to_image(tensor, path):
    # tensor: (1, H, W, C) or (1, H, W)
    t = tensor.squeeze(0)
    if t.ndim == 2:
        # Mask (H, W) -> L mode
        t = t * 255.0
        img_np = t.byte().numpy()
        img = Image.fromarray(img_np, mode='L')
    else:
        # Image (H, W, C) -> RGB
        t = t * 255.0
        img_np = t.byte().numpy()
        img = Image.fromarray(img_np, mode='RGB')
    
    img.save(path)
    print(f"Saved to {path}")

def run_test():
    image_path = "test_img/test.png"
    print(f"Testing with {image_path}...")
    
    try:
        input_tensor, mode = load_image_to_tensor(image_path)
        print(f"Loaded image mode: {mode}, shape: {input_tensor.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    node = TK_Image2Mask()
    
    # Test 1: Normal
    print("Test 1: Normal Mode")
    mask, mask_img = node.process(input_tensor, invert_mask=False)
    save_tensor_to_image(mask, "test_output_mask.png")
    save_tensor_to_image(mask_img, "test_output_image_black.png")

    # Test 2: Inverted
    print("Test 2: Inverted Mode")
    mask_inv, mask_img_inv = node.process(input_tensor, invert_mask=True)
    save_tensor_to_image(mask_inv, "test_output_mask_inverted.png")
    save_tensor_to_image(mask_img_inv, "test_output_image_inverted.png")

if __name__ == "__main__":
    run_test()
