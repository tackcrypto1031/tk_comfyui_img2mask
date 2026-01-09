
import  torch

class TK_Image2Mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "mask_image")
    FUNCTION = "process"
    CATEGORY = "TK_Nodes"

    def process(self, image, invert_mask, mask=None):
        # image shape is usually (B, H, W, C)
        
        # Determine alpha channel
        if mask is not None:
            # If mask is provided, use it as alpha
            # Mask shape is usually (B, H, W) or (H, W). Ensure it matches B of image if possible.
            alpha = mask
            # Ensure alpha has batch dim if image has it
            if alpha.ndim == 2 and image.ndim == 4:
                alpha = alpha.unsqueeze(0)
            
            # Broadcast verify (simple check, comfyui usually handles this matches)
            # Resize might be needed if valid mask but different size? 
            # Usually users ensure size match. We assume match.
            
        elif image.shape[-1] == 4:
            alpha = image[..., 3]
        else:
            # If no alpha and no mask input, assume full opacity
            alpha = torch.ones_like(image[..., 0])

        if invert_mask:
            alpha = 1.0 - alpha

        # MASK output: Just the alpha channel
        mask_out = alpha

        # IMAGE output: White Background + Black Mask
        # Formula: Result = (1 - Alpha) * White + Alpha * Black
        # Result = 1.0 - Alpha
        
        # We need to return an RGB image (3 channels)
        # alpha shape is (B, H, W). unsqueeze to (B, H, W, 1) to broadcast
        alpha_expanded = alpha.unsqueeze(-1)
        
        # Result = 1.0 - alpha
        mask_image_out = 1.0 - alpha_expanded
        
        # Expand to 3 channels for RGB
        mask_image_out = mask_image_out.repeat(1, 1, 1, 3)

        return (mask_out, mask_image_out)

NODE_CLASS_MAPPINGS = {
    "TK_Image2Mask": TK_Image2Mask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK_Image2Mask": "Transparent Image to Mask (TK)"
}
