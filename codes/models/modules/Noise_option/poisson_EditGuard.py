import torch
import torch.nn as nn
import random


class PoissonNoise(nn.Module):
    def __init__(self, opt, device):
        """
        Initialize PoissonNoise layer.
        
        Args:
            opt (dict): Configuration options
            device (torch.device): Device to run on
        """
        super(PoissonNoise, self).__init__()
        self.opt = opt
        self.device = device
        self.vals = 10**4  # Default scaling factor
    
    def forward(self, encoded, cover_img):
        """
        Apply Poisson noise to the encoded image.
        
        Args:
            encoded (torch.Tensor): Watermarked image tensor
            cover_img (torch.Tensor): Original cover image (not used)
            
        Returns:
            torch.Tensor: Noisy image with Poisson noise applied
        """
        y_forw = encoded
        
        # Convert to [0, 1] range if input is in [-1, 1]
        if y_forw.min() < 0:
            y_forw = (y_forw + 1) / 2  # Convert [-1, 1] to [0, 1]
            convert_back = True
        else:
            convert_back = False
        
        # Clamp to ensure valid range for Poisson
        y_forw = torch.clamp(y_forw, 0, 1)
        
        # Check for invalid values
        if torch.isnan(y_forw).any() or torch.isinf(y_forw).any():
            print("WARNING: Invalid values detected, replacing with zeros")
            y_forw = torch.nan_to_num(y_forw, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Apply Poisson noise using the specified logic
        vals = self.vals
        
        try:
            if random.random() < 0.5:
                # Color noise: Apply to all channels
                # Ensure input is valid for Poisson
                poisson_input = y_forw * vals
                poisson_input = torch.clamp(poisson_input, 0, 1e6)  # Prevent overflow
                
                noisy_img_tensor = torch.poisson(poisson_input) / vals
            else:
                # Grayscale noise: Apply to grayscale version
                img_gray_tensor = torch.mean(y_forw, dim=1, keepdim=True)
                
                # Ensure input is valid for Poisson
                poisson_input = img_gray_tensor * vals
                poisson_input = torch.clamp(poisson_input, 0, 1e6)  # Prevent overflow
                
                noisy_gray_tensor = torch.poisson(poisson_input) / vals
                noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)
        
        except RuntimeError as e:
            print(f"Poisson error: {e}")
            print(f"Input range: [{y_forw.min():.6f}, {y_forw.max():.6f}]")
            print(f"Vals: {vals}")
            # Fallback to original image if Poisson fails
            noisy_img_tensor = y_forw

        # Clamp result to valid range
        y_forw = torch.clamp(noisy_img_tensor, 0, 1)
        
        # Convert back to original range if needed
        if convert_back:
            y_forw = y_forw * 2 - 1  # Convert [0, 1] back to [-1, 1]
            y_forw = torch.clamp(y_forw, -1, 1)
        
        return y_forw
    
    def __repr__(self):
        return f"PoissonNoise(vals={self.vals})"