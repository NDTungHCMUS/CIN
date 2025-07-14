import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleJPEGNoise(nn.Module):
    """
    Simple JPEG-like noise that approximates compression artifacts
    without complex DCT operations
    """
    def __init__(self, quality=75):
        super(SimpleJPEGNoise, self).__init__()
        self.quality = quality
        
        # Convert quality to compression strength
        self.compression_strength = (100 - quality) / 100.0  # 0.0 to 1.0
        
    def forward(self, encoded, cover_img):
        x = encoded
        convert_back = False
        
        # Handle range conversion
        if x.min() < -0.1:
            x = (x + 1) / 2
            convert_back = True
        
        x = torch.clamp(x, 0, 1)
        
        # Simple JPEG-like artifacts
        # 1. Slight blur (simulating DCT compression)
        kernel_size = 3
        sigma = self.compression_strength * 0.5
        if sigma > 0:
            # Create Gaussian kernel
            kernel = torch.exp(-torch.arange(-1, 2).float().pow(2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, 3, 1).repeat(3, 1, 1, 1)
            kernel = kernel.to(x.device)
            
            # Apply horizontal blur
            x = F.conv2d(x, kernel, padding=(1, 0), groups=3)
            
            # Apply vertical blur
            kernel = kernel.transpose(2, 3)
            x = F.conv2d(x, kernel, padding=(0, 1), groups=3)
        
        # 2. Quantization noise
        quantization_levels = int(256 * (1 - self.compression_strength * 0.5))
        if quantization_levels < 256:
            x = torch.floor(x * quantization_levels) / quantization_levels
        
        # 3. Add slight uniform noise (simulating compression artifacts)
        noise_strength = self.compression_strength * 0.01
        if noise_strength > 0:
            noise = torch.rand_like(x) * noise_strength - noise_strength/2
            x = x + noise
        
        x = torch.clamp(x, 0, 1)
        
        if convert_back:
            x = x * 2 - 1
            x = torch.clamp(x, -1, 1)
        
        return x


class DiffJPEG(nn.Module):    
    def __init__(self, differentiable=True, quality=75):
        super(DiffJPEG, self).__init__()
        
        # Try to use real JPEG, fallback to simple version
        try:
            if differentiable:
                rounding = diff_round
            else:
                rounding = torch.round
            
            factor = quality_to_factor(quality)
            self.compress = compress_jpeg(rounding=rounding, factor=factor)
            self.decompress = decompress_jpeg(rounding=rounding, factor=factor)
            self.use_real_jpeg = True
            print(f"Using real JPEG with quality={quality}")
            
        # except Exception as e:
        #     print(f"Failed to initialize real JPEG: {e}")
        #     print(f"Using simple JPEG approximation instead")
        #     self.simple_jpeg = SimpleJPEGNoise(quality=quality)
        #     self.use_real_jpeg = False
        

    def forward(self, encoded, cover_img):
        if not self.use_real_jpeg:
            return self.simple_jpeg(encoded, cover_img)
        
        x = encoded
        convert_back = False
        
        if x.min() < -0.1:
            x = (x + 1) / 2
            convert_back = True
        
        x = torch.clamp(x, 0, 1)
        
        # Check for invalid values
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: Invalid values in JPEG input")
            x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
        
        try:
            y, cb, cr = self.compress(x)
            
            # Check if compression produced valid outputs
            if torch.isnan(y).any() or torch.isnan(cb).any() or torch.isnan(cr).any():
                print("WARNING: NaN in compression output, using fallback")
                return self.simple_jpeg(encoded, cover_img)
            
            recovered = self.decompress(y, cb, cr, x.shape[2], x.shape[3])
            
            # Check if decompression produced valid output
            if torch.isnan(recovered).any() or torch.isinf(recovered).any():
                print("WARNING: Invalid decompression output, using fallback")
                return self.simple_jpeg(encoded, cover_img)
            
            recovered = torch.clamp(recovered, 0, 1)
            
        except Exception as e:
            print(f"JPEG processing failed: {e}, using fallback")
            return self.simple_jpeg(encoded, cover_img)
        
        if convert_back:
            recovered = recovered * 2 - 1
            recovered = torch.clamp(recovered, -1, 1)
        
        return recovered