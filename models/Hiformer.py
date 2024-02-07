import torch
import torch.nn as nn
import torchvision.models as models
from einops.layers.torch import Rearrange

# Assume All2Cross, ConvUpsample, and SegmentationHead are defined elsewhere
# If they are not, you will need to define these according to your project's specifications

class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        # Configuration parameters
        self.img_size = img_size
        self.n_classes = n_classes
        
        # All2Cross module that includes backbone and transformers
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)
        
        # ConvUpsample modules for up-sampling small, middle, and large level features
        self.ConvUp_s = ConvUpsample(in_chans=config.swin_pyramid_fm[2], out_chans=[128, 128], upsample=True)
        self.ConvUp_m = ConvUpsample(in_chans=config.swin_pyramid_fm[1], out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=config.swin_pyramid_fm[0], upsample=False)
        
        # Convolution to reduce channels before the segmentation head
        self.conv_pred = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # Segmentation head for final prediction
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )
    
    def forward(self, x):
        # Extract features at different levels
        xs = self.All2Cross(x)
        
        # Process each set of features with the corresponding ConvUpsample module
        small_features_upsampled = self.ConvUp_s(xs[-1])  # Assuming the last one is the smallest
        middle_features_upsampled = self.ConvUp_m(xs[1])  # Assuming the middle one is the middle
        large_features_processed = self.ConvUp_l(xs[0])   # Assuming the first one is the largest
        
        # Combine features from all levels
        combined_features = torch.cat((small_features_upsampled, middle_features_upsampled, large_features_processed), dim=1)
        combined_features = self.conv_pred(combined_features)
        
        # Get segmentation output
        segmentation_output = self.segmentation_head(combined_features)
        
        return segmentation_output

