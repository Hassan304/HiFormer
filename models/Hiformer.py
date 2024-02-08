import torch.nn as nn
from einops.layers.torch import Rearrange
from models.Encoder import All2Cross
from models.Decoder import ConvUpsample, SegmentationHead

# Assuming All2Cross, ConvUpsample, and SegmentationHead are defined elsewhere
class Hiformer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 8, 16]  # Adding patch size for middle level
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)
        
        # Update the channel sizes based on your All2Cross module output
        self.ConvUp_s = ConvUpsample(in_chans=config.swin_pyramid_fm[2], out_chans=[128, 128], upsample=True)
        self.ConvUp_m = ConvUpsample(in_chans=config.swin_pyramid_fm[1], out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=config.swin_pyramid_fm[0], out_chans=[128, 128], upsample=True)  # Assuming you want to upsample large features
        
        # The segmentation head might need to be updated to handle the combined output
        self.segmentation_head = SegmentationHead(
            in_channels=128 * 3,  # Assuming the combined features from all levels have 128*3 channels
            out_channels=n_classes,
            kernel_size=3,
        )    

        self.conv_pred = nn.Sequential(
            nn.Conv2d(128 * 3, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):
        # Get features from All2Cross module
        xs = self.All2Cross(x)
        
        # Process each set of features with the corresponding ConvUpsample module
        # Ensure that the output of All2Cross matches these expectations
        large_features = self.ConvUp_l(xs[0])  # Adjust indexing based on your All2Cross output
        middle_features = self.ConvUp_m(xs[1])
        small_features = self.ConvUp_s(xs[2])
        
        # Combine the features from all levels
        combined_features = torch.cat([large_features, middle_features, small_features], dim=1)
        
        # Reduce channels and upsample
        combined_features = self.conv_pred(combined_features)
        
        # Get the segmentation output
        out = self.segmentation_head(combined_features)
        
        return out
