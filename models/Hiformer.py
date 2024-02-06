import torch.nn as nn
from einops.layers.torch import Rearrange

from models.Encoder import All2Cross
from models.Decoder import ConvUpsample, SegmentationHead


class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 8, 16] # Assuming the middle level has a patch size of 8
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config = config, img_size= img_size, in_chans=in_chans)
        
        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128,128], upsample=True)
        self.ConvUp_m = ConvUpsample(in_chans=192, out_chans=[128,128], upsample=True) # Here Adjust channels accordingly
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)
    
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )    

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        
         # Rearrange and upsample each set of features
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size // self.patch_size[i]), w=(self.img_size // self.patch_size[i]))(embed)
            if i == 0:  # Large level features
                embed = self.ConvUp_l(embed)
            elif i == 1:  # Middle level features (newly added)
                embed = self.ConvUp_m(embed)
            else:  # Small level features
                embed = self.ConvUp_s(embed)
            reshaped_embed.append(embed)
        
        # Combine features from all levels
        combined_features = reshaped_embed[0] + reshaped_embed[1] + reshaped_embed[2]
        combined_features = self.conv_pred(combined_features)

        out = self.segmentation_head(combined_features)
        
        return out  
