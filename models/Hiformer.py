import torch.nn as nn
from einops.layers.torch import Rearrange
from models.Encoder import All2Cross
from models.Decoder import ConvUpsample, SegmentationHead

# Assuming All2Cross, ConvUpsample, and SegmentationHead are defined elsewhere
class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 8, 16]  # Adding patch size for middle level
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)
        
        # Update the channel sizes based on your All2Cross module output
        self.ConvUp_s = ConvUpsample(in_chans= 384, out_chans=[128, 128], upsample=True)
        self.ConvUp_m = ConvUpsample(in_chans= 192, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans= 96, upsample=False)  # Assuming you want to upsample large features
        
        # The segmentation head might need to be updated to handle the combined output
        self.segmentation_head = SegmentationHead(
            in_channels=16,  # Assuming the combined features from all levels have 128*3 channels
            out_channels=n_classes,
            kernel_size=3,
        )    

        self.conv_pred = nn.Sequential(
            nn.Conv2d(128 * 2, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
    for i, embed in enumerate(embeddings):
        embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
        if i == 0:
            embed = self.ConvUp_l(embed)
        elif i == 1:
            embed = self.ConvUp_m(embed)  # Middle-level features processing
        else:
            embed = self.ConvUp_s(embed)
        reshaped_embed.append(embed)
        
    combined_features = torch.cat(reshaped_embed, dim=1)
    C = self.conv_pred(combined_features)
    out = self.segmentation_head(C)
    return out
