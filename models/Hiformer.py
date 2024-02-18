import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from models.Encoder import All2Cross
from models.Decoder import ConvUpsample, SegmentationHead
import torch
# Assuming All2Cross, ConvUpsample, and SegmentationHead are defined elsewhere
class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        #super().__init__()
        super(HiFormer, self).__init__()
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
            nn.Conv2d(384, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.All2Cross(x)  # Get embeddings from All2Cross
        embeddings = [x[:, 1:] for x in xs]  # Remove class token if present
        reshaped_embed = []

        for i, embed in enumerate(embeddings):
            # Reshape embeddings to feature map format
            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size // self.patch_size[i]), w=(self.img_size // self.patch_size[i]))(embed)

            # Apply correct ConvUpsample layer based on embedding level
            if i == 0:  # Large-level features
                embed = self.ConvUp_l(embed)
            elif i == 1:  # Middle-level features
                embed = self.ConvUp_m(embed)
            else:  # Small-level features
                embed = self.ConvUp_s(embed)
        
            reshaped_embed.append(embed)
            
        # Find the maximum height and width to resize all feature maps to match this size
        max_h = max(embed.shape[2] for embed in reshaped_embed)
        max_w = max(embed.shape[3] for embed in reshaped_embed)
        target_size = (max_h, max_w)
        # Resize all embeddings to have the same spatial dimensions
        #resized_embeds = [F.interpolate(embed, size=(max_h, max_w), mode='bilinear', align_corners=False) for embed in reshaped_embed]
        resized_embeds = [F.interpolate(e, size=target_size, mode='bilinear', align_corners=False) for e in reshaped_embed]
        # Now you can combine the resized embeddings directly
        combined_features = torch.cat(resized_embeds, dim=1)  # Concatenate along the channel dimension

        # Proceed with convolution and segmentation head
        C = self.conv_pred(combined_features)
        out = self.segmentation_head(C)
        return out

        # Combine processed embeddings
        # Ensure all embeddings have compatible dimensions before combining
        # This snippet assumes you're simply adding the embeddings. Consider resizing or interpolation if dimensions differ.
        #C = reshaped_embed[0] + reshaped_embed[1] + reshaped_embed[2]  # Adjust as needed based on actual processing logic
        #C = self.conv_pred(C)

        #out = self.segmentation_head(C)
        #return out

    #def calculate_target_hw(self, reshaped_embed):
        
        # Assuming reshaped_embed is a list of tensors with shape [batch_size, channels, height, width]
        #valid_embeds = [embed for embed in reshaped_embed if len(embed.shape) == 2]
        #if not valid_embeds:
          #  raise ValueError("No valid embeddings found with the expected number of dimensions (2).")
        #target_h = min([embed.shape[2] for embed in valid_embeds])
        #target_w = min([embed.shape[3] for embed in valid_embeds])
        #return target_h, target_w
        
    #def forward(self, x):
     #   xs = self.All2Cross(x)
       # embeddings = [x[:, 1:] for x in xs]
       # reshaped_embed = []
       # target_h, target_w = self.calculate_target_hw(xs)  # Define this method based on your model's specifics

       # for i, embed in enumerate(embeddings):
          #  embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
           # if i == 0:
             #   embed = self.ConvUp_l(embed)
           # elif i == 1:
              #  embed = self.ConvUp_m(embed)  # Middle-level features processing
           # else:
               # embed = self.ConvUp_s(embed)
            # Resize to target height and width
            #embed = F.interpolate(embed, size=(target_h, target_w), mode='bilinear', align_corners=False)
           # reshaped_embed.append(embed)
    
        #combined_features = torch.cat(reshaped_embed, dim=1)
       # C = self.conv_pred(combined_features)
        #out = self.segmentation_head(C)
        #return out

