import ml_collections
import os
import wget

os.makedirs('./weights', exist_ok=True)

def get_hiformer_configs():
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]  # Feature maps for small, middle, large levels
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    swin_model_url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
    swin_weights_path = './weights/swin_tiny_patch4_window7_224.pth'
    
    if not os.path.isfile(swin_weights_path):
        print('Downloading Swin-transformer model ...')
        wget.download(swin_model_url, swin_weights_path)    
    cfg.swin_pretrained_path = swin_weights_path

    # CNN Configs (assuming resnet50 for illustration)
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256, 512, 1024]  # Channels for small, middle, large features
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 2, 2]]  # Depth for small, middle, large levels (added middle level depth)
    cfg.num_heads = [3, 6, 12]  # Heads for small, middle, large levels (added middle level heads)
    cfg.mlp_ratio = [4., 4., 4.]  # Ratios for small, middle, large levels (added middle level ratio)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg
