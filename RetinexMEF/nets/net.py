import torch
import torch.nn as nn
try:
    from nets.restormer import *
except ModuleNotFoundError:
    from restormer import *
from timm.models.layers import trunc_normal_, DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class ColorMatrixPredictor(nn.Module):
    """
    Independent color matrix predictor
    Input: Two exposure images (under + over)
    Output: Ideal 3x3 color correction matrix for fused reflectance
    """
    def __init__(self, in_channels=6, dim=64, num_heads=4, simple=True):
        super(ColorMatrixPredictor, self).__init__()
        self.color_base = nn.Parameter(torch.eye(3), requires_grad=True)
        self.simple = simple
        
        if simple:
            # Lightweight version: process concatenated exposures
            self.feature_extract = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # H/2, W/2
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4, W/4
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/8, W/8
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 9)  # 3x3 matrix
            )
            # Zero initialization for stability
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            # Complex version with attention
            self.conv_embed = conv_embedding(in_channels, dim)
            self.generator = query_SABlock(dim=dim, num_heads=num_heads)
            self.color_linear = nn.Linear(dim, 1)
            
            self.apply(self._init_weights)
            for name, p in self.named_parameters():
                if 'generator.attn.v.weight' in name:
                    nn.init.constant_(p, 0)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, img_under, img_over):
        """
        Args:
            img_under: underexposed image [B, 3, H, W]
            img_over: overexposed image [B, 3, H, W]
        Returns:
            color_matrix: [B, 3, 3] correction matrix
        """
        B = img_under.shape[0]
        
        # Concatenate two exposures
        x = torch.cat([img_under, img_over], dim=1)  # [B, 6, H, W]
        
        if self.simple:
            # Simple path
            x_feat = self.feature_extract(x)  # [B, 128, H/8, W/8]
            x_pool = self.global_pool(x_feat).view(B, -1)  # [B, 128]
            color_offset = self.mlp(x_pool).view(B, 3, 3)
            color_matrix = self.color_base + color_offset
        else:
            # Complex path with attention
            x = self.conv_embed(x)  # [B, dim, H/4, W/4]
            x = self.generator(x)  # [B, 10, dim]
            color_tokens = x[:, :9]  # First 9 tokens for 3x3 matrix
            color_matrix = self.color_linear(color_tokens).squeeze(-1).view(B, 3, 3) + self.color_base
        
        return color_matrix


class Global_pred(nn.Module):
    """Predicts global 3x3 color transformation matrix (legacy - for L_net compatibility)"""
    def __init__(self, in_channels=3, out_channels=64, num_heads=4, simple=False):
        super(Global_pred, self).__init__()
        self.color_base = nn.Parameter(torch.eye(3), requires_grad=True)
        self.simple = simple
        
        if simple:
            # Moderate version: lightweight conv + pooling + MLP
            # Extract spatial features before pooling
            self.feature_extract = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # H/2, W/2
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # H/4, W/4
                nn.ReLU(),
            )
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 9)
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            # Original complex version
            self.conv_large = conv_embedding(in_channels, out_channels)
            self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
            self.color_linear = nn.Linear(out_channels, 1)

            self.apply(self._init_weights)

            for name, p in self.named_parameters():
                if name == 'generator.attn.v.weight':
                    nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        if self.simple:
            # Moderate path: lightweight conv + pooling + MLP
            x_feat = self.feature_extract(x)  # [B, 32, H/4, W/4]
            x_pool = self.global_pool(x_feat).view(B, -1)  # [B, 32]
            color_offset = self.mlp(x_pool).view(B, 3, 3)
            color = self.color_base + color_offset
        else:
            # Complex path: original implementation
            x = self.conv_large(x)
            x = self.generator(x)  # [B, 10, out_channels]
            color_tokens = x[:, :9]  # Use first 9 tokens for 3x3 matrix
            color = self.color_linear(color_tokens).squeeze(-1).view(B, 3, 3) + self.color_base
        
        return color


class Global_pred_with_gamma(nn.Module):
    """Predicts global 3x3 color transformation matrix and gamma correction from two exposures"""
    def __init__(self, in_channels=6, out_channels=64, num_heads=4, type='exp', simple=True):
        super(Global_pred_with_gamma, self).__init__()
        if type == 'exp':
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=False)  # False in exposure correction
        else:
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)  # basic color matrix
        self.simple = simple
        
        if simple:
            # Lightweight version: process concatenated exposures
            self.feature_extract = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # H/2, W/2
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4, W/4
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/8, W/8
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)  # 1 gamma + 9 color matrix elements
            )
            # Zero initialization for stability
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            # Complex version with attention - process concatenated exposures
            self.conv_large = conv_embedding(in_channels, out_channels)
            self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
            self.gamma_linear = nn.Linear(out_channels, 1)
            self.color_linear = nn.Linear(out_channels, 1)

            self.apply(self._init_weights)

            for name, p in self.named_parameters():
                if name == 'generator.attn.v.weight':
                    nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def apply_color(image, ccm):
        """Apply color correction matrix to image
        Args:
            image: [H, W, C] tensor
            ccm: [3, 3] color correction matrix
        Returns:
            corrected image [H, W, C]
        """
        shape = image.shape
        image = image.view(-1, 3)  # [H*W, 3]
        image = torch.tensordot(image, ccm, dims=([1], [1]))  # [H*W, 3]
        image = image.view(shape)  # [H, W, 3]
        return image

    def forward(self, img_low, img_high):
        """
        Args:
            img_low: underexposed image [B, 3, H, W]
            img_high: overexposed image [B, 3, H, W]
        Returns:
            gamma: [B, 1] gamma values
            color: [B, 3, 3] color correction matrix
        """
        B = img_low.shape[0]
        # Concatenate two exposures
        x = torch.cat([img_low, img_high], dim=1)  # [B, 6, H, W]
        
        if self.simple:
            # Simple path
            x_feat = self.feature_extract(x)  # [B, 128, H/8, W/8]
            x_pool = self.global_pool(x_feat).view(B, -1)  # [B, 128]
            output = self.mlp(x_pool)  # [B, 10]
            gamma = output[:, 0:1] + self.gamma_base  # [B, 1]
            color_offset = output[:, 1:].view(B, 3, 3)  # [B, 3, 3]
            color = self.color_base + color_offset
        else:
            # Complex path with attention
            x = self.conv_large(x)
            x = self.generator(x)
            gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
            gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
            color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base
        
        return gamma, color


class L_net(nn.Module):
    def __init__(self, num=64, predict_color_matrix=False):
        super(L_net, self).__init__()
        self.predict_color_matrix = predict_color_matrix
        
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )
        
        # Optional color matrix predictor (use simple version for efficiency)
        if predict_color_matrix:
            self.color_predictor = Global_pred(in_channels=3, out_channels=64, num_heads=4, simple=True)

    def forward(self, input):
        L = torch.sigmoid(self.L_net(input))
        
        if self.predict_color_matrix:
            color_matrix = self.color_predictor(input)
            return L, color_matrix
        
        return L


class L_net_ViT(nn.Module):
    """ViT-based L_net that predicts both illumination and color matrix"""
    def __init__(self, in_channels=3, dim=64, num_heads=4):
        super(L_net_ViT, self).__init__()
        
        # Shared feature extraction
        self.conv_embed = conv_embedding(in_channels, dim)
        
        # Global prediction using query-based attention (for color matrix)
        self.color_predictor = query_SABlock(dim=dim, num_heads=num_heads)
        self.color_linear = nn.Linear(dim, 1)
        self.color_base = nn.Parameter(torch.eye(3), requires_grad=True)
        
        # Spatial feature extraction for illumination
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        
        # Illumination branch (spatial upsampling)
        self.illum_upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
        
        for name, p in self.named_parameters():
            if 'color_predictor.attn.v.weight' in name:
                nn.init.constant_(p, 0)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        B = input.shape[0]
        
        # Extract features
        x = self.conv_embed(input)  # [B, dim, H/4, W/4]
        
        # Spatial features for illumination
        x_spatial = self.spatial_conv(x)  # [B, dim, H/4, W/4]
        illum = self.illum_upsample(x_spatial)  # [B, 1, H, W]
        
        # Global features for color matrix (same approach as Global_pred)
        x_global = self.color_predictor(x)  # [B, 10, dim]
        color_tokens = x_global[:, :9]  # Use first 9 tokens for 3x3 matrix
        color_matrix = self.color_linear(color_tokens).squeeze(-1).view(B, 3, 3) + self.color_base
        
        return illum, color_matrix

class SRE(nn.Module): # Shared R 
    def __init__(self,
                 out_channels=3,
                 dim=32,
                 num_blocks=3,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(SRE, self).__init__() 

        self.patch_embed1 = nn.Conv2d(6,dim, kernel_size=3,stride=1, padding=1, bias=bias)

        self.encoder = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])

        self.output1 = nn.Sequential(
            nn.Conv2d(int(dim),  out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        
        self.act = nn.Sigmoid()  

    def forward(self, x1,x2):
        R=self.encoder(self.patch_embed1(torch.concatenate([x1,x2],1)))
        Rhat=self.act(self.output1(R))
        
        return Rhat,R
    

class fusionnet(nn.Module):
    def __init__(self,
                 out_channels=3,
                 dim=64,
                 num_blocks=3,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 use_color_matrix=False,
                 use_gamma_correction=False,
                 gamma_type='exp',
                 color_matrix_boost=1.0):
        super(fusionnet, self).__init__() 

        self.SRE=SRE()
        self.use_color_matrix = use_color_matrix
        self.use_gamma_correction = use_gamma_correction
        self.color_matrix_boost = color_matrix_boost
        
        self.patch_embed2 = nn.Conv2d(3,dim, kernel_size=3,stride=1, padding=1, bias=bias)
        self.decoder=nn.ModuleList([TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type,crossatt=True) for i in range(2*num_blocks)])
        

        self.output2 = nn.Sequential(
            nn.Conv2d(int(dim),  out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)

        # Independent color matrix predictor
        if use_color_matrix:
            self.color_predictor = ColorMatrixPredictor(in_channels=6, dim=64, num_heads=4, simple=True)
        
        # Independent gamma correction predictor
        if use_gamma_correction:
            self.gamma_predictor = Global_pred_with_gamma(in_channels=6, out_channels=64, num_heads=4, type=gamma_type, simple=True)

        self.act = nn.Sigmoid()  
    
    def _boost_color_matrix(self, matrix):
        if matrix is None or self.color_matrix_boost == 1.0:
            return matrix
        identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype).unsqueeze(0).expand_as(matrix)
        return identity + self.color_matrix_boost * (matrix - identity)
                    
          
    def forward(self, x1, x2, L, color_matrix_external=None):
        Rhat,_=self.SRE(x1,x2)
        
        # Determine which color correction to apply
        color_matrix = None
        gamma_val = None
        
        if self.use_gamma_correction:
            # Gamma correction mode: predict both gamma and color matrix from two exposures
            gamma_val, color_matrix = self.gamma_predictor(x1, x2)
            color_matrix = self._boost_color_matrix(color_matrix)
            # Apply color correction and gamma to Rhat
            B, C, H, W = Rhat.shape
            Rhat_permuted = Rhat.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
            Rhat_corrected = torch.stack([
                self.gamma_predictor.apply_color(Rhat_permuted[i,:,:,:], color_matrix[i,:,:]) ** gamma_val[i,:]
                for i in range(B)
            ], dim=0)
            Rhat = Rhat_corrected.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
            Rhat = torch.clamp(Rhat, 0, 1)
        else:
            # Standard color matrix mode (if gamma correction is not used)
            if color_matrix_external is not None:
                # Use external color matrix from L_net_ViT or L_net
                color_matrix = color_matrix_external
            elif self.use_color_matrix:
                # Predict color matrix from two exposures (independent prediction)
                color_matrix = self.color_predictor(x1, x2)
            
            if color_matrix is not None:
                color_matrix = self._boost_color_matrix(color_matrix)
                # Apply color correction: Rhat_corrected = color_matrix @ Rhat
                B, C, H, W = Rhat.shape
                Rhat_flat = Rhat.view(B, C, -1)  # [B, 3, H*W]
                Rhat_corrected = torch.bmm(color_matrix, Rhat_flat)  # [B, 3, 3] @ [B, 3, H*W] = [B, 3, H*W]
                Rhat = Rhat_corrected.view(B, C, H, W)
                Rhat = torch.clamp(Rhat, 0, 1)
        
        R=self.patch_embed2(Rhat)

        # Interpolate L to match R's spatial dimensions for decoder
        if L.shape[2:] != R.shape[2:]:
            L_resized = torch.nn.functional.interpolate(L, size=R.shape[2:], mode='bilinear', align_corners=False)
        else:
            L_resized = L

        for k in range(len(self.decoder)):
            R=R+self.decoder[k](R)*L_resized

        R = torch.clamp(Rhat+self.output2(R),0,1)
        
        # Ensure L matches final R dimensions for output
        if L.shape[2:] != R.shape[2:]:
            L = torch.nn.functional.interpolate(L, size=R.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute final output
        output = R * L
        
        # Return appropriate outputs
        if gamma_val is not None:
            return output, R, Rhat, gamma_val, color_matrix
        elif color_matrix is not None:
            return output, R, Rhat, color_matrix
        
        return output, R, Rhat



def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,3,128,128).astype(np.float32)).cuda()
    x2=torch.tensor(np.random.rand(2,1,128,128).astype(np.float32)).cuda()
    
    print("Testing fusionnet without color matrix...")
    model = fusionnet()
    model.cuda()
    y = model(x,x,x2)
    print('output shape:', y[0].shape, 'outputs:', len(y))
    
    print("\nTesting fusionnet with color matrix...")
    model_color = fusionnet(use_color_matrix=True)
    model_color.cuda()
    y_color = model_color(x,x,x2)
    print('output shape:', y_color[0].shape, 'outputs:', len(y_color))
    if len(y_color) == 4:
        print('color matrix shape:', y_color[3].shape)
    
    print("\nTesting fusionnet with external color matrix...")
    model_no_color = fusionnet(use_color_matrix=False)
    model_no_color.cuda()
    external_color = torch.eye(3).unsqueeze(0).repeat(2, 1, 1).cuda()
    y_external = model_no_color(x, x, x2, color_matrix_external=external_color)
    print('output shape:', y_external[0].shape, 'outputs:', len(y_external))
    if len(y_external) == 4:
        print('color matrix shape:', y_external[3].shape)
    
    print("\nTesting Global_pred module...")
    color_pred = Global_pred(in_channels=3, out_channels=64, num_heads=4)
    color_pred.cuda()
    color_matrix = color_pred(x)
    print('color_matrix shape:', color_matrix.shape)
    
    print("\nTesting Global_pred_with_gamma module (simple)...")
    gamma_color_pred = Global_pred_with_gamma(in_channels=6, out_channels=64, num_heads=4, type='exp', simple=True)
    gamma_color_pred.cuda()
    gamma, color_matrix = gamma_color_pred(x, x)
    print('gamma shape:', gamma.shape, 'color_matrix shape:', color_matrix.shape)
    
    # Test apply_color method
    img_test = x[0].permute(1, 2, 0)  # [H, W, C]
    img_corrected = gamma_color_pred.apply_color(img_test, color_matrix[0])
    print('color corrected image shape:', img_corrected.shape)
    
    # Test gamma application
    img_gamma_corrected = img_corrected ** gamma[0]
    print('gamma corrected image shape:', img_gamma_corrected.shape)
    
    print("\nTesting fusionnet with gamma correction...")
    model_gamma = fusionnet(use_color_matrix=False, use_gamma_correction=True, gamma_type='exp')
    model_gamma.cuda()
    y_gamma = model_gamma(x, x, x2)
    print('output shape:', y_gamma[0].shape, 'outputs:', len(y_gamma))
    if len(y_gamma) >= 5:
        print('gamma shape:', y_gamma[3].shape, 'color_gamma shape:', y_gamma[4].shape)
    
    print("\nTesting L_net without color matrix...")
    l_net = L_net(predict_color_matrix=False)
    l_net.cuda()
    illum = l_net(x)
    print('illumination shape:', illum.shape)
    
    print("\nTesting L_net with color matrix...")
    l_net_color = L_net(predict_color_matrix=True)
    l_net_color.cuda()
    illum_c, color_mat_c = l_net_color(x)
    print('illumination shape:', illum_c.shape, 'color_matrix shape:', color_mat_c.shape)
    
    print("\nTesting L_net_ViT module...")
    l_net_vit = L_net_ViT(in_channels=3, dim=64, num_heads=4)
    l_net_vit.cuda()
    illum, color_mat = l_net_vit(x)
    print('illumination shape:', illum.shape, 'color_matrix shape:', color_mat.shape)


if __name__ == '__main__':
    unit_test()