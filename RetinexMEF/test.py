import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from nets.net import L_net, L_net_ViT, fusionnet
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration flags - should match training configuration
use_vit_L_net = False  # Set to True if model was trained with L_net_ViT
use_color_matrix = True  # Set to True if model was trained with color matrix (from img1+img2)
use_L_net_color = True  # Set to True if CNN L_net was trained with color matrix
use_gamma_correction = False  # Set to True if model was trained with gamma correction
gamma_type = 'exp'  # 'exp' for fixed gamma (exposure correction), other for learnable gamma
color_matrix_boost_alpha = 1.0  # >1 amplifies M via M_boost = I + alpha*(M-I); 1.0 keeps prediction unchanged
use_exposure_fusion_for_L = False  # Enable Mertens exposure fusion for illumination blending
use_adaptive_gamma_correction = False  # Enable adaptive gamma correction after fusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_fusion=fusionnet(use_color_matrix=use_color_matrix,
                     use_gamma_correction=use_gamma_correction,
                     gamma_type=gamma_type,
                     color_matrix_boost=color_matrix_boost_alpha).to(device)
if use_vit_L_net:
    net_L = L_net_ViT().to(device)
else:
    net_L = L_net(predict_color_matrix=use_L_net_color).to(device)

# --- 1. 定義金字塔工具 (標準實作) ---
def laplacian_pyramid(img, levels=5):
    pyr = []
    current = img
    for _ in range(levels - 1):
        down = F.avg_pool2d(current, kernel_size=2, stride=2)
        up = F.interpolate(down, size=current.shape[-2:], mode='bilinear', align_corners=False)
        pyr.append(current - up) # High freq details
        current = down
    pyr.append(current) # Low freq base
    return pyr

def gaussian_pyramid(img, levels=5):
    pyr = [img]
    current = img
    for _ in range(levels - 1):
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        pyr.append(current)
    return pyr

def reconstruct_pyramid(pyr):
    image = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        up = F.interpolate(image, size=pyr[i].shape[-2:], mode='bilinear', align_corners=False)
        image = up + pyr[i]
    return image

# --- 2. 核心融合邏輯 ---
def artifact_free_L_fusion(L_u, L_o):
    """
    論文認證的無偽影融合：Pyramid Blending + Exposure Weight Only
    """
    # A. 計算權重 (只看曝光度，Sigma加大至0.4以求平滑)
    sigma = 0.4
    w_u = torch.exp(-torch.pow(L_u - 0.5, 2) / (2 * sigma ** 2))
    w_o = torch.exp(-torch.pow(L_o - 0.5, 2) / (2 * sigma ** 2)) + 1e-12

    # B. 歸一化
    w_sum = w_u + w_o
    w_u = w_u / w_sum
    w_o = w_o / w_sum

    # C. 金字塔融合 (解決光暈與接縫的唯一解)
    levels = 5
    pyr_Lu = laplacian_pyramid(L_u, levels)
    pyr_Lo = laplacian_pyramid(L_o, levels)
    pyr_Wu = gaussian_pyramid(w_u, levels)
    pyr_Wo = gaussian_pyramid(w_o, levels)

    pyr_fused = []
    for i in range(levels):
        # L_fused = L_u * W_u + L_o * W_o
        pyr_fused.append(pyr_Lu[i] * pyr_Wu[i] + pyr_Lo[i] * pyr_Wo[i])

    # D. 重建
    L_fused = reconstruct_pyramid(pyr_fused)
    
    return torch.clamp(L_fused, 0.0, 1.0)

def adaptive_gamma_correction(L_fused, target_mean=0.6):
    """
    自動計算最佳 Gamma 值，將 L_fused 的平均亮度拉到 target_mean。
    """
    # 1. 計算當前亮度
    current_mean = torch.mean(L_fused) + 1e-6
    
    # 2. 根據公式 log(target) = gamma * log(current) 反推 gamma
    # Target = Current ^ gamma  =>  log(Target) = gamma * log(Current)
    gamma = torch.log(torch.tensor(target_mean)) / torch.log(current_mean)
    
    # 3. 限制 Gamma 範圍 (安全機制)
    # 避免 Gamma 太大(變全白)或太小(變全黑)
    gamma = torch.clamp(gamma, 0.5, 2.5)
    
    # 4. 應用
    L_corrected = torch.pow(L_fused, gamma)
    return L_corrected
# path_model=r"exp/12_07_04_21/model/ckpt_36.pth"  # Update to your trained model path
path_model=r"exp/12_11_02_18/model/ckpt_30.pth"  # Update to your trained model path
# path_model=r"exp/12_03_16_27/model/ckpt_40.pth"  # Update to your trained model path
# path_model = r"exp/12_09_18_14/model/ckpt_16.pth"  # Update to your trained model path
# path_model=r"model/ckpt.pth"  # Update to your trained model path

# Path to SICE_test50 or separate folders for under/over exposed images
use_paired_dataset = True  # Set to True to use SICE_test50 format (folders with img_1.png, img_2.png)
path_test_dataset = r"data/SICE_test50"  # Path to SICE_test50 dataset

# Alternative: separate folders for under/over exposed images
path_img1=r"test_case/under"
path_img2=r"test_case/over"
path_result=r"results/c2wom"  # Path to save fusion results

exposure_adjustment=False #If True, adjust the exposure level according to the preset exposure value.
exposureLevel=0.4

checkpoint = torch.load(path_model)
net_fusion.load_state_dict(checkpoint['net_fusion'])
net_fusion.eval()
net_L.load_state_dict(checkpoint['net_L'])
net_L.eval()
os.makedirs(path_result,exist_ok=True)

# Prepare image pairs based on dataset format
if use_paired_dataset:
    # Load from SICE_test50 format: each subfolder contains img_1.png (low) and img_2.png (high)
    sample_folders = sorted([d for d in os.listdir(path_test_dataset) if os.path.isdir(os.path.join(path_test_dataset, d))])
    image_pairs = []
    for folder in sample_folders:
        folder_path = os.path.join(path_test_dataset, folder)
        img1_path = os.path.join(folder_path, "img_1.png")
        img2_path = os.path.join(folder_path, "img_2.png")
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            image_pairs.append((img1_path, img2_path, folder))
else:
    # Load from separate folders
    image_pairs = []
    for imgname in os.listdir(path_img1):
        img1_path = os.path.join(path_img1, imgname)
        img2_path = os.path.join(path_img2, imgname)
        if os.path.exists(img2_path):
            image_pairs.append((img1_path, img2_path, imgname))

with torch.no_grad():
    for img1_path, img2_path, output_name in tqdm(image_pairs):
        img1 = image_read(img1_path, 'RGB')
        img2 = image_read(img2_path, 'RGB')
        
        # Resize if image is too large (max dimension 1024)
        max_dim = 1024
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # First, ensure both images have the same dimensions
        if img1.shape[:2] != img2.shape[:2]:
            # Use the smaller dimensions to avoid upscaling
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
            h1, w1 = target_h, target_w
            h2, w2 = target_h, target_w
        
        # Then, resize both images together if they exceed max_dim
        if max(h1, w1) > max_dim:
            scale = max_dim / max(h1, w1)
            new_h, new_w = int(h1 * scale), int(w1 * scale)
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        img1 = img1.transpose(2,0,1)[np.newaxis,...]/255.
        img2 = img2.transpose(2,0,1)[np.newaxis,...]/255.
        
        img1, img2 = torch.FloatTensor(img1).to(device), torch.FloatTensor(img2).to(device)
        
        # Handle both L_net and L_net_ViT outputs
        if use_vit_L_net or use_L_net_color:
            L1, color_mat1 = net_L(img1)
            L2, color_mat2 = net_L(img2)
        else:
            L1, L2 = net_L(img1), net_L(img2)

        if use_exposure_fusion_for_L:
            L = artifact_free_L_fusion(L1, L2)
        elif exposure_adjustment:
            k=calculate_k(L1,L2,exposureLevel)
            L=FuseIlluminance(L1,L2,k)
        else:
            L=(L1+L2)/2

        if use_adaptive_gamma_correction:
            L = adaptive_gamma_correction(L, target_mean=0.6)
        
        # fusionnet handles color correction internally:
        # - If use_gamma_correction=True: predicts gamma + color matrix from img1+img2, applies to Rhat
        # - If use_color_matrix=True: predicts/uses color matrix for Rhat
        # - External color matrix from L_net is ignored if use_gamma_correction=True
        if use_L_net_color and not use_gamma_correction:
            # Use synthetic reference for L_net color prediction (only if gamma correction is off)
            # img_synth = (img1 + img2) / 2
            # _, color_mat_synth = net_L(img_synth)
            color_mat_synth = (color_mat1 + color_mat2) / 2
            fusion_output = net_fusion(img1, img2, L, color_matrix_external=color_mat_synth)
        else:
            # fusionnet handles color matrix/gamma correction internally
            fusion_output = net_fusion(img1, img2, L)
        
        # Handle fusion output
        # Gamma mode: (output, R, Rhat, gamma_val, color_matrix)
        # Color mode: (output, R, Rhat, color_matrix)
        # Plain mode: (output, R, Rhat)
        Ihat = fusion_output[0]
        R = fusion_output[1]
        Rhat = fusion_output[2]
        
        # Use Ihat which is already R*L (and gamma corrected if enabled)
        imgout=np.uint8(np.squeeze(Ihat.detach().cpu().numpy()).transpose(1,2,0)*255)
        output_filename = f"{output_name}.png" if not output_name.endswith('.png') else output_name
        plt.imsave(os.path.join(path_result, output_filename), imgout)


