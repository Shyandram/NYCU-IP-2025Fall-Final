import os
import argparse
import glob
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import pyiqa
import torch
import kornia

"""
MEF evaluation over SICE2_sorted.
- For PSNR/SSIM: uses SICE2_Label as ground truth reference.
- CC: Pearson correlation coefficient vs. ground truth.
- NMI: Normalized mutual information vs. ground truth.
- Qcb: Colorfulness (Hasler & Süsstrunk) computed on fused image.
- Qnice: Simple naturalness proxy based on luminance and saturation balance (reported as experimental, not a standard).
"""

def read_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def read_image_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_to_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[:2] == b.shape[:2]:
        return a
    return cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size != b.size:
        min_size = min(a.size, b.size)
        a = a[:min_size]
        b = b[:min_size]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def normalized_mutual_information(a: torch.Tensor, b: torch.Tensor, bins: int = 256) -> float:
    # Compute NMI using joint histogram
    a_scaled = (a * 255.0).clamp(0, 255).long()
    b_scaled = (b * 255.0).clamp(0, 255).long()
    
    # Create 2D histogram manually using bincount
    a_flat = a_scaled.flatten()
    b_flat = b_scaled.flatten()
    
    # Create joint histogram
    joint_idx = a_flat * 256 + b_flat
    hgram_flat = torch.bincount(joint_idx, minlength=256*256).float()
    hgram = hgram_flat.reshape(256, 256)
    
    pxy = hgram / torch.sum(hgram)
    px = torch.sum(pxy, dim=1)
    py = torch.sum(pxy, dim=0)
    
    # Compute entropies
    px_nz = px[px > 0]
    py_nz = py[py > 0]
    pxy_nz = pxy[pxy > 0]
    
    Hx = -torch.sum(px_nz * torch.log(px_nz + 1e-12))
    Hy = -torch.sum(py_nz * torch.log(py_nz + 1e-12))
    Hxy = -torch.sum(pxy_nz * torch.log(pxy_nz + 1e-12))
    
    Ixy = Hx + Hy - Hxy
    denom = torch.max(Hx, Hy)
    
    if denom < 1e-12:
        return 0.0
    return float((Ixy / denom).item())


def colorfulness_hasler_susstrunk(img_bgr: torch.Tensor) -> float:
    # Qcb: Colorfulness as per Hasler & Süsstrunk
    img = img_bgr * 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg = torch.std(rg)
    std_yb = torch.std(yb)
    mean_rg = torch.mean(rg)
    mean_yb = torch.mean(yb)
    C = torch.sqrt(std_rg**2 + std_yb**2) + 0.3 * torch.sqrt(mean_rg**2 + mean_yb**2)
    return float(C)


def qnice_proxy(img_bgr: torch.Tensor) -> float:
    # Experimental proxy: combine mid-range luminance and moderate saturation preference
    img_rgb = img_bgr[..., [2, 1, 0]]  # BGR to RGB
    img_hsv = kornia.color.rgb_to_hsv(img_rgb.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
    h, s, v = img_hsv[..., 0], img_hsv[..., 1], img_hsv[..., 2]
    # Preference for saturation around 0.3~0.6 and luminance around 0.5
    sat_pref = torch.exp(-((s - 0.45) ** 2) / (2 * (0.2 ** 2)))
    lum_pref = torch.exp(-((v - 0.5) ** 2) / (2 * (0.2 ** 2)))
    score = torch.mean(0.5 * sat_pref + 0.5 * lum_pref)
    return float(score)


def compute_metrics_for_scene(fused_path: str, gt_path: str) -> dict:
    fused_gray_np = read_image_gray(fused_path)
    fused_color_np = read_image_color(fused_path)
    gt_gray_np = read_image_gray(gt_path)
    gt_gray_np = resize_to_match(gt_gray_np, fused_gray_np)
    
    # Convert to torch tensors
    fused_gray = torch.tensor(fused_gray_np, dtype=torch.float32) / 255.0
    fused_color = torch.tensor(fused_color_np, dtype=torch.float32) / 255.0
    gt_gray = torch.tensor(gt_gray_np, dtype=torch.float32) / 255.0

    # PSNR & SSIM vs ground truth
    try:
        mse = torch.mean((gt_gray - fused_gray) ** 2)
        psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse > 0 else float('inf')
    except Exception:
        psnr_val = 0.0
    try:
        ssim_val = structural_similarity(gt_gray.numpy(), fused_gray.numpy(), data_range=1.0)
    except Exception:
        ssim_val = 0.0

    # CC & NMI vs ground truth
    cc_val = torch.corrcoef(torch.stack([gt_gray.flatten(), fused_gray.flatten()]))[0, 1].item()
    nmi_val = normalized_mutual_information(gt_gray, fused_gray)

    # Qcb (colorfulness) & Qnice proxy on fused only
    qcb = colorfulness_hasler_susstrunk(fused_color)
    qnice = qnice_proxy(fused_color)

    # NIQE & BRISQUE using pyiqa
    try:
        niqe_model = pyiqa.create_metric('niqe')
        niqe_score = niqe_model(fused_gray.unsqueeze(0).unsqueeze(0)).item()
    except Exception:
        niqe_score = float('nan')

    try:
        brisque_model = pyiqa.create_metric('brisque')
        brisque_score = brisque_model(fused_gray.unsqueeze(0).unsqueeze(0)).item()
    except Exception:
        brisque_score = float('nan')

    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'CC': cc_val,
        'NMI': nmi_val,
        'Qcb': qcb,
        'Qnice': qnice,
        'NIQE': niqe_score,
        'BRISQUE': brisque_score,
    }


def find_fused_for_scene(fused_root: str, scene_id: str) -> str:
    # Try exact filename matches (scene_id with common extensions)
    for ext in ('.png', '.jpg', '.jpeg', '.tif'):
        p = os.path.join(fused_root, scene_id + ext)
        if os.path.isfile(p):
            return p
    # Fallback: search files containing scene_id
    candidates = [p for p in glob.glob(os.path.join(fused_root, '*')) if os.path.isfile(p)]
    for p in candidates:
        if scene_id in os.path.basename(p):
            return p
    return ''


def find_gt_for_scene(gt_root: str, scene_id: str) -> str:
    # Try exact filename matches (scene_id with common extensions)
    for ext in ('.PNG', '.png', '.JPG', '.jpg', '.jpeg', '.tif'):
        p = os.path.join(gt_root, scene_id + ext)
        if os.path.isfile(p):
            return p
    # Fallback: search files containing scene_id
    candidates = [p for p in glob.glob(os.path.join(gt_root, '*')) if os.path.isfile(p)]
    for p in candidates:
        basename = os.path.basename(p)
        if basename.startswith(scene_id + '.') or basename.startswith(scene_id + ':'):
            return p
    return ''


def main():
    parser = argparse.ArgumentParser(description='MEF Evaluation on SICE2_sorted')
    parser.add_argument('--inputs_root', type=str, default='data/SICE2_sorted', help='Path to SICE2_sorted root')
    parser.add_argument('--gt_root', type=str, default='data/SICE2_Label', help='Path to SICE2_Label ground truth')
    parser.add_argument('--fused_root', type=str, default='results/c2wom', help='Path to directory containing fused results')
    parser.add_argument('--csv_out', type=str, default='exp/eval/c2wom.csv', help='Output CSV path')
    args = parser.parse_args()

    scene_dirs = [d for d in sorted(os.listdir(args.inputs_root)) if os.path.isdir(os.path.join(args.inputs_root, d))]
    results = []

    for scene_id in tqdm(scene_dirs, desc='Evaluating scenes'):
        scene_path = os.path.join(args.inputs_root, scene_id)
        
        fused_path = find_fused_for_scene(args.fused_root, scene_id)
        if not fused_path:
            # Skip if we cannot find fused image
            continue
        
        gt_path = find_gt_for_scene(args.gt_root, scene_id)
        if not gt_path:
            # Skip if we cannot find ground truth
            continue

        try:
            metrics = compute_metrics_for_scene(fused_path, gt_path)
            metrics['scene'] = scene_id
            results.append(metrics)
        except Exception as e:
            # Skip on error, but log minimal info
            print(f"Error processing scene {scene_id}: {e}")
            continue

    if len(results) == 0:
        print('No results computed. Check fused_root filenames vs scene IDs.')
        return

    # Dataset-level averages
    keys = ['PSNR', 'SSIM', 'CC', 'NMI', 'Qcb', 'Qnice', 'NIQE', 'BRISQUE']
    avg = {k: float(np.nanmean([r[k] for r in results])) for k in keys}

    print('Dataset averages over', len(results), 'scenes:')
    for k in keys:
        print(f'{k}: {avg[k]:.4f}')

    # Write CSV
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, 'w') as f:
        header = 'scene,' + ','.join(keys) + '\n'
        f.write(header)
        for r in results:
            line = r['scene'] + ',' + ','.join(f"{r[k]:.6f}" for k in keys) + '\n'
            f.write(line)
    print('Saved per-scene metrics to', args.csv_out)

    # Save dataset-level averages to log file
    log_path = os.path.splitext(args.csv_out)[0] + '.log'
    with open(log_path, 'w') as log_file:
        log_file.write('Dataset averages over ' + str(len(results)) + ' scenes:\n')
        for k in keys:
            log_file.write(f'{k}: {avg[k]:.4f}\n')
    print('Saved dataset averages to log file:', log_path)


if __name__ == '__main__':
    main()
