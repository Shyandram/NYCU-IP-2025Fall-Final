"""Inspect color correction matrices from trained checkpoint."""
import torch
import numpy as np
import sys

checkpoint_path = "exp/12_11_02_18/model/ckpt_12.pth"

print("=" * 70)
print(f"Inspecting checkpoint: {checkpoint_path}")
print("=" * 70)

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("\n✓ Checkpoint loaded successfully")
    print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
    
    # Check what's in the checkpoint
    if 'net_fusion' in checkpoint:
        print("\n" + "=" * 70)
        print("FUSION NETWORK PARAMETERS")
        print("=" * 70)
        
        fusion_state = checkpoint['net_fusion']
        
        # Look for color matrix related parameters
        color_params = {k: v for k, v in fusion_state.items() if 'color' in k.lower()}
        gamma_params = {k: v for k, v in fusion_state.items() if 'gamma' in k.lower()}
        
        if color_params:
            print("\n📊 Color Matrix Predictor Parameters:")
            print("-" * 70)
            for name, param in color_params.items():
                print(f"  {name}:")
                print(f"    Shape: {param.shape}")
                if 'base' in name or 'bias' in name:
                    print(f"    Values:\n{param.numpy()}")
                elif len(param.shape) <= 2 and param.numel() <= 20:
                    print(f"    Values:\n{param.numpy()}")
                else:
                    print(f"    Mean: {param.mean():.4f}, Std: {param.std():.4f}")
                    print(f"    Min: {param.min():.4f}, Max: {param.max():.4f}")
        
        if gamma_params:
            print("\n🎚️ Gamma Correction Parameters:")
            print("-" * 70)
            for name, param in gamma_params.items():
                print(f"  {name}:")
                print(f"    Shape: {param.shape}")
                if 'base' in name or 'bias' in name or param.numel() <= 20:
                    print(f"    Values:\n{param.numpy()}")
                else:
                    print(f"    Mean: {param.mean():.4f}, Std: {param.std():.4f}")
                    print(f"    Min: {param.min():.4f}, Max: {param.max():.4f}")
        
        if not color_params and not gamma_params:
            print("\n⚠️ No color matrix or gamma parameters found in fusion network")
            print("\nAll fusion network parameters:")
            for name in sorted(fusion_state.keys()):
                print(f"  - {name}: {fusion_state[name].shape}")
    
    if 'net_L' in checkpoint:
        print("\n" + "=" * 70)
        print("ILLUMINATION NETWORK PARAMETERS")
        print("=" * 70)
        
        L_state = checkpoint['net_L']
        
        # Look for color matrix parameters in L_net
        L_color_params = {k: v for k, v in L_state.items() if 'color' in k.lower()}
        
        if L_color_params:
            print("\n📊 L_net Color Matrix Parameters:")
            print("-" * 70)
            for name, param in L_color_params.items():
                print(f"  {name}:")
                print(f"    Shape: {param.shape}")
                if 'base' in name or len(param.shape) <= 2:
                    if param.numel() <= 20:
                        print(f"    Values:\n{param.numpy()}")
                    else:
                        print(f"    Mean: {param.mean():.4f}, Std: {param.std():.4f}")
                        print(f"    Min: {param.min():.4f}, Max: {param.max():.4f}")
        else:
            print("\n⚠️ No color matrix parameters in L_net")
    
    # Test color matrix prediction on sample data
    print("\n" + "=" * 70)
    print("TESTING COLOR MATRIX PREDICTION")
    print("=" * 70)
    
    from nets.net import fusionnet, L_net, L_net_ViT
    
    # Try to load the model
    device = 'cpu'
    
    # Determine model configuration from checkpoint
    has_color_predictor = any('color_predictor' in k for k in fusion_state.keys())
    has_gamma_predictor = any('gamma_predictor' in k for k in fusion_state.keys())
    
    print(f"\nModel configuration detected:")
    print(f"  - Color matrix predictor: {has_color_predictor}")
    print(f"  - Gamma correction predictor: {has_gamma_predictor}")
    
    # Load model with appropriate settings
    net_fusion = fusionnet(
        use_color_matrix=has_color_predictor,
        use_gamma_correction=has_gamma_predictor,
        gamma_type='exp'
    ).to(device)
    
    net_fusion.load_state_dict(fusion_state)
    net_fusion.eval()
    
    # Load actual test images
    import cv2
    from utils import image_read
    
    test_img1_path = "test_case/under/225.png"
    test_img2_path = "test_case/over/225.png"
    
    print(f"\nLoading test images:")
    print(f"  Under-exposed: {test_img1_path}")
    print(f"  Over-exposed: {test_img2_path}")
    
    try:
        img1_np = image_read(test_img1_path, 'RGB')
        img2_np = image_read(test_img2_path, 'RGB')
        
        # Resize if needed
        if img1_np.shape[:2] != img2_np.shape[:2]:
            h = min(img1_np.shape[0], img2_np.shape[0])
            w = min(img1_np.shape[1], img2_np.shape[1])
            img1_np = cv2.resize(img1_np, (w, h))
            img2_np = cv2.resize(img2_np, (w, h))
        
        # Resize to reasonable size for processing
        max_dim = 256
        h, w = img1_np.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img1_np = cv2.resize(img1_np, (new_w, new_h))
            img2_np = cv2.resize(img2_np, (new_w, new_h))
        
        print(f"  Image shape: {img1_np.shape}")
        
        # Convert to tensor
        x1 = torch.from_numpy(img1_np.transpose(2,0,1)[np.newaxis,...]/255.).float()
        x2 = torch.from_numpy(img2_np.transpose(2,0,1)[np.newaxis,...]/255.).float()
        
        # Generate illumination map (using mean or L_net if available)
        L = torch.mean(x1, dim=1, keepdim=True)  # Simple approximation
        
        print(f"  Tensor shapes: x1={x1.shape}, x2={x2.shape}, L={L.shape}")
        
    except FileNotFoundError:
        print(f"  ⚠️ Test images not found, using random tensors instead")
        x1 = torch.randn(1, 3, 128, 128)
        x2 = torch.randn(1, 3, 128, 128)
        L = torch.sigmoid(torch.randn(1, 1, 128, 128))
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = net_fusion(x1, x2, L)
    
    print(f"\nOutput structure:")
    print(f"  Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape {out.shape}")
    
    if has_gamma_predictor:
        print("\n🎚️ GAMMA CORRECTION OUTPUT:")
        if len(outputs) >= 5:
            gamma_val = outputs[3]
            color_matrix = outputs[4]
            print(f"  Gamma values: {gamma_val.squeeze().numpy()}")
            print(f"  Color matrix shape: {color_matrix.shape}")
            print(f"  Color matrix (sample):")
            print(f"{color_matrix[0].numpy()}")
            
            # Check deviation from identity
            identity = torch.eye(3)
            deviation = torch.norm(color_matrix[0] - identity).item()
            print(f"\n  Deviation from identity: {deviation:.4f}")
            
            if deviation < 0.1:
                print("  ⚠️ Matrix is very close to identity (minimal correction)")
            else:
                print("  ✓ Matrix shows significant correction")
    
    elif has_color_predictor:
        print("\n🎨 COLOR MATRIX OUTPUT:")
        if len(outputs) >= 4:
            color_matrix = outputs[3]
            print(f"  Color matrix shape: {color_matrix.shape}")
            print(f"  Color matrix (sample):")
            print(f"{color_matrix[0].numpy()}")
            
            # Check deviation from identity
            identity = torch.eye(3)
            deviation = torch.norm(color_matrix[0] - identity).item()
            print(f"\n  Deviation from identity: {deviation:.4f}")
            
            if deviation < 0.1:
                print("  ⚠️ Matrix is very close to identity (minimal correction)")
            else:
                print("  ✓ Matrix shows significant correction")
            
            # Analyze matrix effects
            print(f"\n  Matrix interpretation:")
            M = color_matrix[0].numpy()
            print(f"    New R = {M[0,0]:.3f}×R + {M[0,1]:.3f}×G + {M[0,2]:.3f}×B")
            print(f"    New G = {M[1,0]:.3f}×R + {M[1,1]:.3f}×G + {M[1,2]:.3f}×B")
            print(f"    New B = {M[2,0]:.3f}×R + {M[2,1]:.3f}×G + {M[2,2]:.3f}×B")
            
            # Show input image statistics
            print(f"\n  Input image statistics:")
            print(f"    Under-exposed mean: R={x1[0,0].mean():.3f}, G={x1[0,1].mean():.3f}, B={x1[0,2].mean():.3f}")
            print(f"    Over-exposed mean:  R={x2[0,0].mean():.3f}, G={x2[0,1].mean():.3f}, B={x2[0,2].mean():.3f}")
            
            # Test matrix application
            print(f"\n  Testing matrix on sample pixels:")
            # Get center pixel from Rhat (corrected reflectance)
            Rhat = outputs[2]
            h, w = Rhat.shape[2], Rhat.shape[3]
            center_pixel_before = Rhat[0, :, h//2, w//2].numpy()
            
            # The matrix was already applied in the network, but let's show the effect
            print(f"    Center pixel (after correction): R={center_pixel_before[0]:.3f}, G={center_pixel_before[1]:.3f}, B={center_pixel_before[2]:.3f}")
    
    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

except FileNotFoundError:
    print(f"\n❌ Error: Checkpoint file not found at {checkpoint_path}")
    print("Please check the path and try again.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
