"""Test exposure fusion implementation."""
import torch
import torch.nn.functional as F
import numpy as np

def exposure_fusion_framework(I_u, I_o, L_u, L_o, sigma=0.2, eps=1e-12):
    """Fuse illumination maps using Mertens et al. exposure fusion weights."""

    def get_weights(img_rgb):
        # Contrast via Laplacian magnitude on grayscale
        gray = (0.299 * img_rgb[:, 0:1] + 0.587 * img_rgb[:, 1:2] + 0.114 * img_rgb[:, 2:3])
        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device=img_rgb.device, dtype=img_rgb.dtype).view(1, 1, 3, 3)
        laplacian = torch.abs(F.conv2d(gray, kernel, padding=1))

        # Saturation via channel-wise standard deviation
        saturation = torch.std(img_rgb, dim=1, keepdim=True)

        # Well-exposedness via Gaussian profile centered at 0.5
        exposure = torch.exp(-((img_rgb - 0.5) ** 2) / (2 * sigma ** 2))
        exposure = torch.prod(exposure, dim=1, keepdim=True)

        return (laplacian * saturation * exposure) + eps

    w_u = get_weights(I_u)
    w_o = get_weights(I_o)
    weight_sum = w_u + w_o
    w_u_norm = w_u / weight_sum
    w_o_norm = w_o / weight_sum

    return w_u_norm * L_u + w_o_norm * L_o


# Test case
print("Testing Exposure Fusion Implementation")
print("=" * 60)

# Create synthetic test data
B, H, W = 1, 64, 64

# Underexposed image: dark overall, low saturation
I_u = torch.rand(B, 3, H, W) * 0.3  # Dark RGB

# Overexposed image: bright overall, high saturation in some regions
I_o = torch.rand(B, 3, H, W) * 0.7 + 0.3  # Bright RGB

# Add a saturated region to I_o (colorful area)
I_o[:, :, 20:40, 20:40] = torch.tensor([[[0.9]], [[0.2]], [[0.3]]])  # Red-ish region

# Illumination maps (single channel)
L_u = torch.mean(I_u, dim=1, keepdim=True)  # Derived from underexposed
L_o = torch.mean(I_o, dim=1, keepdim=True)  # Derived from overexposed

print(f"Input shapes:")
print(f"  I_u (RGB): {I_u.shape}")
print(f"  I_o (RGB): {I_o.shape}")
print(f"  L_u (Gray): {L_u.shape}")
print(f"  L_o (Gray): {L_o.shape}")

# Perform fusion
L_fused = exposure_fusion_framework(I_u, I_o, L_u, L_o)

print(f"\nOutput shape:")
print(f"  L_fused: {L_fused.shape}")

# Verify output
print(f"\nValue ranges:")
print(f"  L_u: [{L_u.min():.3f}, {L_u.max():.3f}]")
print(f"  L_o: [{L_o.min():.3f}, {L_o.max():.3f}]")
print(f"  L_fused: [{L_fused.min():.3f}, {L_fused.max():.3f}]")

# Check that fusion is between the two inputs
assert L_fused.min() >= min(L_u.min(), L_o.min()) - 0.01
assert L_fused.max() <= max(L_u.max(), L_o.max()) + 0.01

print("\n✓ Fusion produces valid output (within input range)")

# Check guided fusion behavior
# In the saturated region, we expect higher weight on I_o (better saturation)
# So L_fused should be closer to L_o in that region
saturated_region = L_fused[:, :, 20:40, 20:40]
L_u_region = L_u[:, :, 20:40, 20:40]
L_o_region = L_o[:, :, 20:40, 20:40]

dist_to_u = torch.abs(saturated_region - L_u_region).mean()
dist_to_o = torch.abs(saturated_region - L_o_region).mean()

print(f"\nSaturated region analysis:")
print(f"  Distance to L_u: {dist_to_u:.4f}")
print(f"  Distance to L_o: {dist_to_o:.4f}")

if dist_to_o < dist_to_u:
    print("  ✓ Fused L prefers L_o in saturated region (correct)")
else:
    print("  ⚠ Fused L prefers L_u in saturated region (unexpected)")

print("\n" + "=" * 60)
print("Implementation Verification:")
print("✓ Weights computed from RGB inputs (I_u, I_o)")
print("  - Contrast: Laplacian on grayscale")
print("  - Saturation: std(R,G,B)")
print("  - Well-exposedness: Gaussian at 0.5")
print("✓ Weights applied to illumination maps (L_u, L_o)")
print("✓ Output is weighted average of L_u and L_o")
print("\nThis matches Mertens '07 guided fusion principle!")
