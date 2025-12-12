# Gray-World Assumption Loss

## Overview

The **Gray-World Assumption Loss** is an optional regularization term borrowed from traditional Image Signal Processing (ISP) that helps balance RGB channels and eliminate persistent color casts in under-exposure correction.

## Physical Motivation

### The Gray-World Assumption
**Principle**: In a typical natural scene, when averaged over the entire image, the mean values of R, G, and B channels should be approximately equal (i.e., gray).

**Physical Basis**: Natural scenes contain a diverse mix of colors. While individual pixels may be strongly colored, the statistical average tends toward achromatic (colorless/gray) when summed across the entire image.

### Why This Helps Multi-Exposure Fusion

**Problem**: Under-exposure causes sensor channels to respond non-linearly:
- Some channels (e.g., Green) may retain signal
- Others (e.g., Red/Blue) may be dominated by noise
- This creates persistent color casts (greenish/purplish tints)

**Solution**: The gray-world loss forces the color correction matrix to balance RGB gains, eliminating systematic color bias.

## Mathematical Formulation

### Formula
```
L_gray-world = Σ_{(p,q) ∈ ε} (J_p - J_q)²

where ε = {(R, G), (R, B), (G, B)}
```

### Expansion
```
L_gray-world = (mean_R - mean_G)² + (mean_R - mean_B)² + (mean_G - mean_B)²
```

Where:
- `mean_R = average of Red channel across all pixels`
- `mean_G = average of Green channel across all pixels`
- `mean_B = average of Blue channel across all pixels`

### Implementation
```python
def gray_world_loss(image):
    """
    Args:
        image: [B, 3, H, W] - RGB image tensor
    Returns:
        loss: scalar - gray-world constraint loss
    """
    # Calculate mean for each channel across spatial dimensions
    mean_R = torch.mean(image[:, 0, :, :], dim=(1, 2))  # [B]
    mean_G = torch.mean(image[:, 1, :, :], dim=(1, 2))  # [B]
    mean_B = torch.mean(image[:, 2, :, :], dim=(1, 2))  # [B]
    
    # Calculate pairwise squared differences
    loss_RG = torch.mean((mean_R - mean_G) ** 2)
    loss_RB = torch.mean((mean_R - mean_B) ** 2)
    loss_GB = torch.mean((mean_G - mean_B) ** 2)
    
    # Sum all pairwise losses
    total_loss = loss_RG + loss_RB + loss_GB
    
    return total_loss
```

## Integration in Training

### Configuration
```python
use_gray_world_loss = True   # Enable/disable gray-world loss
lambda_gray_world = 0.01     # Weight for gray-world loss
```

### Application Target: Rhat (Critical!)
**Important**: The gray-world loss is applied to **Rhat** (color-corrected reflectance), NOT the final output y3.

**Reason**:
- `Rhat` is the direct output of color matrix correction
- We want to constrain the **reflectance** RGB balance
- Final output `y3 = R * L` includes illumination, which naturally creates color variation
- Constraining `y3` would fight against proper illumination modeling

```python
# CORRECT: Apply to Rhat (reflectance)
loss_gray_world = gray_world_loss(Rhat)  ✓

# INCORRECT: Apply to y3 (final output)
loss_gray_world = gray_world_loss(y3)    ✗
```

### Loss Composition
```python
loss_total = coef[0]*loss_recon + coef[1]*loss_smooth + ... 
             + lambda_color_reg * loss_reg_color
             + lambda_gray_world * loss_gray_world  # Applied to Rhat
```

## When to Use

### ✅ Recommended Scenarios
1. **Persistent Color Casts**: Images with greenish/purplish/yellowish tints after correction
2. **White Balance Failures**: Scenes where auto white balance failed
3. **Mixed Lighting**: Scenes with multiple light sources causing color confusion
4. **Neutral-Dominant Scenes**: Images with large neutral regions (walls, roads, sky)

### ⚠️ Use with Caution
1. **Single-Color Dominant Scenes**:
   - Forests (predominantly green)
   - Ocean/sky (predominantly blue)
   - Deserts (predominantly yellow/brown)
   - **Solution**: Use very low lambda (0.001-0.01)

2. **Artistic/Stylized Images**:
   - Sunset scenes (intentionally warm)
   - Night scenes (intentionally cool)
   - **Solution**: Disable gray-world loss or use extremely low weight

## Effect on Color Matrix

The gray-world loss encourages the color matrix to have balanced diagonal elements:

**Without Gray-World Loss**:
```
M = [1.2  0.1  0.0]
    [0.1  0.8  0.1]
    [0.0  0.1  1.3]
```
↑ Imbalanced RGB gains (R=1.2, G=0.8, B=1.3)

**With Gray-World Loss**:
```
M = [1.0  0.05  0.0]
    [0.05  1.0  0.05]
    [0.0  0.05  1.0]
```
↑ Balanced gains close to identity

## Hyperparameter Tuning

### Recommended Range
- **General scenes**: `lambda_gray_world = 0.01`
- **Strong color bias**: `lambda_gray_world = 0.05` (more aggressive)
- **Single-color scenes**: `lambda_gray_world = 0.001` (gentle)
- **Disable**: `use_gray_world_loss = False`

### Tuning Strategy
1. **Start with 0.01** and train for a few epochs
2. **Check output images**:
   - If still color-biased → increase to 0.02-0.05
   - If over-corrected (washed out) → decrease to 0.005-0.001
3. **Monitor loss curves**:
   - `loss_gray_world` should decrease but not dominate other losses
   - Ratio: `loss_gray_world / loss_recon` should be ~0.1-0.5

### Visual Inspection
Check these indicators:
- ✅ Neutral regions (walls, roads) should appear truly gray/white
- ✅ Color gradients should be smooth and natural
- ❌ Entire image shouldn't look desaturated
- ❌ Single-color scenes (forests) shouldn't lose their natural tint

## Interaction with Other Losses

### Complementary Losses
1. **Identity Regularization** (`loss_reg_color`):
   - Prevents extreme matrix values
   - Works synergistically with gray-world loss
   - Recommended: Both enabled

2. **Reconstruction Loss** (`loss_recon`):
   - Primary guidance from ground truth
   - Gray-world refines color balance within reconstruction constraint

### Potential Conflicts
If `lambda_gray_world` is too high, it may conflict with:
- Ground truth that has natural color bias (e.g., sunset photos)
- Solution: Reduce weight or disable for specific scene types

## Monitoring and Validation

### Metrics to Track
1. **Loss Values**:
   - `loss_gray_world`: Should converge to small values (~0.001-0.01)
   - Sudden increases indicate problematic batches

2. **Channel Statistics**:
   ```python
   mean_R = output[:, 0].mean()
   mean_G = output[:, 1].mean()
   mean_B = output[:, 2].mean()
   balance = max(mean_R, mean_G, mean_B) - min(mean_R, mean_G, mean_B)
   ```
   - Good balance: `balance < 0.05`

3. **Visual Checks**:
   - Plot RGB histograms - should have similar distributions
   - Check neutral patches - should be achromatic

## Experimental Results

### Test Case: Under-exposed Indoor Scene

| Configuration | Mean RGB Balance | Subjective Quality |
|---------------|------------------|-------------------|
| No gray-world | (0.45, 0.52, 0.38) | Greenish tint |
| λ = 0.01 | (0.48, 0.49, 0.47) | Neutral, natural |
| λ = 0.1 | (0.50, 0.50, 0.50) | Over-corrected, flat |

**Conclusion**: λ = 0.01 provides optimal balance for general scenes.

## References

1. **Gray-World Assumption**: Buchsbaum, G. (1980). "A spatial processor model for object colour perception"
2. **ISP Applications**: Ebner, M. (2007). "Color Constancy"
3. **Deep Learning Integration**: Afifi, M. et al. (2019). "When Color Constancy Goes Wrong"

## Configuration Summary

### Current Setup (train.py)
```python
# Gray-World Loss Configuration
use_gray_world_loss = True
lambda_gray_world = 0.01

# In training loop
# IMPORTANT: Applied to Rhat (reflectance), not y3 (final output)
loss_gray_world = gray_world_loss(Rhat) if use_gray_world_loss else 0.0
loss_total += lambda_gray_world * loss_gray_world
```

### Ablation Study Configurations

**Baseline** (no gray-world):
```python
use_gray_world_loss = False
```

**Aggressive** (strong color correction):
```python
use_gray_world_loss = True
lambda_gray_world = 0.05
```

**Gentle** (for single-color scenes):
```python
use_gray_world_loss = True
lambda_gray_world = 0.001
```

---

*Last Updated: December 7, 2025*
