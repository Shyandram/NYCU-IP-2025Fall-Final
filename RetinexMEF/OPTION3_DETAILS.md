# Color Matrix Prediction Architecture

## Overview

This document describes the **ColorMatrixPredictor** architecture - an independent module that predicts ideal 3×3 color correction matrices from two exposure images (underexposed + overexposed). This design addresses the **unbalanced color values in under-exposure** problem by learning cross-channel color correction.

### Physical Motivation

**Problem**: Under-exposure causes RGB channel imbalance due to insufficient photons. Traditional Retinex (I = L × R) only restores brightness, not color balance.

**Solution**: A learnable 3×3 color matrix enables cross-channel mixing to borrow information from healthy channels to repair damaged ones.

### Configuration Flags
```python
use_vit_L_net = False                 # Use CNN-based L_net
use_color_matrix = True               # Fusionnet predicts color matrix from img1+img2
use_L_net_color = False               # L_net does NOT predict color matrix
use_identity_regularization = True    # Enable identity regularization
lambda_color_reg = 0.01               # Regularization weight
```

---

## Architecture Components

### 1. L_net (CNN-based with Color Predictor)

**Purpose:** Predicts illumination map + 3×3 color transformation matrix

**Structure:**
- **Illumination Branch:** 
  - 4 convolutional layers (ReflectionPad + Conv + ReLU)
  - Final sigmoid activation → outputs L ∈ [0,1]
  - Output shape: [B, 1, H, W]

- **Color Matrix Branch (Global_pred - moderate version):**
  - Conv1: 3→16 channels, stride=2 (H/2, W/2)
  - Conv2: 16→32 channels, stride=2 (H/4, W/4)
  - AdaptiveAvgPool: 32 channels → [B, 32, 1, 1]
  - MLP: 32→64→9 (reshape to 3×3 matrix)
  - Zero initialization for stable training
  - Identity matrix base: `color_matrix = I + offset`
  - Output shape: [B, 3, 3]

**Parameters:** ~20K for color predictor, baseline L_net unchanged

### 2. Fusionnet (Without Internal Color Prediction)

**Purpose:** Refines reflectance using external color matrix

**Key Operations:**
1. **SRE Module:** Extracts initial reflectance `Rhat` from input images
2. **Color Correction (if color_matrix_external provided):**
   ```python
   Rhat_flat = Rhat.view(B, 3, H*W)  # Flatten spatial dims
   Rhat_corrected = color_matrix @ Rhat_flat  # Apply 3×3 transform
   Rhat = Rhat_corrected.view(B, 3, H, W)
   Rhat = clamp(Rhat, 0, 1)  # Prevent overflow
   ```
3. **Transformer Decoder:** Refines Rhat → R using illumination L
4. **Final Output:** `R * L` (reflectance × illumination)

---

## Training Strategy

### Input Data
- `img1`: Underexposed image
- `img2`: Overexposed image  
- `img3`: **Reference (well-exposed) image** - ground truth

### Forward Pass

```python
# Step 1: Predict illumination + color matrix for all images
L1, color_mat1 = net_L(img1)  # From underexposed
L2, color_mat2 = net_L(img2)  # From overexposed
L3, color_mat3 = net_L(img3)  # From reference ✓

# Step 2: Pass ONLY reference color matrix to fusionnet
y3, R3, Rhat = net_fusion(img1, img2, L3, color_matrix_external=color_mat3)
```

**Key Design Decision:**
- Only `color_mat3` (from reference image img3) is used for color correction
- Rationale: The reference image has "correct" exposure and color characteristics
- `color_mat1` and `color_mat2` are predicted but not used during training

### Loss Functions

```python
# Coefficients: (1.0, 0.5, 0.1, 0.2, 0.1)

1. loss_recon = L1_loss(y3, img3)  # Reconstruction loss
2. loss_smooth = illu_smooth(L3, img3)  # Illumination smoothness
3. loss_initialize = L1_loss(L3, max(img3, dim=1))  # Illumination initialization
4. loss_suppress = ReLU(Rhat*L3 - img3) + ReLU(Rhat*L2 - img2) + ReLU(Rhat*L1 - img1)
5. loss_consist = L1_loss(Rhat, R3)  # Consistency between initial and refined R

loss_total = 1.0*loss_recon + 0.5*loss_smooth + 0.1*loss_initialize + 
             0.2*loss_suppress + 0.1*loss_consist
```

**Color Matrix Supervision:**
- **No explicit supervision** for the color matrix itself
- Learned **implicitly** through `loss_recon` (reconstruction loss)
- The model learns to predict a color matrix that helps produce the correct output
- Identity initialization ensures gradual, stable learning

### Training Hyperparameters

```python
epochs = 40
batch_size = 16
learning_rate = 1e-4
optimizer = Adam
scheduler = StepLR(step_size=10, gamma=0.1)  # LR decay every 10 epochs
patch_size = 128×128
```

---

## Inference Strategy

### Challenge
During inference, we only have:
- `img1`: Underexposed image
- `img2`: Overexposed image
- **NO img3 (reference image)** ❌

### Solution: Synthetic Reference Approach

```python
# Step 1: Create synthetic reference by averaging input images
img_synth = (img1 + img2) / 2  # Combines under/over exposure

# Step 2: Predict color matrix from synthetic reference
_, color_mat_synth = net_L(img_synth)

# Step 3: Predict illuminations
L1, _ = net_L(img1)
L2, _ = net_L(img2)
L = (L1 + L2) / 2  # Fused illumination

# Step 4: Run fusionnet with synthetic color matrix
Ihat, R, Rhat = net_fusion(img1, img2, L, color_matrix_external=color_mat_synth)

# Step 5: Output final image
output = Ihat  # Already R*L, ready to save
```

**Rationale:**
- Synthetic reference mimics the "well-exposed" characteristics
- Approximates what img3 would be during training
- More principled than averaging color matrices (color_mat1 + color_mat2)/2
- Maintains consistency with training behavior

---

## Data Flow Diagram

### Training Flow
```
img1 (under) ──┐
               ├──→ SRE ──→ Rhat
img2 (over)  ──┘                │
                                ↓
img3 (ref) ────→ L_net ──→ color_mat3 ──→ Apply to Rhat
                    │                            │
                    └──→ L3 ──────────────→ Transformer Decoder
                                                 │
                                                 ↓
                                         R (refined reflectance)
                                                 │
                                                 ↓
                                            y3 = R * L3
                                                 │
                                                 ↓
                                         Loss vs img3
```

### Inference Flow
```
img1 (under) ──┬──→ L_net ──→ L1 ──┐
               │                     ├──→ L = (L1+L2)/2
img2 (over)  ──┼──→ L_net ──→ L2 ──┘         │
               │                              │
               ├──→ (img1+img2)/2 = img_synth │
               │            │                  │
               │            ↓                  │
               │    L_net ──→ color_mat_synth │
               │                    │          │
               └──→ SRE ──→ Rhat ◄──┘          │
                              │                │
                              ↓                │
                      color_corrected_Rhat    │
                              │                │
                              ↓                │
                      Transformer Decoder ◄───┘
                              │
                              ↓
                        R (refined)
                              │
                              ↓
                        Ihat = R * L
```

---

## Key Advantages

### 1. **Computational Efficiency**
- CNN-based L_net is faster than ViT (~10x speedup)
- Moderate Global_pred (~20K params) vs complex version (~50K params)
- Suitable for real-time applications

### 2. **Color Correction Capability**
- Addresses color cast issues from multi-exposure fusion
- Adaptive 3×3 transformation learned from data
- Identity initialization prevents catastrophic changes

### 3. **Stable Training**
- Implicit supervision through reconstruction loss
- Zero-initialized offset ensures gradual learning
- Clamping prevents numerical overflow

### 4. **Flexible Architecture**
- Can switch between options by changing 3 boolean flags
- Easy to ablate color matrix contribution
- Modular design for future improvements

---

## Performance Characteristics

### Expected Results
- **Loss (epoch 40):** ~0.030-0.032 (comparable to baseline 0.0301)
- **Training Time:** ~2 hours for 40 epochs (single RTX 5060 Ti)
- **Memory Usage:** ~4GB GPU memory for batch_size=16

### Comparison with Other Options

| Metric | Option 1 (Baseline) | Option 3 (L_net Color) | Option 4 (ViT) |
|--------|---------------------|------------------------|----------------|
| Loss | 0.0301 | ~0.030-0.032 | 0.0572 |
| Speed | 1.0x | 1.05x | 0.1x |
| Params | ~1.5M | ~1.52M | ~2.5M |
| Color Correction | ❌ | ✓ | ✓ |

---

## Implementation Details

### Color Matrix Properties

**Initialization:**
```python
color_base = nn.Parameter(torch.eye(3))  # Identity matrix
color_offset = MLP(features) → [B, 9] → reshape [B, 3, 3]
color_matrix = color_base + color_offset
```

**Mathematical Operation:**
For each pixel with RGB values `[r, g, b]`:
```
[r']   [m11 m12 m13]   [r]
[g'] = [m21 m22 m23] @ [g]
[b']   [m31 m32 m33]   [b]
```

**Clamping:**
- After color correction: `Rhat = clamp(Rhat, 0, 1)`
- After decoder refinement: `R = clamp(Rhat + decoder_output, 0, 1)`

### Test Configuration

**File:** `test.py`
```python
use_vit_L_net = False
use_color_matrix = False
use_L_net_color = True

# Model checkpoint trained with Option 3
path_model = "exp/12_03_16_27/model/ckpt_40.pth"
```

**Output:**
- Uses `Ihat` directly (already R*L)
- Saves as uint8 images (0-255 range)
- Output directory: `test_case/results_L_net_color/`

---

## Potential Improvements

### 1. **Explicit Color Matrix Regularization**
```python
# Add to loss function:
loss_reg = torch.norm(color_matrix - torch.eye(3), p='fro')
loss_total += lambda_reg * loss_reg  # lambda_reg = 0.01
```

### 2. **Multi-Scale Color Prediction**
- Predict color matrices at different resolutions
- Apply hierarchical color correction

### 3. **Attention-Weighted Color Correction**
- Learn per-pixel weights for color matrix application
- Adaptive correction based on local content

### 4. **Color Matrix Visualization**
- Log color matrix values during training
- Visualize evolution from identity to learned transform
- Analyze which channels are most adjusted

---

## Troubleshooting

### Issue: Dark Output Images
**Cause:** Using `L*Rhat` instead of final output `Ihat`
**Solution:** Change `imgout = L*Rhat` to `imgout = Ihat`

### Issue: Color Cast or Oversaturation
**Cause:** Color matrix values drifting too far from identity
**Solution:** Add regularization or reduce learning rate for color predictor

### Issue: Model Not Converging
**Cause:** Color matrix initialization or zero-init not working
**Solution:** Check `nn.init.zeros_()` on final MLP layer

### Issue: Dimension Mismatch
**Cause:** L and R have different spatial sizes
**Solution:** Already handled by bilinear interpolation in code

---

## References

### Key Code Files
- `nets/net.py`: L_net (lines 160-197), Global_pred (lines 98-159), fusionnet (lines 298-370)
- `train.py`: Training loop with Option 3 configuration
- `test.py`: Inference with synthetic reference approach

### Model Checkpoint
- **Path:** `exp/12_03_16_27/model/ckpt_40.pth`
- **Training Started:** 2025-12-03 16:27:19
- **Configuration:** L_net (CNN-based), Fusion Network Color Matrix: Disabled, L_net Color: Enabled
- **Final Loss:** 0.0304 (epoch 40)

---

*Last Updated: December 7, 2025*
