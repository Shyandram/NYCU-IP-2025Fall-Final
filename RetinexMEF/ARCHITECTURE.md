# Retinex-MEF Architecture Documentation

## Overview

This project implements a Retinex-based Multi-Exposure Fusion (MEF) network for combining under-exposed and over-exposed images into a well-exposed result. The architecture supports multiple configurations with optional color correction capabilities.

## Core Architecture Flow

```
Input: Under-exposed (img1) + Over-exposed (img2) + Reference (img3)
                                    |
                    +---------------+---------------+
                    |                               |
                 L_net                          fusionnet
            (Illumination)                   (Reflectance + Fusion)
                    |                               |
              Predicts L1, L2, L3         img1 + img2 → SRE → Rhat
                    |                               |
              L = (L1 + L2) / 2          [Optional: Color Matrix]
                    |                               |
                    +---------------+---------------+
                                    |
                            Final Output: I = L × R
```

## Module Breakdown

### 1. **L_net** (Illumination Network)

**Purpose**: Predicts smooth illumination maps from input images.

**Architecture**:
- 4 convolutional layers with ReLU activation
- ReflectionPad2d for edge handling
- Sigmoid output for [0,1] range
- Optional color matrix prediction branch

**Input**: RGB image [B, 3, H, W]

**Output**: 
- Standard: Illumination map L [B, 1, H, W]
- With color matrix: (L [B, 1, H, W], color_matrix [B, 3, 3])

**Variants**:
- `L_net(predict_color_matrix=False)`: Original CNN-based (default)
- `L_net(predict_color_matrix=True)`: CNN + simple color predictor
- `L_net_ViT()`: ViT-based with dual outputs (illumination + color)

```python
# Standard usage
net_L = L_net()
L = net_L(img)  # [B, 1, H, W]

# With color matrix
net_L = L_net(predict_color_matrix=True)
L, color_matrix = net_L(img)  # [B, 1, H, W], [B, 3, 3]
```

---

### 2. **SRE** (Shared Reflectance Estimator)

**Purpose**: Extracts shared reflectance information from two input images.

**Architecture**:
- Concatenates two images: [img1, img2] → [B, 6, H, W]
- 1 conv layer + 2 TransformerBlocks (Restormer)
- Sigmoid activation for reflectance output

**Input**: Two RGB images [B, 3, H, W] each

**Output**: 
- Rhat: Reflectance estimate [B, 3, H, W]
- R: Intermediate features [B, 32, H, W]

```python
Rhat, R = SRE(img1, img2)
```

---

### 3. **fusionnet** (Fusion Network)

**Purpose**: Fuses reflectance and illumination with optional color correction.

**Architecture Flow**:

```
img1, img2 → SRE → Rhat [B, 3, H, W]
                     |
         [Optional: Color Correction]
              color_matrix @ Rhat
                     |
           patch_embed → R [B, 64, H, W]
                     |
          6× TransformerBlock Decoder
           (with L guidance: R + decoder(R) * L)
                     |
              output_conv → R'
                     |
          Final: R = clamp(Rhat + R')
                     |
               Output: I = R × L
```

**Key Features**:

1. **Color Matrix Application** (optional):
   - Applied to Rhat before decoder
   - Can use external matrix (from L_net) or predict internally
   - Transformation: `Rhat_corrected = color_matrix @ Rhat`
   - Applied in both training and inference

2. **Decoder with Illumination Guidance**:
   - 6 cross-attention TransformerBlocks
   - Each block: `R = R + decoder[k](R) * L`
   - L is resized to match R dimensions if needed

3. **Residual Connection**:
   - Final output: `R = Rhat + decoder_output`
   - Preserves original reflectance structure

**Input**: 
- img1, img2: Input images [B, 3, H, W]
- L: Illumination map [B, 1, H, W]
- color_matrix_external (optional): [B, 3, 3]

**Output**:
- Without color: (I, R, Rhat)
- With color: (I, R, Rhat, color_matrix)

Where:
- I = R × L (final fused image)
- R = final reflectance
- Rhat = initial reflectance estimate

---

### 4. **Global_pred** (Color Matrix Predictor)

**Purpose**: Predicts global 3×3 color transformation matrix.

**Two Variants**:

#### Simple Version (`simple=True`) - **Default**
```
Input RGB [B, 3, H, W]
      ↓
Global Average Pooling → [B, 3]
      ↓
MLP: 3 → 32 → 9
      ↓
Reshape to [B, 3, 3]
      ↓
Add to Identity Matrix Base
      ↓
Color Matrix [B, 3, 3]
```

- **Fast**: Only ~300 parameters
- **Stable**: Zero-initialized (starts as identity)
- **Efficient**: Single forward pass

#### Complex Version (`simple=False`)
```
Input RGB [B, 3, H, W]
      ↓
Conv Embedding (4× downsample) → [B, 64, H/4, W/4]
      ↓
Query-based Self-Attention Block (10 learnable queries)
      ↓
Extract 9 tokens → Linear → [B, 9]
      ↓
Reshape + Add to Base → [B, 3, 3]
```

- **Complex**: ~50K+ parameters
- **Powerful**: Attention-based global reasoning
- **Slower**: Multiple attention operations

---

## Configuration Options

The architecture supports **5 main configurations** through flags:

### Training Configuration (`train.py`)

```python
use_vit_L_net = False    # Use ViT-based L_net
use_color_matrix = False # Enable fusionnet color predictor
use_L_net_color = False  # Enable L_net color predictor
```

### Configuration Matrix

| Option | use_vit_L_net | use_L_net_color | use_color_matrix | Description |
|--------|---------------|-----------------|------------------|-------------|
| **1** (Baseline) | False | False | False | Original CNN L_net, no color correction |
| **2** | False | False | True | CNN L_net + fusionnet predicts color matrix |
| **3** | False | True | False | CNN L_net predicts color matrix |
| **4** | True | N/A | False | ViT L_net predicts illumination + color matrix |
| **5** | True | N/A | True | ViT L_net + fusionnet both predict color (not recommended) |

---

## Complete Data Flow Examples

### Option 1: Baseline (Simplest)

```
img1, img2, img3
      ↓
   L_net(img1) → L1 [B, 1, H, W]
   L_net(img2) → L2 [B, 1, H, W]
   L_net(img3) → L3 [B, 1, H, W]
      ↓
   L = (L1 + L2) / 2
      ↓
   fusionnet(img1, img2, L)
      ↓
   SRE(img1, img2) → Rhat
      ↓
   Decoder(Rhat, L) → R
      ↓
   Output: I = R × L
```

### Option 3: L_net with Color Matrix

```
img1, img2, img3
      ↓
   L_net(img1) → (L1, color_mat1)
   L_net(img2) → (L2, color_mat2)
   L_net(img3) → (L3, color_mat3)
      ↓
   L = (L1 + L2) / 2
      ↓
   fusionnet(img1, img2, L, color_matrix_external=color_mat3)
      ↓
   SRE(img1, img2) → Rhat
      ↓
   Rhat_corrected = color_mat3 @ Rhat  # Color correction
      ↓
   Decoder(Rhat_corrected, L) → R
      ↓
   Output: I = R × L
```

---

## Loss Functions

The training uses 5 loss components:

1. **loss_recon**: Reconstruction loss between output and reference
   ```python
   loss_recon = MSE(output, img3)
   ```

2. **loss_smooth**: Smoothness constraint on illumination
   ```python
   loss_smooth = illu_smooth(L1) + illu_smooth(L2) + illu_smooth(L3)
   ```

3. **loss_initialize**: Initialization loss for reflectance
   ```python
   loss_initialize = MSE(Rhat, img3/L3)
   ```

4. **loss_suppress**: Suppresses extreme illumination values
   ```python
   loss_suppress = Mean((L - 0.5)^2)
   ```

5. **loss_consist**: Consistency between reflectance estimates
   ```python
   loss_consist = MSE(R_from_different_pairs)
   ```

**Total Loss**:
```python
loss_total = coef[0] * loss_recon + 
             coef[1] * loss_smooth + 
             coef[2] * loss_initialize + 
             coef[3] * loss_suppress + 
             coef[4] * loss_consist
```

Default coefficients: `(1, 0.5, 0.1, 0.2, 0.1)`

---

## Color Matrix Details

### What is a Color Matrix?

A 3×3 matrix that transforms RGB colors:

```python
[R']   [m11 m12 m13]   [R]
[G'] = [m21 m22 m23] × [G]
[B']   [m31 m32 m33]   [B]
```

### Initialization

- **color_base**: Learnable parameter initialized as identity matrix
- **color_offset**: Predicted offset, initialized to zero
- **Final matrix**: `color_matrix = color_base + color_offset`

This ensures the network starts with no color change (identity transformation).

### Application

Applied pixel-wise to reflectance:

```python
Rhat_flat = Rhat.view(B, 3, H*W)  # [B, 3, H*W]
Rhat_corrected = color_matrix @ Rhat_flat  # [B, 3, 3] @ [B, 3, H*W]
Rhat_corrected = Rhat_corrected.view(B, 3, H, W)
```

### Purpose

- Adjust white balance
- Correct color cast
- Modify color temperature
- Enhance saturation/contrast
- Learned end-to-end from reconstruction loss

---

## Model Parameters

### Component Sizes

| Component | Parameters | Description |
|-----------|-----------|-------------|
| L_net | ~130K | 4 conv layers (64 channels) |
| L_net + Simple Color | ~130.3K | L_net + Global_pred(simple) |
| L_net_ViT | ~200K+ | ViT-based with dual outputs |
| SRE | ~80K | 2 TransformerBlocks (32 dim) |
| fusionnet (base) | ~1.2M | 6 TransformerBlocks (64 dim) |
| fusionnet + Simple Color | ~1.2M | fusionnet + Global_pred(simple) |
| Global_pred (simple) | ~300 | Pooling + 2 linear layers |
| Global_pred (complex) | ~50K | Conv + Attention + Linear |

**Total Network Size**:
- Option 1 (Baseline): ~1.4M parameters
- Option 3 (L_net + color): ~1.4M parameters
- Option 4 (ViT): ~1.5M parameters

---

## Inference Pipeline

### Standard Inference (test.py)

```python
# Setup
net_L = L_net(predict_color_matrix=use_L_net_color)
net_fusion = fusionnet(use_color_matrix=use_color_matrix)
net_L.eval()
net_fusion.eval()

# Process
with torch.no_grad():
    # Predict illumination
    if use_L_net_color:
        L1, color_mat1 = net_L(img1)
        L2, color_mat2 = net_L(img2)
    else:
        L1, L2 = net_L(img1), net_L(img2)
    
    # Fuse illumination
    L = (L1 + L2) / 2
    
    # Fusion
    if use_L_net_color:
        color_mat_avg = (color_mat1 + color_mat2) / 2
        output = net_fusion(img1, img2, L, color_mat_avg)
    else:
        output = net_fusion(img1, img2, L)
    
    # Extract result
    if len(output) == 4:
        I, R, Rhat, _ = output
    else:
        I, R, Rhat = output
    
    # Save final result
    result = (L * Rhat).cpu().numpy()
```

---

## Best Practices & Recommendations

### 1. **Start with Baseline (Option 1)**
   - Simplest architecture
   - Fastest training
   - Proven performance (loss: 0.0301)
   - No color correction overhead

### 2. **Try Option 3 for Color Enhancement**
   - If baseline results have color issues
   - Minimal parameter increase
   - Simple color predictor (fast)
   - Good balance of complexity/performance

### 3. **Avoid ViT Unless Necessary**
   - ViT (Option 4) performed worse (loss: 0.0572)
   - Much slower training
   - More parameters
   - May need more data

### 4. **Training Tips**
   - Use learning rate scheduling (StepLR with γ=0.1)
   - Start with lr=1e-4
   - Monitor all 5 loss components
   - Use tensorboard/wandb for visualization
   - Save checkpoints every epoch

### 5. **Data Preprocessing**
   - Resize large images (max 1024px)
   - Extract 128×128 patches
   - Random sampling for diversity
   - Normalize to [0, 1]

---

## File Structure

```
Retinex-MEF/
├── nets/
│   ├── net.py           # Main network architectures
│   └── restormer.py     # TransformerBlock implementation
├── train.py             # Training script
├── test.py              # Inference script
├── utils.py             # Loss functions and utilities
├── data/
│   ├── SICE_processing.py  # Data preprocessing
│   └── SICE_training50/    # Training patches
└── model/
    └── ckpt.pth         # Model checkpoint
```

---

## Usage Examples

### Training

```bash
# Option 1: Baseline
python train.py  # Default: all flags False

# Option 3: L_net with color matrix
# Edit train.py: use_L_net_color = True
python train.py

# Specify GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Testing

```bash
# Edit test.py to match training configuration
# Set: use_vit_L_net, use_color_matrix, use_L_net_color
python test.py
```

### Quick Unit Test

```bash
cd nets
python net.py  # Runs unit_test() function
```

---

## Performance Comparison

Based on 40 epochs training on SICE dataset:

| Configuration | Final Loss | Reconstruction Loss | Speed | Recommendation |
|---------------|-----------|-------------------|-------|----------------|
| Option 1 (Baseline) | 0.0301 | 0.0107 | ★★★★★ | **Best** |
| Option 2 (Fusion Color) | ~0.032* | ~0.011* | ★★★★☆ | Good |
| Option 3 (L_net Color) | ~0.032* | ~0.011* | ★★★★☆ | Good |
| Option 4 (ViT) | 0.0572 | 0.0186 | ★★☆☆☆ | Not recommended |
| Option 5 (ViT + Fusion) | ~0.058* | ~0.019* | ★★☆☆☆ | Not recommended |

*Estimated values (not yet trained)

---

## Citation

If you use this architecture, please cite the original Retinex-MEF paper and mention the architectural improvements.

---

## Future Improvements

1. **Supervised color correction**: Add ground truth color matrices
2. **Attention visualization**: Visualize what the network focuses on
3. **Lightweight variants**: MobileNet-based L_net
4. **Multi-scale fusion**: Process multiple resolutions
5. **Perceptual losses**: Add LPIPS or VGG-based losses

---

## Troubleshooting

### Common Issues

1. **Size mismatch between L and R**
   - Solution: Automatic interpolation implemented in fusionnet

2. **Color matrix causes artifacts**
   - Solution: Reduce learning rate, check initialization
   - Try simple=True for Global_pred

3. **Training unstable with ViT**
   - Solution: Lower learning rate to 5e-5
   - Increase batch size if possible

4. **Out of memory**
   - Solution: Reduce batch size
   - Use smaller patches (64×64)
   - Process fewer patches per image

---

**Last Updated**: December 2025
