import os
# Comment out the hardcoded GPU setting to allow command-line override
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from nets.net import L_net, L_net_ViT, fusionnet
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import wandb


def gray_world_loss(image):
    """
    Gray-World Assumption Loss
    Assumption: In a normal natural image, the mean values of R, G, B channels should be approximately equal (gray).
    
    Formula: L_col = sum_{(p,q) in epsilon} (J_p - J_q)^2
             where epsilon = {(R,G), (R,B), (G,B)}
    
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


# Model Architecture Configuration
use_vit_L_net = False  # Set to True to use L_net_ViT (predicts illumination + color matrix)
use_color_matrix = True  # Set to True to enable color matrix prediction in fusionnet (from img1+img2)
use_L_net_color = True  # Set to True to enable color matrix prediction in L_net (CNN-based)
use_gamma_correction = False  # Set to True to enable gamma correction with Global_pred_with_gamma
gamma_type = 'exp'  # 'exp' for fixed gamma (exposure correction), other for learnable gamma
use_identity_regularization = False  # Set to True to add identity regularization to color matrix
lambda_color_reg = 0.001  # Weight for identity regularization (e.g., 0.001 to 0.01)
use_gray_world_loss = False  # Set to True to add gray-world assumption loss (balances RGB channels)
lambda_gray_world = 0.01  # Weight for gray-world loss (use low values like 0.001-0.01 for single-color scenes)

# Hyperparameters
coef=(1,0.5,0.1,0.2,0.1)
num_epochs =40

lr = 1e-4
step_size=10
gamma=0.1
weight_decay = 0
batch_size=16
GPU_number = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
training_data_path = r"data/SICE_training50"

# Wandb configuration (set to False to disable)
use_wandb = True
wandb_project = "Retinex-MEF"
wandb_name = time.strftime("%m_%d_%H_%M", time.localtime())
# Set your wandb API key here (or use environment variable WANDB_API_KEY)
# Get your API key from https://wandb.ai/authorize
wandb_api_key = ''  # Set to your API key string, or leave as None to use environment variable

# Initialize models based on configuration
net_fusion = fusionnet(use_color_matrix=use_color_matrix, use_gamma_correction=use_gamma_correction, gamma_type=gamma_type).to(device)
if use_vit_L_net:
    net_L = L_net_ViT(in_channels=3, dim=64, num_heads=4).to(device)
    print("Using L_net_ViT (ViT-based illumination + color matrix prediction)")
else:
    net_L = L_net(predict_color_matrix=use_L_net_color).to(device)
    if use_L_net_color:
        print("Using L_net (CNN-based) with color matrix prediction")
    else:
        print("Using original L_net (CNN-based illumination prediction only)")

optimizer1 = torch.optim.Adam(net_fusion.parameters(), lr=lr, weight_decay=weight_decay)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
optimizer2 = torch.optim.Adam(net_L.parameters(), lr=lr, weight_decay=weight_decay)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

trainloader = DataLoader(SICE_training(file_path=training_data_path),batch_size=batch_size, shuffle=True, num_workers=0)

exppath=os.path.join("exp",time.strftime("%m_%d_%H_%M", time.localtime()))
os.makedirs(exppath,exist_ok=True)
os.makedirs(os.path.join(exppath,'model'),exist_ok=True)

# Initialize Weights & Biases
if use_wandb:
    # Login with API key if provided
    if wandb_api_key is not None:
        wandb.login(key=wandb_api_key)
    
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "step_size": step_size,
            "gamma": gamma,
            "weight_decay": weight_decay,
            "patch_size": 128,
            "stride": 128,
            "coef_recon": coef[0],
            "coef_smooth": coef[1],
            "coef_initialize": coef[2],
            "coef_suppress": coef[3],
            "coef_consist": coef[4],
            "training_data_path": training_data_path,
            "use_vit_L_net": use_vit_L_net,
            "use_color_matrix": use_color_matrix,
        }
    )
    wandb.watch([net_fusion, net_L], log="all", log_freq=100)

# Print configuration summary
print("="*60)
print("Model Architecture:")
print(f"  L_net: {'L_net_ViT (ViT-based)' if use_vit_L_net else 'L_net (CNN-based)'}")
print(f"  L_net Color Matrix: {'Enabled' if use_L_net_color else 'Disabled'}")
print(f"  Fusion Network Color Matrix: {'Enabled (from img1+img2)' if use_color_matrix else 'Disabled'}")
print(f"  Gamma Correction: {'Enabled (type={})'.format(gamma_type) if use_gamma_correction else 'Disabled'}")
print(f"  Identity Regularization: {'Enabled (lambda={})'.format(lambda_color_reg) if use_identity_regularization else 'Disabled'}")
print(f"  Gray-World Loss: {'Enabled (lambda={})'.format(lambda_gray_world) if use_gray_world_loss else 'Disabled'}")
print("="*60)
print("GPU Configuration:")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    # print(f"CUDA_VISIBLE_DEVICES: {GPU_number}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("Using CPU")
print(f"Device: {device}")
print("="*60)

writer = SummaryWriter(os.path.join(exppath, 'tensorboard'))
log_file = open(os.path.join(exppath, 'training_log.txt'), 'w')
log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
log_file.write(f"Model Architecture:\n")
log_file.write(f"  L_net: {'L_net_ViT (ViT-based)' if use_vit_L_net else 'L_net (CNN-based)'}\n")
log_file.write(f"  L_net Color Matrix: {'Enabled' if use_L_net_color else 'Disabled'}\n")
log_file.write(f"  Fusion Network Color Matrix: {'Enabled (from img1+img2)' if use_color_matrix else 'Disabled'}\n")
log_file.write(f"  Identity Regularization: {'Enabled (lambda={})'.format(lambda_color_reg) if use_identity_regularization else 'Disabled'}\n")
log_file.write(f"  Gray-World Loss: {'Enabled (lambda={})'.format(lambda_gray_world) if use_gray_world_loss else 'Disabled'}\n")
log_file.write(f"Device: {device}\n")
if torch.cuda.is_available():
    log_file.write(f"CUDA_VISIBLE_DEVICES: {GPU_number}\n")
    log_file.write(f"Number of GPUs: {torch.cuda.device_count()}\n")
    for i in range(torch.cuda.device_count()):
        log_file.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
log_file.write(f"Batch size: {batch_size}, Epochs: {num_epochs}, Learning rate: {lr}\n")
log_file.write(f"Coefficients: {coef}\n\n")
log_file.flush()
prev_time = time.time()

for epoch in range(num_epochs):

    net_fusion.train()
    net_L.train()

    losslist_total=[]
    losslist_recon=[]
    losslist_smooth=[]
    losslist_initialize=[]
    losslist_suppress=[]
    losslist_consist=[]
    losslist_reg_color=[]
    losslist_gray_world=[]

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, (img1,img2,img3,index) in pbar:

        img1= img1.cuda()
        img2= img2.cuda()
        img3= img3.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # Forward pass through L_net
        if use_vit_L_net or use_L_net_color:
            L1, color_mat1 = net_L(img1)
            L2, color_mat2 = net_L(img2)
            L3, color_mat3 = net_L(img3)
            # Pass color matrix from img3 to fusionnet if L_net predicts it
            # Note: If use_gamma_correction=True, fusionnet will use its own color matrix instead
            fusion_output = net_fusion(img1, img2, L3, color_matrix_external=color_mat3)
        else:
            L1, L2, L3 = net_L(img1), net_L(img2), net_L(img3)
            # fusionnet will predict color matrix and/or gamma from img1+img2
            fusion_output = net_fusion(img1, img2, L3)
        
        # Handle fusion output (varies based on color_matrix and gamma_correction settings)
        # Gamma mode: (output, R, Rhat, gamma_val, color_matrix)
        # Color mode: (output, R, Rhat, color_matrix)
        # Plain mode: (output, R, Rhat)
        y3 = fusion_output[0]
        R3 = fusion_output[1]
        Rhat = fusion_output[2]

        loss_recon=F.l1_loss(y3,img3)
        loss_smooth=illu_smooth(L3,img3)
        loss_initialize=F.l1_loss(L3,torch.max(img3,1,keepdim=True)[0])
        loss_suppress=torch.mean(F.relu(Rhat*(L3.detach())-img3)+F.relu(Rhat*(L2.detach())-img2)+F.relu(Rhat*(L1.detach())-img1))
        loss_consist=F.l1_loss(Rhat,R3)
        
        # Identity regularization for color matrix
        loss_reg_color = torch.tensor(0.0).to(device)
        if use_identity_regularization and len(fusion_output) >= 4:
            # Get color matrix from output (position depends on mode)
            if use_gamma_correction:
                # Gamma mode: color_matrix is at position 4
                color_matrix_used = fusion_output[4]
            else:
                # Color mode: color_matrix is at position 3
                color_matrix_used = fusion_output[3]
            identity = torch.eye(3).to(device).unsqueeze(0).repeat(color_matrix_used.size(0), 1, 1)
            loss_reg_color = torch.norm(color_matrix_used - identity, p='fro')
        
        # Gray-world assumption loss (balances RGB channels of reflectance)
        # Applied to Rhat (color-corrected reflectance) instead of final output
        loss_gray_world = torch.tensor(0.0).to(device)
        if use_gray_world_loss:
            loss_gray_world = gray_world_loss(Rhat)
        
        loss_total=coef[0]*loss_recon+coef[1]*loss_smooth+coef[2]*loss_initialize+coef[3]*loss_suppress+coef[4]*loss_consist
        if use_identity_regularization and len(fusion_output) >= 4:
            loss_total = loss_total + lambda_color_reg * loss_reg_color
        if use_gray_world_loss:
            loss_total = loss_total + lambda_gray_world * loss_gray_world

        loss_total.backward()

        optimizer1.step()
        optimizer2.step()

        losslist_total.append(loss_total.item())
        losslist_recon.append(loss_recon.item())
        losslist_smooth.append(loss_smooth.item())
        losslist_initialize.append(loss_initialize.item())
        losslist_suppress.append(loss_suppress.item())
        losslist_consist.append(loss_consist.item())
        if use_identity_regularization and len(fusion_output) >= 4:
            losslist_reg_color.append(loss_reg_color.item())
        if use_gray_world_loss:
            losslist_gray_world.append(loss_gray_world.item())


        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        # Log losses to TensorBoard and wandb every 50 iterations
        if i % 50 == 0:
            writer.add_scalar('Loss/total', loss_total.item(), batches_done)
            writer.add_scalar('Loss/recon', loss_recon.item(), batches_done)
            writer.add_scalar('Loss/smooth', loss_smooth.item(), batches_done)
            writer.add_scalar('Loss/initialize', loss_initialize.item(), batches_done)
            writer.add_scalar('Loss/suppress', loss_suppress.item(), batches_done)
            writer.add_scalar('Loss/consist', loss_consist.item(), batches_done)
            if use_identity_regularization and len(fusion_output) >= 4:
                writer.add_scalar('Loss/color_reg', loss_reg_color.item(), batches_done)
            if use_gray_world_loss:
                writer.add_scalar('Loss/gray_world', loss_gray_world.item(), batches_done)
            
            if use_wandb:
                log_dict = {
                    "batch/loss_total": loss_total.item(),
                    "batch/loss_recon": loss_recon.item(),
                    "batch/loss_smooth": loss_smooth.item(),
                    "batch/loss_initialize": loss_initialize.item(),
                    "batch/loss_suppress": loss_suppress.item(),
                    "batch/loss_consist": loss_consist.item(),
                    "batch_step": batches_done,
                }
                if use_identity_regularization and len(fusion_output) >= 4:
                    log_dict["batch/loss_color_reg"] = loss_reg_color.item()
                if use_gray_world_loss:
                    log_dict["batch/loss_gray_world"] = loss_gray_world.item()
                wandb.log(log_dict)
        
        # Log demo images every 1000 iterations
        if batches_done % 1000 == 0 and batches_done > 0:
            net_fusion.eval()
            net_L.eval()
            with torch.no_grad():
                # Use current batch for visualization
                vis_img1 = img1[:4].clamp(0, 1)
                vis_img2 = img2[:4].clamp(0, 1)
                vis_img3 = img3[:4].clamp(0, 1)
                
                # Get illumination maps
                if use_vit_L_net or use_L_net_color:
                    vis_L1, vis_color1 = net_L(vis_img1)
                    vis_L2, vis_color2 = net_L(vis_img2)
                    vis_L3, vis_color3 = net_L(vis_img3)
                    vis_fusion_output = net_fusion(vis_img1, vis_img2, vis_L3, color_matrix_external=vis_color3)
                else:
                    vis_L1, vis_L2, vis_L3 = net_L(vis_img1), net_L(vis_img2), net_L(vis_img3)
                    vis_fusion_output = net_fusion(vis_img1, vis_img2, vis_L3)
                
                # Get fusion output (handle different output lengths)
                # Gamma mode: (output, R, Rhat, gamma_val, color_matrix) = 5
                # Color mode: (output, R, Rhat, color_matrix) = 4
                # Plain mode: (output, R, Rhat) = 3
                vis_y3 = vis_fusion_output[0]
                vis_R3 = vis_fusion_output[1]
                vis_Rhat = vis_fusion_output[2]
                
                # Log to TensorBoard
                writer.add_images('Iteration/input_dark', vis_img1, batches_done)
                writer.add_images('Iteration/input_bright', vis_img2, batches_done)
                writer.add_images('Iteration/input_reference', vis_img3, batches_done)
                writer.add_images('Iteration/output_fused', vis_y3.clamp(0, 1), batches_done)
                writer.add_images('Iteration/reflectance', vis_R3.clamp(0, 1), batches_done)
                writer.add_images('Iteration/illumination_L3', vis_L3.repeat(1, 3, 1, 1).clamp(0, 1), batches_done)
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "iter_images/input_dark": [wandb.Image(vis_img1[j].cpu()) for j in range(len(vis_img1))],
                        "iter_images/input_bright": [wandb.Image(vis_img2[j].cpu()) for j in range(len(vis_img2))],
                        "iter_images/input_reference": [wandb.Image(vis_img3[j].cpu()) for j in range(len(vis_img3))],
                        "iter_images/output_fused": [wandb.Image(vis_y3[j].clamp(0, 1).cpu()) for j in range(len(vis_y3))],
                        "iter_images/reflectance": [wandb.Image(vis_R3[j].clamp(0, 1).cpu()) for j in range(len(vis_R3))],
                        "iter_images/illumination_L3": [wandb.Image(vis_L3[j].repeat(3, 1, 1).clamp(0, 1).cpu()) for j in range(len(vis_L3))],
                        "iteration": batches_done,
                    })
            
            net_fusion.train()
            net_L.train()
        
        # Update progress bar
        pbar.set_postfix({
            'loss_total': f"{np.mean(losslist_total):.4f}",
            'loss_recon': f"{np.mean(losslist_recon):.4f}",
            'loss_smooth': f"{np.mean(losslist_smooth):.4f}",
            'ETA': str(time_left).split('.')[0]
        })


    # Log epoch-level metrics to TensorBoard
    writer.add_scalar('Epoch/loss_total', np.mean(losslist_total), epoch)
    writer.add_scalar('Epoch/loss_recon', np.mean(losslist_recon), epoch)
    writer.add_scalar('Epoch/loss_smooth', np.mean(losslist_smooth), epoch)
    writer.add_scalar('Epoch/loss_initialize', np.mean(losslist_initialize), epoch)
    writer.add_scalar('Epoch/loss_suppress', np.mean(losslist_suppress), epoch)
    writer.add_scalar('Epoch/loss_consist', np.mean(losslist_consist), epoch)
    if use_identity_regularization and len(losslist_reg_color) > 0:
        writer.add_scalar('Epoch/loss_color_reg', np.mean(losslist_reg_color), epoch)
    if use_gray_world_loss and len(losslist_gray_world) > 0:
        writer.add_scalar('Epoch/loss_gray_world', np.mean(losslist_gray_world), epoch)
    writer.add_scalar('Epoch/learning_rate', scheduler1.get_last_lr()[0], epoch)
    
    # Log epoch metrics to wandb
    if use_wandb:
        epoch_log = {
            "epoch": epoch + 1,
            "epoch/loss_total": np.mean(losslist_total),
            "epoch/loss_recon": np.mean(losslist_recon),
            "epoch/loss_smooth": np.mean(losslist_smooth),
            "epoch/loss_initialize": np.mean(losslist_initialize),
            "epoch/loss_suppress": np.mean(losslist_suppress),
            "epoch/loss_consist": np.mean(losslist_consist),
            "epoch/learning_rate": scheduler1.get_last_lr()[0],
        }
        if use_identity_regularization and len(losslist_reg_color) > 0:
            epoch_log["epoch/loss_color_reg"] = np.mean(losslist_reg_color)
        if use_gray_world_loss and len(losslist_gray_world) > 0:
            epoch_log["epoch/loss_gray_world"] = np.mean(losslist_gray_world)
        wandb.log(epoch_log)
    
    # Log demo images every 5 epochs
    if epoch % 5 == 0:
        net_fusion.eval()
        net_L.eval()
        with torch.no_grad():
            # Get first batch for visualization
            demo_img1, demo_img2, demo_img3, _ = next(iter(trainloader))
            demo_img1 = demo_img1.cuda()
            demo_img2 = demo_img2.cuda()
            demo_img3 = demo_img3.cuda()
            
            # Get illumination maps
            if use_vit_L_net or use_L_net_color:
                demo_L1, demo_color1 = net_L(demo_img1)
                demo_L2, demo_color2 = net_L(demo_img2)
                demo_L3, demo_color3 = net_L(demo_img3)
                demo_fusion_output = net_fusion(demo_img1, demo_img2, demo_L3, color_matrix_external=demo_color3)
            else:
                demo_L1, demo_L2, demo_L3 = net_L(demo_img1), net_L(demo_img2), net_L(demo_img3)
                demo_fusion_output = net_fusion(demo_img1, demo_img2, demo_L3)
            
            # Get fusion output
            if len(demo_fusion_output) == 4:
                demo_y3, demo_R3, demo_Rhat, _ = demo_fusion_output
            else:
                demo_y3, demo_R3, demo_Rhat = demo_fusion_output
            
            # Log input images (show first 4 images from batch)
            writer.add_images('Images/input_dark', demo_img1[:4].clamp(0, 1), epoch)
            writer.add_images('Images/input_bright', demo_img2[:4].clamp(0, 1), epoch)
            writer.add_images('Images/input_reference', demo_img3[:4].clamp(0, 1), epoch)
            
            # Log output and illumination maps
            writer.add_images('Images/output_fused', demo_y3[:4].clamp(0, 1), epoch)
            writer.add_images('Images/reflectance', demo_R3[:4].clamp(0, 1), epoch)
            writer.add_images('Images/reflectance_hat', demo_Rhat[:4].clamp(0, 1), epoch)
            writer.add_images('Images/illumination_L3', demo_L3[:4].repeat(1, 3, 1, 1).clamp(0, 1), epoch)
            
            # Log images to wandb
            if use_wandb:
                wandb.log({
                    "images/input_dark": [wandb.Image(demo_img1[j].clamp(0, 1).cpu()) for j in range(min(4, len(demo_img1)))],
                    "images/input_bright": [wandb.Image(demo_img2[j].clamp(0, 1).cpu()) for j in range(min(4, len(demo_img2)))],
                    "images/input_reference": [wandb.Image(demo_img3[j].clamp(0, 1).cpu()) for j in range(min(4, len(demo_img3)))],
                    "images/output_fused": [wandb.Image(demo_y3[j].clamp(0, 1).cpu()) for j in range(min(4, len(demo_y3)))],
                    "images/reflectance": [wandb.Image(demo_R3[j].clamp(0, 1).cpu()) for j in range(min(4, len(demo_R3)))],
                    "images/reflectance_hat": [wandb.Image(demo_Rhat[j].clamp(0, 1).cpu()) for j in range(min(4, len(demo_Rhat)))],
                    "images/illumination_L3": [wandb.Image(demo_L3[j].repeat(3, 1, 1).clamp(0, 1).cpu()) for j in range(min(4, len(demo_L3)))],
                    "epoch": epoch + 1,
                })
        
        net_fusion.train()
        net_L.train()
    
    # Log epoch summary
    log_message = (
        f"[Epoch {epoch+1}/{num_epochs}] "
        f"loss_total: {np.mean(losslist_total):.6f}, "
        f"loss_recon: {np.mean(losslist_recon):.6f}, "
        f"loss_smooth: {np.mean(losslist_smooth):.6f}, "
        f"loss_initialize: {np.mean(losslist_initialize):.6f}, "
        f"loss_suppress: {np.mean(losslist_suppress):.6f}, "
        f"loss_consist: {np.mean(losslist_consist):.6f}, "
        f"LR: {scheduler1.get_last_lr()[0]:.6f}\n"
    )
    log_file.write(log_message)
    log_file.flush()
    print(log_message.strip())
    
    # adjust the learning rate and save model                
    scheduler1.step()  
    scheduler2.step() 

    if epoch % 1 == 0:
        checkpoint = {
                    'net_fusion': net_fusion.state_dict(),
                    'net_L': net_L.state_dict(),
                    }
        torch.save(checkpoint, os.path.join(exppath,'model', 'ckpt_%s.pth' % (str(epoch+1))))

log_file.write(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
log_file.close()
writer.close()

if use_wandb:
    wandb.finish()

print(f"Training log saved to {os.path.join(exppath, 'training_log.txt')}")
print(f"TensorBoard logs saved to {os.path.join(exppath, 'tensorboard')}")
print(f"Run 'tensorboard --logdir={os.path.join(exppath, 'tensorboard')}' to view")
if use_wandb:
    print(f"Wandb logs available at: {wandb.run.url if wandb.run else 'N/A'}")