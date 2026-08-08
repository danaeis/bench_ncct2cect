import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm

from data.vindr_dataset import get_vindr_dataloader
from models.vce_model import VCEModel
from utils.image_pool import ImagePool
from utils.visualization import save_image_grid

def parse_args():
    parser = argparse.ArgumentParser(description='Train VCE-CESM model on Vindr dataset')
    parser.add_argument('--config', type=str, default='configs/vindr_train_config.yaml',
                      help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """Create necessary directories for training."""
    for key, path in config['paths'].items():
        Path(path).mkdir(parents=True, exist_ok=True)

def get_dataloaders(config):
    """Create train and validation dataloaders."""
    # Create dataset
    dataset = get_vindr_dataloader(
        pairs_csv=os.path.join(config['dataset']['root_dir'], config['dataset']['pairs_csv']),
        batch_size=1,  # Will be batched after splitting
        num_workers=0,  # No workers for dataset creation
        phase='train',
        slice_selection=config['dataset']['slice_selection'],
        max_slices=config['dataset']['max_slices']
    ).dataset

    # Split into train and validation
    train_size = int(len(dataset) * config['dataset']['train_val_split'])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader

def train(config):
    """Main training function."""
    # Setup
    setup_directories(config)
    device = torch.device(f'cuda:{config["hardware"]["gpu_ids"][0]}' 
                        if torch.cuda.is_available() and config["hardware"]["gpu_ids"] 
                        else 'cpu')
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Create model
    model = VCEModel(config['model'])
    model.to(device)
    
    # Setup image pools for fake images
    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)
    
    # Create tensorboard writer
    if config['logging']['tensorboard']:
        writer = SummaryWriter(config['paths']['log_dir'])
    
    # Training loop
    total_iters = 0
    for epoch in range(config['training']['num_epochs']):
        model.train()
        
        # Training epoch
        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for i, batch in enumerate(pbar):
                total_iters += 1
                
                # Get data
                real_A = batch['input'].to(device)
                real_B = batch['target'].to(device)
                
                # Forward pass
                fake_B = model.netG_A(real_A)  # G_A(A)
                rec_A = model.netG_B(fake_B)   # G_B(G_A(A))
                fake_A = model.netG_B(real_B)  # G_B(B)
                rec_B = model.netG_A(fake_A)   # G_A(G_B(B))
                
                # Identity loss
                if config['training']['lambda_identity'] > 0:
                    idt_A = model.netG_A(real_B)
                    loss_idt_A = model.criterionIdt(idt_A, real_B) * config['training']['lambda_identity']
                    idt_B = model.netG_B(real_A)
                    loss_idt_B = model.criterionIdt(idt_B, real_A) * config['training']['lambda_identity']
                else:
                    loss_idt_A = 0
                    loss_idt_B = 0
                
                # GAN loss
                pred_fake = model.netD_A(fake_B)
                loss_G_A = model.criterionGAN(pred_fake, True)
                pred_fake = model.netD_B(fake_A)
                loss_G_B = model.criterionGAN(pred_fake, True)
                
                # Cycle loss
                loss_cycle_A = model.criterionCycle(rec_A, real_A) * config['training']['lambda_cycle']
                loss_cycle_B = model.criterionCycle(rec_B, real_B) * config['training']['lambda_cycle']
                
                # Combined loss
                loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
                
                # Update generators
                model.optimizer_G.zero_grad()
                loss_G.backward()
                model.optimizer_G.step()
                
                # Update discriminators
                # D_A
                pred_real = model.netD_A(real_B)
                loss_D_real = model.criterionGAN(pred_real, True)
                fake_B_ = fake_B_pool.query(fake_B)
                pred_fake = model.netD_A(fake_B_.detach())
                loss_D_fake = model.criterionGAN(pred_fake, False)
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                model.optimizer_D_A.zero_grad()
                loss_D_A.backward()
                model.optimizer_D_A.step()
                
                # D_B
                pred_real = model.netD_B(real_A)
                loss_D_real = model.criterionGAN(pred_real, True)
                fake_A_ = fake_A_pool.query(fake_A)
                pred_fake = model.netD_B(fake_A_.detach())
                loss_D_fake = model.criterionGAN(pred_fake, False)
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                model.optimizer_D_B.zero_grad()
                loss_D_B.backward()
                model.optimizer_D_B.step()
                
                # Update progress bar
                pbar.set_postfix({
                    'G_loss': f'{loss_G.item():.3f}',
                    'D_A_loss': f'{loss_D_A.item():.3f}',
                    'D_B_loss': f'{loss_D_B.item():.3f}'
                })
                
                # Log to tensorboard
                if config['logging']['tensorboard'] and total_iters % config['logging']['metrics_frequency'] == 0:
                    writer.add_scalar('loss_G', loss_G.item(), total_iters)
                    writer.add_scalar('loss_D_A', loss_D_A.item(), total_iters)
                    writer.add_scalar('loss_D_B', loss_D_B.item(), total_iters)
                
                if config['logging']['tensorboard'] and total_iters % config['logging']['images_frequency'] == 0:
                    writer.add_images('real_A', real_A, total_iters)
                    writer.add_images('fake_B', fake_B, total_iters)
                    writer.add_images('rec_A', rec_A, total_iters)
                    writer.add_images('real_B', real_B, total_iters)
                    writer.add_images('fake_A', fake_A, total_iters)
                    writer.add_images('rec_B', rec_B, total_iters)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['paths']['checkpoints_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            model.save_networks(checkpoint_path)
            
        # Validation
        model.eval()
        val_loss_G = 0
        val_loss_D = 0
        with torch.no_grad():
            for batch in val_loader:
                real_A = batch['input'].to(device)
                real_B = batch['target'].to(device)
                
                fake_B = model.netG_A(real_A)
                fake_A = model.netG_B(real_B)
                
                # Calculate validation losses
                loss_G = model.criterionGAN(model.netD_A(fake_B), True)
                loss_D = (model.criterionGAN(model.netD_A(real_B), True) + 
                         model.criterionGAN(model.netD_A(fake_B), False)) * 0.5
                
                val_loss_G += loss_G.item()
                val_loss_D += loss_D.item()
        
        val_loss_G /= len(val_loader)
        val_loss_D /= len(val_loader)
        
        if config['logging']['tensorboard']:
            writer.add_scalar('val_loss_G', val_loss_G, epoch)
            writer.add_scalar('val_loss_D', val_loss_D, epoch)
    
    if config['logging']['tensorboard']:
        writer.close()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(config) 