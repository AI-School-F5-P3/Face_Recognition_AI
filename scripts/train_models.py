import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import wandb  # For experiment tracking

from typing import Dict, Any, Optional

class FaceEmbeddingModel(nn.Module):
    """Simple CNN for face embedding generation."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

class TripletLoss(nn.Module):
    """Triplet loss for training face embeddings."""
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class Trainer:
    """Training manager for face detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = FaceEmbeddingModel(
            embedding_dim=config['model']['embedding_dim']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Initialize loss function
        self.criterion = TripletLoss(margin=config['training']['margin'])
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch_idx, (anchor, positive, negative) in enumerate(pbar):
                # Move data to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                anchor_embed = self.model(anchor)
                positive_embed = self.model(positive)
                negative_embed = self.model(negative)
                
                # Calculate loss
                loss = self.criterion(anchor_embed, positive_embed, negative_embed)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                wandb.log({'batch_loss': loss.item()})
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for anchor, positive, negative in tqdm(val_loader, desc='Validating'):
                # Move data to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                anchor_embed = self.model(anchor)
                positive_embed = self.model(positive)
                negative_embed = self.model(negative)
                
                # Calculate loss
                loss = self.criterion(anchor_embed, positive_embed, negative_embed)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None) -> None:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        # Initialize wandb
        wandb.init(
            project=self.config['wandb']['project'],
            config=self.config
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.logger.info(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
            self.logger.info(f'Training Loss: {train_loss:.4f}')
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.logger.info(f'Validation Loss: {val_loss:.4f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                
                # Log to wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, filename)
        self.logger.info(f'Saved checkpoint: {filename}')

def main():
    parser = argparse.ArgumentParser(description='Train face detection models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to training configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # TODO: Initialize data loaders
    # train_loader = ...
    # val_loader = ...
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()



"""The scripts folder typically contains utility scripts that help with model management, training, and setup. In this case:

download_weights.py - Downloads pre-trained model weights for face detection and feature extraction
train_models.py - Handles the training pipeline for custom face detection/feature extraction models"""

"""download_weights.py:


Downloads pre-trained model weights from specified URLs
Features:

Progress bar for downloads using tqdm
MD5 checksum verification
Skips existing files if MD5 matches
Configurable output directory
Error handling and logging
Supports multiple model types




train_models.py:


Implements the training pipeline for face detection/embedding models
Features:

Custom CNN architecture for face embeddings
Triplet loss implementation for face verification
Training manager with configuration support
Integration with Weights & Biases (wandb) for experiment tracking
Checkpoint saving and loading
Validation support
Progress bars and logging



To use these scripts:

For downloading weights:

bashCopypython download_weights.py --output-dir models/

For training:

bashCopypython train_models.py --config configs/training_config.yaml"""