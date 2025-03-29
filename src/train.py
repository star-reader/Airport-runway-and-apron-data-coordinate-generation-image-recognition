import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_preprocessing import prepare_dataset, create_dataloaders
from model import UNet, CombinedLoss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型，验证损失: {best_val_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    return model

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据集
    dataset_dir = '../data/raw'
    processed_dir = '../data/processed'
    
    image_paths, mask_paths = prepare_dataset(dataset_dir, processed_dir)
    train_loader, val_loader = create_dataloaders(image_paths, mask_paths, batch_size=8)
    
    # 初始化模型、损失函数和优化器
    model = UNet(n_channels=3, n_classes=4)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        device=device
    )
    
    print("训练完成!")

if __name__ == "__main__":
    main()
