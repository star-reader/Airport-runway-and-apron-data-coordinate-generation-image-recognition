import os
import numpy as np
import cv2
import json
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import albumentations as A
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class AirportDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 将图像转换为PyTorch张量
        image = image.transpose(2, 0, 1).astype('float32') / 255.0
        mask = np.expand_dims(mask, 0).astype('float32') / 255.0
        
        return torch.from_numpy(image), torch.from_numpy(mask)

def prepare_dataset(dataset_dir, output_dir, img_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    image_files = [f for f in os.listdir(os.path.join(dataset_dir, 'images')) if f.endswith(('.jpg', '.png', '.tif'))]
    coord_files = [f for f in os.listdir(os.path.join(dataset_dir, 'coordinates')) if f.endswith('.json')]
    
    image_paths = []
    mask_paths = []
    
    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        coord_file = f"{img_name}.json"
        
        if coord_file in coord_files:
            img_path = os.path.join(dataset_dir, 'images', img_file)
            img = cv2.imread(img_path)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            with open(os.path.join(dataset_dir, 'coordinates', coord_file), 'r') as f:
                coord_data = json.load(f)
            # 在掩码上绘制跑道、机坪、滑行道
            for feature in coord_data['features']:
                feature_type = feature['properties']['type']
                coords = feature['geometry']['coordinates']
                
                if feature['geometry']['type'] == 'Polygon':
                    coords = np.array(coords[0], dtype=np.int32)
                    if feature_type == 'runway':
                        cv2.fillPoly(mask, [coords], 1)
                    elif feature_type == 'apron':
                        cv2.fillPoly(mask, [coords], 2)
                    elif feature_type == 'taxiway':
                        cv2.fillPoly(mask, [coords], 3)
                elif feature['geometry']['type'] == 'LineString':
                    coords = np.array(coords, dtype=np.int32)
                    if feature_type == 'taxiway':
                        cv2.polylines(mask, [coords], False, 3, 5)
            img_resized = cv2.resize(img, img_size)
            mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            
            # 保存处理后的图像和掩码
            processed_img_path = os.path.join(output_dir, 'images', img_file)
            processed_mask_path = os.path.join(output_dir, 'masks', f"{img_name}_mask.png")
            
            cv2.imwrite(processed_img_path, img_resized)
            cv2.imwrite(processed_mask_path, mask_resized)
            
            image_paths.append(processed_img_path)
            mask_paths.append(processed_mask_path)
    
    return image_paths, mask_paths

def create_dataloaders(image_paths, mask_paths, batch_size=8, val_split=0.2):
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42
    )
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
    ])
    
    val_transform = A.Compose([])
    train_dataset = AirportDataset(train_img_paths, train_mask_paths, transform=train_transform)
    val_dataset = AirportDataset(val_img_paths, val_mask_paths, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
