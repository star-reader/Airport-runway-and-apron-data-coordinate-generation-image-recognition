import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping, LineString, Polygon
from PIL import Image
import geopandas as gpd
from model import UNet
from sklearn.cluster import DBSCAN

class GeoJSONGenerator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = UNet(n_channels=3, n_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.class_map = {
            1: 'runway',
            2: 'apron',
            3: 'taxiway'
        }
    
    def preprocess_image(self, image_path, target_size=(512, 512)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        image = cv2.resize(image, target_size)
        image = image.transpose(2, 0, 1).astype('float32') / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        return image, original_size
    
    def predict(self, image_path):
        image, original_size = self.preprocess_image(image_path)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            output = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        pred = cv2.resize(pred.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        return pred
    
    def mask_to_geojson(self, mask, image_path, geo_transform=None):
        features = []
        
        if geo_transform is None:
            height, width = mask.shape
            geo_transform = [0, 1, 0, 0, 0, 1]
        
        for class_id, class_name in self.class_map.items():
            binary_mask = (mask == class_id).astype(np.uint8)
        
            if np.sum(binary_mask) > 0:
                # 连接组件分析
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # 过滤小区域
                        if class_name == 'taxiway':
                            # 对于滑行道，使用骨架提取算法获取中心线
                            skeleton = self._extract_skeleton(binary_mask)
                            lines = self._skeleton_to_lines(skeleton)
                            
                            for line in lines:
                                # 将像素坐标转换为地理坐标
                                geo_coords = self._pixel_to_geo(line, geo_transform)
                                
                                feature = {
                                    "type": "Feature",
                                    "properties": {
                                        "type": class_name
                                    },
                                    "geometry": {
                                        "type": "LineString",
                                        "coordinates": geo_coords
                                    }
                                }
                                features.append(feature)
                        else:
                            epsilon = 0.002 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            geo_coords = self._pixel_to_geo(approx.squeeze(), geo_transform)
                            if not np.array_equal(geo_coords[0], geo_coords[-1]):
                                geo_coords.append(geo_coords[0])
                            
                            feature = {
                                "type": "Feature",
                                "properties": {
                                    "type": class_name
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [geo_coords]
                                }
                            }
                            features.append(feature)
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return geojson
    
    def _extract_skeleton(self, binary_mask):
        skeleton = np.zeros_like(binary_mask)
        img = binary_mask.copy()
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        from skimage.morphology import skeletonize
        skeleton = skeletonize(img > 0).astype(np.uint8) * 255
        return skeleton
    
    def _skeleton_to_lines(self, skeleton):
        points = np.column_stack(np.where(skeleton > 0))
        if len(points) == 0:
            return []
        
        clustering = DBSCAN(eps=10, min_samples=3).fit(points)
        lines = []
        for label in set(clustering.labels_):
            if label == -1:  # 噪声点
                continue
            cluster_points = points[clustering.labels_ == label]
            
            if len(cluster_points) > 1:
                ordered_points = [cluster_points[0]]
                remaining_points = cluster_points[1:].tolist()
                
                while remaining_points:
                    last_point = ordered_points[-1]
                    distances = [np.sum((last_point - np.array(p))**2) for p in remaining_points]
                    idx = np.argmin(distances)
                    ordered_points.append(remaining_points.pop(idx))
                
                line = [[p[1], p[0]] for p in ordered_points]  # 从(row,col)转换为(x,y)
                lines.append(line)
        
        return lines
    
    def _pixel_to_geo(self, pixels, geo_transform):
        if len(pixels) == 0:
            return []
        
        if isinstance(pixels, np.ndarray) and pixels.ndim == 2:
            geo_coords = []
            for pixel in pixels:
                if len(pixel) >= 2:  # 确保至少有x和y坐标
                    x = geo_transform[0] + pixel[0] * geo_transform[1] + pixel[1] * geo_transform[2]
                    y = geo_transform[3] + pixel[0] * geo_transform[4] + pixel[1] * geo_transform[5]
                    geo_coords.append([x, y])
            return geo_coords
        else:
            # 单个点
            x = geo_transform[0] + pixels[0] * geo_transform[1] + pixels[1] * geo_transform[2]
            y = geo_transform[3] + pixels[0] * geo_transform[4] + pixels[1] * geo_transform[5]
            return [x, y]
    
    def process_image(self, image_path, output_path=None):
        mask = self.predict(image_path)
        geo_transform = None
        try:
            with rasterio.open(image_path) as src:
                geo_transform = src.transform.to_gdal()
        except:
            print(f"无法从{image_path}获取地理信息，使用像素坐标")
        geojson = self.mask_to_geojson(mask, image_path, geo_transform)
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_features.geojson"
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=4)
        
        print(f"GeoJSON已保存到 {output_path}")
        return geojson

def main():
    model_path = '../models/best_model.pth'
    generator = GeoJSONGenerator(model_path)
    test_image_path = '../data/test/example_airport.tif'
    output_path = '../results/example_airport_features.geojson'
    generator.process_image(test_image_path, output_path)

if __name__ == "__main__":
    main()
