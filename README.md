# 机场设施识别与GeoJSON生成系统

本项目实现了一个深度学习系统，可以从卫星图像中识别机场跑道、机坪和滑行道，并生成对应的GeoJSON数据。

## 功能特点

- 使用U-Net深度学习模型进行语义分割
- 支持识别跑道、机坪和滑行道
- 生成标准GeoJSON格式数据
- 可视化预测结果
- 支持处理地理参考图像(GeoTIFF)

## 项目结构

```
├── data
│   ├── raw           # 原始数据
│   ├── processed     # 预处理后的数据
│   └── test          # 测试图像
├── models            # 保存训练好的模型
├── results           # 推理结果和可视化
└── src               # 源代码
    ├── data_preprocessing.py  # 数据预处理
    ├── model.py               # 模型定义
    ├── train.py               # 训练代码
    ├── geojson_generator.py   # GeoJSON生成
    └── inference.py           # 推理脚本
```

## 数据要求

- 原始数据应包含机场卫星图像和对应的坐标标注
- 坐标标注应为GeoJSON格式，包含跑道、机坪和滑行道的几何信息

## 安装和配置

1. 克隆仓库:

```bash
git clone https://github.com/username/airport-runway-recognition.git
cd airport-runway-recognition
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
cd src
python train.py
```

### 处理单张图像并生成GeoJSON

```bash
python inference.py --model ../models/best_model.pth --image ../data/test/example_airport.tif --output_dir ../results
```

### 批量处理图像

```bash
python batch_process.py --model ../models/best_model.pth --input_dir ../data/test --output_dir ../results
```

## 输出示例

生成的GeoJSON文件包含跑道、机坪和滑行道的几何信息:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "type": "runway"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "type": "taxiway"
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [...]
      }
    }
  ]
}
```
