# HeadPoseLab 训练模块

基于 README 规划的 CNN + LSTM 训练脚手架，负责读取已标注的头部序列数据并训练 5 类姿态识别模型（1=正，2=下，3=左，4=右，5=歪）。

## 目录结构
```
pose_model/
  configs/         # 配置文件
  data/            # 数据根目录（可挂载实际数据）
  datasets/        # 数据加载逻辑
  models/          # CNN + LSTM 结构
  utils/           # 日志、度量、随机种子
  train.py         # 训练入口
  eval.py          # 测试评估入口
  inference_demo.py# 简单推理示例
  requirements.txt
```

## 数据格式
```
data/
  head_pose/
    images/
      head_00001.jpg
      ...
    labels.json   # 头部姿态标签 + 头部点归一化坐标
  hand_pose/
    images/
      hand_00001.jpg   # 左右手截图横向拼接
      ...
    labels.json   # 手部姿态标签 + 左/右手点归一化坐标
```

## 准备环境
```bash
cd pose_model
py -3.10 -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

## 运行训练
```bash
cd pose_model
.\.venv\Scripts\Activate
python train.py --config configs/default.yaml
```
模型与日志目录由 `configs/default.yaml` 中的 `train.save_dir` 控制。

## 评估
```bash
cd pose_model
.\.venv\Scripts\Activate
python eval.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## 推理示例
```bash
python inference_demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt --images_dir path/to/images
```
`images_dir` 需包含按时间排序的一段序列（数量不少于配置的 `sequence_length`）。
