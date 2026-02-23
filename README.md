# HeadPoseLab 训练模块

项目当前支持两条可切换架构：
- `multi_task_visual`：视觉主导（CNN/ResNet + 时序编码器）
- `geometry_first`：几何主导（关键点时序特征 + 轻量 GRU，可选视觉专家）

目标类别为 5 类头姿态（1=正，2=下，3=左，4=右，5=歪）与 4 类手势（移除原始手势类 2 后映射为 1..4）。

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

高准确率实验配置（头姿态优先）：
```bash
python train.py --config configs/exp_head_priority.yaml
```

第一性重构实验（几何优先）：
```bash
python train.py --config configs/exp_geometry_first.yaml
```
训练会保存：
- `best_head_f1.pt`（主模型）
- `best_val_loss.pt`
- `last.pt`

## 评估
```bash
cd pose_model
.\.venv\Scripts\Activate
python eval.py --config configs/default.yaml --checkpoint checkpoints/exp_roi_tcn_attn/best.pt
python eval.py --config configs/exp_head_priority.yaml --checkpoint checkpoints/exp_head_priority/best_head_f1.pt
python eval.py --config configs/exp_geometry_first.yaml --checkpoint checkpoints/exp_geometry_first/best_head_f1.pt
```

## 推理示例
```bash
python inference_demo.py --config configs/default.yaml --checkpoint checkpoints/exp_roi_tcn_attn/best.pt --person_dir path/to/person_dir
python inference_demo.py --config configs/exp_geometry_first.yaml --checkpoint checkpoints/exp_geometry_first/best_head_f1.pt --person_dir path/to/person_dir
```
`person_dir` 需包含 `head_pose/images`、`hand_pose/images`。当 `person_dir` 同时包含对应 `labels.json` 时，`geometry_first` 可直接使用关键点特征并启用时序平滑推理。
