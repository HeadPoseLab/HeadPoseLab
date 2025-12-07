# HeadPoseLab 训练模块

基于 CNN + LSTM 的姿态分类训练脚手架，读取已标注的多人人头部序列，训练 7 类姿态模型并导出权重。

## 目录
```
pose_model/
  configs/         # 配置
  data/            # 数据根目录（放 person_xxx）
  datasets/        # 数据加载
  models/          # CNN/LSTM
  utils/           # 日志/指标/种子/损失
  train.py         # 训练
  eval.py          # 测试评估
  inference_demo.py# 推理示例
  requirements.txt
```

## 数据格式
```
pose_model/data/
  person_001/
    images/
      head_00001.jpg
      ...
    labels.txt      # 每行: 文件名;类别编号(1-7)
  person_002/
    ...
```

## 安装与运行
- macOS (zsh):
  ```bash
  cd pose_model
  python3 -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python3 train.py --config configs/default.yaml
  # 评估
  python3 eval.py --config configs/default.yaml --checkpoint checkpoints/best.pt
  ```
- Windows (PowerShell):
  ```powershell
  cd pose_model
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python train.py --config configs/default.yaml
  # 评估
  python eval.py --config configs/default.yaml --checkpoint checkpoints/best.pt
  ```

## 推理示例
```bash
python inference_demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt --images_dir path/to/images
```
`images_dir` 需包含按时间排序的一段序列（数量不少于 `sequence_length`）。

## 类别不平衡处理（配置可调）
- `filter.enabled: true`：生成序列后按主标签下采样多数类，最多保留为最少类的 `max_factor` 倍。
- `sampler.balanced: true`：训练时使用 `WeightedRandomSampler` 过采样少数类。
- `loss.class_weights: auto`：按类别频次逆比自动加权；可将 `loss.type` 设为 `focal` 启用 Focal Loss。
需要调整平衡策略时，修改 `configs/default.yaml` 即可。
