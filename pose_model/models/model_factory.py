from __future__ import annotations

import torch

from pose_model.datasets.multimodal_sequence_dataset import HAND_NUM_CLASSES, HEAD_NUM_CLASSES
from .geometry_first_model import GeometryFirstPoseModel
from .multi_task_model import MultiTaskPoseModel


def get_model_architecture(cfg: dict) -> str:
    return str(cfg.get("model", {}).get("architecture", "multi_task_visual")).lower()


def model_requires_images(cfg: dict) -> bool:
    arch = get_model_architecture(cfg)
    if arch == "geometry_first":
        return bool(cfg.get("model", {}).get("appearance_expert", {}).get("enabled", False))
    return True


def build_pose_model(cfg: dict, device: torch.device | None = None):
    model_cfg = cfg["model"]
    arch = get_model_architecture(cfg)

    if arch in {"multi_task_visual", "visual", "hybrid_multitask"}:
        model = MultiTaskPoseModel(
            backbone=model_cfg["backbone"],
            feature_dim=model_cfg["feature_dim"],
            temporal_encoder=model_cfg.get("temporal_encoder", "transformer"),
            head_temporal_encoder=model_cfg.get("head_temporal_encoder", None),
            hand_temporal_encoder=model_cfg.get("hand_temporal_encoder", None),
            tcn_channels=model_cfg.get("tcn_channels", None),
            tcn_kernel=model_cfg.get("tcn_kernel", 3),
            tcn_dilations=model_cfg.get("tcn_dilations", None),
            tcn_dropout=model_cfg.get("tcn_dropout", 0.2),
            head_tcn_channels=model_cfg.get("head_tcn_channels", None),
            hand_tcn_channels=model_cfg.get("hand_tcn_channels", None),
            head_tcn_kernel=model_cfg.get("head_tcn_kernel", None),
            hand_tcn_kernel=model_cfg.get("hand_tcn_kernel", None),
            head_tcn_dilations=model_cfg.get("head_tcn_dilations", None),
            hand_tcn_dilations=model_cfg.get("hand_tcn_dilations", None),
            head_tcn_dropout=model_cfg.get("head_tcn_dropout", None),
            hand_tcn_dropout=model_cfg.get("hand_tcn_dropout", None),
            transformer_cfg=model_cfg.get("transformer", None),
            head_transformer_cfg=model_cfg.get("head_transformer", None),
            hand_transformer_cfg=model_cfg.get("hand_transformer", None),
            shared_backbone=model_cfg.get("shared_backbone", False),
            shared_temporal=model_cfg.get("shared_temporal", False),
            adapter_enabled=model_cfg.get("adapter", {}).get("enabled", False),
            adapter_dim=model_cfg.get("adapter", {}).get("dim", None),
            adapter_dropout=model_cfg.get("adapter", {}).get("dropout", 0.1),
            keypoint_fusion_enabled=model_cfg.get("keypoint_fusion", {}).get("enabled", False),
            keypoint_hidden_dim=model_cfg.get("keypoint_fusion", {}).get("hidden_dim", None),
            keypoint_dropout=model_cfg.get("keypoint_fusion", {}).get("dropout", 0.1),
            head_use_attn_pool=model_cfg.get("head_attention_pool", False),
            head_attn_pool_dropout=model_cfg.get("head_attention_dropout", 0.1),
            hand_use_attn_pool=model_cfg.get("hand_attention_pool", False),
            hand_attn_pool_dropout=model_cfg.get("hand_attention_dropout", 0.1),
            num_head_classes=model_cfg.get("num_head_classes", HEAD_NUM_CLASSES),
            num_hand_classes=model_cfg.get("num_hand_classes", HAND_NUM_CLASSES),
            freeze_backbone=model_cfg["freeze_backbone"],
            freeze_stages=model_cfg.get("freeze_stages", -1),
            pretrained=model_cfg.get("pretrained", True),
            resnet_variant=model_cfg.get("resnet_variant", "resnet50"),
            cnn_branch_channels=model_cfg.get("cnn_branch_channels", None),
            fusion=model_cfg.get("fusion", "concat"),
            fusion_dropout=model_cfg.get("fusion_dropout", 0.0),
        )
    elif arch == "geometry_first":
        temporal_cfg = model_cfg.get("geometry_temporal", {})
        expert_cfg = model_cfg.get("appearance_expert", {})
        model = GeometryFirstPoseModel(
            geom_dim=model_cfg.get("geom_dim", 128),
            hidden_dim=temporal_cfg.get("hidden_dim", 128),
            num_layers=temporal_cfg.get("num_layers", 2),
            dropout=temporal_cfg.get("dropout", 0.1),
            bidirectional=temporal_cfg.get("bidirectional", False),
            num_head_classes=model_cfg.get("num_head_classes", HEAD_NUM_CLASSES),
            num_hand_classes=model_cfg.get("num_hand_classes", HAND_NUM_CLASSES),
            use_head_attn_pool=model_cfg.get("head_attention_pool", True),
            use_hand_attn_pool=model_cfg.get("hand_attention_pool", True),
            attn_pool_dropout=model_cfg.get("head_attention_dropout", 0.1),
            appearance_expert_enabled=expert_cfg.get("enabled", False),
            appearance_backbone=expert_cfg.get("backbone", "simple_cnn"),
            appearance_feature_dim=expert_cfg.get("feature_dim", 96),
            appearance_shared_backbone=expert_cfg.get("shared_backbone", True),
            appearance_dropout=expert_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")

    if device is not None:
        model = model.to(device)
    return model, arch


def collect_backbone_modules(model) -> list:
    if hasattr(model, "get_backbone_modules"):
        return list(model.get_backbone_modules())
    modules = []
    for name in ("backbone_head", "backbone_hand"):
        module = getattr(model, name, None)
        if module is not None and module not in modules:
            modules.append(module)
    return modules
