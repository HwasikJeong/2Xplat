import os
import torch
import traceback
import torch.nn.functional as F
from typing import Any
from torch import nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange
from safetensors.torch import load_file
from dataclasses import dataclass

from src.model.sh_eval import _spherical_harmonics
from src.model.transformer import TransformerBlock
from src.model.dpt_head import DPTHead
from src.model.prope_custom import PropeDotProductAttention
from src.model.depth_anything.da_model.da3 import DepthAnything3
from src.model.gaussians import GaussianRenderer, GaussianField
from src.utils.camera_utils import (
    invert_SE3, compute_rays, compute_plucmap, fxfycxcy_to_K, mat_to_quat, quat_to_mat
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    """Initialise Linear and Embedding weights with N(0, 0.02) and reset norm layers."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------

@dataclass
class CameraBundle:
    pred_i_fxfycxcy: torch.Tensor  # (B, V, 4)
    pred_i_c2w:      torch.Tensor  # (B, V, 4, 4)
    pred_t_fxfycxcy: torch.Tensor  # (B, T, 4)
    pred_t_c2w:      torch.Tensor  # (B, T, 4, 4)
    gt_i_fxfycxcy:   torch.Tensor  # (B, V, 4)
    gt_i_c2w:        torch.Tensor  # (B, V, 4, 4)
    gt_t_fxfycxcy:   torch.Tensor  # (B, T, 4)
    gt_t_c2w:        torch.Tensor  # (B, T, 4, 4)


# ---------------------------------------------------------------------------
# Geometry Expert (DepthAnything3)
# ---------------------------------------------------------------------------

class GeometryExpert:
    """
    Predicts camera poses and intrinsics for all views using DA3.
    """

    def __init__(
        self,
        pose_regressor: DepthAnything3,
        scene_scale: float,
        inference_mode: bool,
        camera_mode: str | None,
    ):
        """
        Args:
            pose_regressor: DA3 model used to predict poses and intrinsics.
            scene_scale: Denominator for translation normalisation (from data config).
            inference_mode: If True, ``camera_mode`` controls which cameras are used.
            camera_mode: One of ``gt_pose_gt_intr``, ``pred_pose_gt_intr``,
                ``pred_pose_pred_intr``; ignored during training.
        """
        self.pose_regressor = pose_regressor
        self.scene_scale = scene_scale
        self.inference_mode = inference_mode
        self.camera_mode = camera_mode

    def predict_cameras(
        self,
        input_data_dict: dict,
        target_data_dict: dict,
    ) -> CameraBundle:
        """Run DA3 on all views, normalize poses to scene scale, and apply camera mode."""
        gt_i_fxfycxcy = input_data_dict["fxfycxcy"].float()
        gt_i_c2w = input_data_dict["c2w"].float()
        gt_t_fxfycxcy = target_data_dict["fxfycxcy"].float()
        gt_t_c2w = target_data_dict["c2w"].float()

        pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w = self._run_da3_and_normalize(
            input_data_dict["image"], target_data_dict["image"],
        )

        if self.inference_mode:
            pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w = self._select_cameras(
                pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w,
                gt_i_fxfycxcy, gt_i_c2w, gt_t_fxfycxcy, gt_t_c2w,
            )

        return CameraBundle(
            pred_i_fxfycxcy=pred_i_fxfycxcy,
            pred_i_c2w=pred_i_c2w,
            pred_t_fxfycxcy=pred_t_fxfycxcy,
            pred_t_c2w=pred_t_c2w,
            gt_i_fxfycxcy=gt_i_fxfycxcy,
            gt_i_c2w=gt_i_c2w,
            gt_t_fxfycxcy=gt_t_fxfycxcy,
            gt_t_c2w=gt_t_c2w,
        )

    def _run_da3_and_normalize(
        self,
        input_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward DA3 on all input+target images and normalize cameras to scene scale."""
        _, v, _, h, w = input_images.shape
        with torch.autocast(device_type="cuda", enabled=False):
            all_images = torch.cat([input_images, target_images], dim=1)
            output = self.pose_regressor(all_images)

            i_fxfycxcy_raw = output["fxfycxcy"][:, :v].float()
            i_c2w_raw = output["extrinsics"][:, :v].float()
            t_fxfycxcy_raw = output["fxfycxcy"][:, v:].float()
            t_c2w_raw = output["extrinsics"][:, v:].float()

            # DA3 outputs intrinsics at its own resolution; rescale to the model's input resolution.
            da_h, da_w = input_images.shape[-2:]
            intr_scale = torch.tensor(
                [w / float(da_w), h / float(da_h), w / float(da_w), h / float(da_h)],
                dtype=i_fxfycxcy_raw.dtype,
                device=i_fxfycxcy_raw.device,
            ).view(1, 1, 4)

            i_fxfycxcy = i_fxfycxcy_raw * intr_scale
            t_fxfycxcy = t_fxfycxcy_raw * intr_scale
            i_c2w, t_c2w = self._shared_scene_normalization(
                input_c2ws=i_c2w_raw,
                target_c2ws=t_c2w_raw,
                scene_scale=self.scene_scale,
            )
        return i_fxfycxcy, i_c2w, t_fxfycxcy, t_c2w

    @staticmethod
    def _shared_scene_normalization(
        input_c2ws: torch.Tensor,                 # (V, 4, 4) or (B, V, 4, 4)
        target_c2ws: torch.Tensor | None = None,  # (T, 4, 4) or (B, T, 4, 4)
        scene_scale: float = 1.0,
    ):
        """
        Match scene normalization used in dataset.py:
        1) Build canonical frame from input poses only (Gram-Schmidt).
        2) Apply same transform to both input/target poses.
        3) Scale translations by max abs translation from normalized input poses.
        """
        squeeze_input = input_c2ws.ndim == 3
        squeeze_target = target_c2ws is not None and target_c2ws.ndim == 3

        if squeeze_input:
            input_c2ws = input_c2ws.unsqueeze(0)
        if target_c2ws is not None and squeeze_target:
            target_c2ws = target_c2ws.unsqueeze(0)

        position_avg = input_c2ws[:, :, :3, 3].mean(dim=1)   # (B, 3)
        forward_avg = input_c2ws[:, :, :3, 2].mean(dim=1)    # (B, 3)
        down_avg = input_c2ws[:, :, :3, 1].mean(dim=1)       # (B, 3)

        forward_avg = F.normalize(forward_avg, dim=-1)
        down_proj = (down_avg * forward_avg).sum(dim=-1, keepdim=True) * forward_avg
        down_avg = F.normalize(down_avg - down_proj, dim=-1)
        right_avg = torch.cross(down_avg, forward_avg, dim=-1)

        pos_avg = torch.eye(
            4, dtype=input_c2ws.dtype, device=input_c2ws.device
        ).expand(input_c2ws.shape[0], 4, 4).clone()
        pos_avg[:, :3, 0] = right_avg
        pos_avg[:, :3, 1] = down_avg
        pos_avg[:, :3, 2] = forward_avg
        pos_avg[:, :3, 3] = position_avg
        pos_avg_inv = torch.linalg.inv(pos_avg)

        input_c2ws = torch.matmul(pos_avg_inv.unsqueeze(1), input_c2ws)
        if target_c2ws is not None:
            target_c2ws = torch.matmul(pos_avg_inv.unsqueeze(1), target_c2ws)

        translations = input_c2ws[:, :, :3, 3].clone().detach()
        scene_extent = translations.abs().amax(dim=(1, 2))

        scale = 1.0 / (scene_scale * scene_extent)
        # Avoid in-place writes on sliced views (e.g. [..., :3, 3]) to keep
        # autograd version tracking consistent.
        input_scaled_t = input_c2ws[:, :, :3, 3] * scale[:, None, None]
        input_c2ws = torch.cat(
            [
                torch.cat([input_c2ws[:, :, :3, :3], input_scaled_t.unsqueeze(-1)], dim=-1),
                input_c2ws[:, :, 3:, :],
            ],
            dim=-2,
        )
        if target_c2ws is not None:
            target_scaled_t = target_c2ws[:, :, :3, 3] * scale[:, None, None]
            target_c2ws = torch.cat(
                [
                    torch.cat([target_c2ws[:, :, :3, :3], target_scaled_t.unsqueeze(-1)], dim=-1),
                    target_c2ws[:, :, 3:, :],
                ],
                dim=-2,
            )

        if squeeze_input:
            input_c2ws = input_c2ws.squeeze(0)
        if target_c2ws is not None and squeeze_target:
            target_c2ws = target_c2ws.squeeze(0)

        if target_c2ws is None:
            return input_c2ws
        return input_c2ws, target_c2ws

    def _select_cameras(
        self,
        pred_i_fxfycxcy: torch.Tensor,
        pred_i_c2w: torch.Tensor,
        pred_t_fxfycxcy: torch.Tensor,
        pred_t_c2w: torch.Tensor,
        gt_i_fxfycxcy: torch.Tensor,
        gt_i_c2w: torch.Tensor,
        gt_t_fxfycxcy: torch.Tensor,
        gt_t_c2w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose between predicted and GT cameras according to camera_mode."""
        if self.camera_mode == "gt_pose_gt_intr":
            return gt_i_fxfycxcy, gt_i_c2w, gt_t_fxfycxcy, gt_t_c2w
        if self.camera_mode == "pred_pose_gt_intr":
            return gt_i_fxfycxcy, pred_i_c2w, gt_t_fxfycxcy, pred_t_c2w
        if self.camera_mode == "pred_pose_pred_intr":
            return pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w
        raise ValueError(
            f"Unsupported camera_mode: {self.camera_mode!r}. "
            "Use one of: gt_pose_gt_intr, pred_pose_gt_intr, pred_pose_pred_intr."
        )


# ---------------------------------------------------------------------------
# Appearance Expert (MVP)
# ---------------------------------------------------------------------------

class AppearanceExpert:
    """
    Three-stage transformer (MVP) that maps ray-encoded images to a 3D Gaussian field.
    """

    def __init__(
        self,
        # nn.Module references (owned by TwoExpertModel)
        image_tokenizer: nn.Sequential,
        stage1: nn.ModuleList,
        stage2: nn.ModuleList,
        stage3: nn.ModuleList,
        merge_block1: nn.Conv2d,
        resize_block1: nn.Linear,
        merge_block2: nn.Conv2d,
        resize_block2: nn.Linear,
        attention2: PropeDotProductAttention,
        attention3: PropeDotProductAttention,
        dpt_head: DPTHead,
        gaussian_decoder: nn.Sequential,
        register_token_init: nn.Parameter,
        # scalar hyper-parameters
        patch_size: int,
        num_register_tokens: int,
        group_size: int,
        color_dim: int,
        opacity_dim: int,
        sh_degree: int,
        opacity_degree: int,
        scale_bias: float,
        scale_max: float,
        opacity_bias: float,
        max_dist: float,
        inference_mode: bool,
    ):
        """Store references to all nn.Module components and scalar hyper-parameters.

        All nn.Module arguments are owned and registered by ``TwoExpertModel``;
        this class holds non-owning references and orchestrates the forward pass.
        """
        self.image_tokenizer = image_tokenizer
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.merge_block1 = merge_block1
        self.resize_block1 = resize_block1
        self.merge_block2 = merge_block2
        self.resize_block2 = resize_block2
        self.attention2 = attention2
        self.attention3 = attention3
        self.dpt_head = dpt_head
        self.gaussian_decoder = gaussian_decoder
        self.register_token_init = register_token_init
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.group_size = group_size
        self.color_dim = color_dim
        self.opacity_dim = opacity_dim
        self.sh_degree = sh_degree
        self.opacity_degree = opacity_degree
        self.scale_bias = scale_bias
        self.scale_max = scale_max
        self.opacity_bias = opacity_bias
        self.max_dist = max_dist
        self.inference_mode = inference_mode

    def predict_gaussians(
        self,
        raymap_images: torch.Tensor,   # (B, V, C, H, W)
        i_w2c: torch.Tensor,           # (B, V, 4, 4)
        Ks: torch.Tensor,              # (B, V, 3, 3)
        i_fxfycxcy: torch.Tensor,      # (B, V, 4)
        i_c2w: torch.Tensor,           # (B, V, 4, 4)
        t_c2w: torch.Tensor,           # (B, T, 4, 4)
    ) -> GaussianField:
        """Run the full three-stage transformer pipeline and decode Gaussian parameters."""
        _, _, _, h, w = raymap_images.shape
        x, s1_patch_tokens = self._stage1_tokenize_and_encode(raymap_images)
        x, s2_patch_tokens = self._stage2_cross_view_and_downsample(x, i_w2c, Ks, h, w)
        s3_patch_tokens = self._stage3_global_cross_view(x, i_w2c, Ks)
        return self._decode_gaussians(
            s1_patch_tokens, s2_patch_tokens, s3_patch_tokens,
            i_fxfycxcy, i_c2w, t_c2w, h, w,
        )

    def _stage1_tokenize_and_encode(
        self,
        raymap_images: torch.Tensor,
    ):
        """Patch-embed ray-map images with register tokens, then run stage-1 transformer."""
        b, v, _, h, w = raymap_images.shape
        register_tokens = self.register_token_init.repeat(b, v, 1, 1)
        x = self.image_tokenizer(raymap_images)
        x = rearrange(x, "b (v l) d -> b v l d", v=v)
        x = torch.cat([register_tokens, x], dim=2)
        x = rearrange(x, "b v l d -> (b v) l d")
        x = self._run_stage1_blocks(x, None)

        r_tokens, s1_patch_tokens = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
        r_tokens = self.resize_block1(r_tokens)

        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        i_tokens = rearrange(s1_patch_tokens, "b (hh ww) d -> b d hh ww", hh=h_patches, ww=w_patches)
        i_tokens = self.merge_block1(i_tokens)
        i_tokens = rearrange(i_tokens, "b d hh ww -> b (hh ww) d", hh=h_patches // 2, ww=w_patches // 2)

        x = torch.cat([r_tokens, i_tokens], dim=1)
        x = rearrange(x, "(b g v) l d -> (b g) (v l) d", g=v // self.group_size, v=self.group_size)
        return x, s1_patch_tokens

    def _stage2_cross_view_and_downsample(
        self,
        x: torch.Tensor,
        i_w2c: torch.Tensor,
        Ks: torch.Tensor,
        h: int,
        w: int,
    ):
        """Run grouped cross-view attention (stage 2) and spatially downsample patch tokens."""
        v = i_w2c.shape[1]
        info = {
            "num_input_views": v,
            "w2c": rearrange(i_w2c, "b (g v) ... -> (b g) v ...", g=v // self.group_size, v=self.group_size),
            "Ks": rearrange(Ks, "b (g v) ... -> (b g) v ...", g=v // self.group_size, v=self.group_size),
            "attn2": self.attention2,
        }
        x = self._run_stage2_blocks(x, info)

        r_tokens, s2_patch_tokens = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
        r_tokens = self.resize_block2(r_tokens)

        h_patches = (h // self.patch_size) // 2
        w_patches = (w // self.patch_size) // 2
        i_tokens = rearrange(s2_patch_tokens, "b (hh ww) d -> b d hh ww", hh=h_patches, ww=w_patches)
        i_tokens = self.merge_block2(i_tokens)
        i_tokens = rearrange(i_tokens, "b d hh ww -> b (hh ww) d", hh=h_patches // 2, ww=w_patches // 2)

        x = torch.cat([r_tokens, i_tokens], dim=1)
        x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
        return x, s2_patch_tokens

    def _stage3_global_cross_view(
        self,
        x: torch.Tensor,
        i_w2c: torch.Tensor,
        Ks: torch.Tensor,
    ) -> torch.Tensor:
        """Run global cross-view attention (stage 3) across all input views."""
        v = i_w2c.shape[1]
        info = {
            "num_input_views": v,
            "attn3": self.attention3,
            "w2c": i_w2c,
            "Ks": Ks,
        }
        x = self._run_stage3_blocks(x, info)
        return x[:, self.num_register_tokens:]

    def _decode_gaussians(
        self,
        s1_patch_tokens: torch.Tensor,
        s2_patch_tokens: torch.Tensor,
        s3_patch_tokens: torch.Tensor,
        i_fxfycxcy: torch.Tensor,
        i_c2w: torch.Tensor,
        t_c2w: torch.Tensor,
        h: int,
        w: int,
    ) -> GaussianField:
        """Fuse multi-scale tokens with DPT head and decode into 3D Gaussian parameters."""
        b, v = i_c2w.shape[:2]
        t = t_c2w.shape[1] if t_c2w is not None else 0
        output_tokens = self.dpt_head(
            [s1_patch_tokens, s2_patch_tokens, s3_patch_tokens], [h, w], self.patch_size,
        )
        output_tokens = rearrange(output_tokens, "(b v) l d -> b (v l) d", v=v)
        gaussians = self.gaussian_decoder(output_tokens)
        gaussians = rearrange(
            gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d",
            v=v,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        xyz, feature, scale, rotation, opacity = torch.split(
            gaussians, [3, self.color_dim, 3, 4, self.opacity_dim], dim=-1,
        )
        xyz = xyz.float()
        feature = feature.float()
        scale = scale.float()
        rotation = rotation.float()
        opacity = opacity.float()

        with torch.autocast(device_type="cuda", enabled=False):
            rayo_gs, rayd_gs = compute_rays(i_fxfycxcy, i_c2w, h, w)
            scale = (scale + self.scale_bias).clamp(max=self.scale_max)
            # Bias only the DC (sh0) component; higher-order terms are bias-free.
            opacity[..., 0] = opacity[..., 0] + self.opacity_bias
            feature = rearrange(feature, "b n (c d) -> b n d c", c=3).contiguous()
            opacity = rearrange(opacity, "b n (c d) -> b n d c", c=1).contiguous()
            dist = xyz.mean(dim=-1, keepdim=True).sigmoid() * self.max_dist
            xyz = dist * rayd_gs + rayo_gs

            if not self.inference_mode:
                dirs = xyz[:, None, :, :] - t_c2w[..., :3, 3][..., None, :]  # (B, T, N, 3)
                opacity_broad = torch.broadcast_to(
                    opacity[..., None, :, :, :], (b, t, opacity.shape[1], -1, 1),
                )
                opacity_precompute = _spherical_harmonics(self.opacity_degree, dirs, opacity_broad)
            else:
                opacity_precompute = None

        return GaussianField(
            xyz=xyz,
            feature=feature,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            opacity_precompute=opacity_precompute,
        )

    def _run_stage1_blocks(self, x: torch.Tensor, info: dict | None) -> torch.Tensor:
        """Run all stage-1 transformer blocks with per-view self-attention."""
        for block in self.stage1:
            x = block(x, False, 1, info)
        return x

    def _run_stage2_blocks(self, x: torch.Tensor, info: dict) -> torch.Tensor:
        """Run stage-2 blocks, alternating between per-view and grouped cross-view attention."""
        g = self.group_size
        v = info["num_input_views"]
        for i, block in enumerate(self.stage2):
            if i % 2 == 0:
                x = rearrange(x, "(b g) (v l) d -> (b g v) l d", g=v // g, v=g)
                x = block(x, False, 2, info)
                x = rearrange(x, "(b g v) l d -> (b g) (v l) d", g=v // g, v=g)
            else:
                x = block(x, True, 2, info)
        return rearrange(x, "(b g) (v l) d -> (b g v) l d", g=v // g, v=g)

    def _run_stage3_blocks(self, x: torch.Tensor, info: dict) -> torch.Tensor:
        """Run stage-3 blocks, alternating between per-view and global cross-view attention."""
        v = info["num_input_views"]
        for i, block in enumerate(self.stage3):
            if i % 2 == 0:
                x = rearrange(x, "b (v l) d -> (b v) l d", v=v)
                x = block(x, False, 3, info)
                x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
            else:
                x = block(x, True, 3, info)
        return rearrange(x, "b (v l) d -> (b v) l d", v=v)


# ---------------------------------------------------------------------------
# TwoExpertModel — compositor
# ---------------------------------------------------------------------------

class TwoExpertModel(nn.Module):
    """Compositor that wires the GeometryExpert and AppearanceExpert together."""

    def __init__(self, config: Any) -> None:
        """Initialise all sub-modules and compose experts from the OmegaConf config.

        Args:
            config: OmegaConf config object. Presence of a ``config.inference`` key
                selects inference mode; absence means training mode.
        """
        super().__init__()

        # Extract all config values (no self.config stored)
        self.dim1 = config.model.dim1
        self.dim2 = config.model.dim2
        self.dim3 = config.model.dim3
        self.patch_size = config.model.patch_size
        self.num_register_tokens = config.model.num_register_tokens
        self.group_size = config.model.group_size
        self.head_dim = config.model.head_dim
        self.inter_multi = config.model.inter_multi
        self.qk_norm = config.model.qk_norm
        self.in_channels = config.model.in_channels
        self.stage1_nlayer = config.model.stage1_nlayer
        self.stage2_nlayer = config.model.stage2_nlayer
        self.stage3_nlayer = config.model.stage3_nlayer
        self.sh_degree = config.model.gaussians.sh_degree
        self.opacity_degree = config.model.gaussians.opacity_degree
        self.near_plane = config.model.gaussians.near_plane
        self.far_plane = config.model.gaussians.far_plane
        self.scale_bias = config.model.gaussians.scale_bias
        self.scale_max = config.model.gaussians.scale_max
        self.opacity_bias = config.model.gaussians.opacity_bias
        self.max_dist = config.model.gaussians.max_dist
        self.da_model_name = config.model.da_model_name
        self.da_weights_path = getattr(config.model, "da_model_weights_path", None)
        self.mvp_weights_path = getattr(config.model, "mvp_weights_path", None)
        self.scene_scale = config.data.scene_scale
        self.width = config.data.resize_w
        self.height = config.data.resize_h
        self.inference_mode = hasattr(config, "inference")
        self.camera_mode = (
            getattr(config.inference, "camera_mode", "pred_pose_pred_intr")
            if self.inference_mode else None
        )
        self.use_pose_optimization = bool(
            getattr(config.inference, "pose_optimization", False)
            if self.inference_mode else False
        )

        # Derived from SH degree config; computed once here, not on every forward.
        self.color_dim = 3 * (self.sh_degree + 1) ** 2
        self.opacity_dim = 1 * (self.opacity_degree + 1) ** 2

        # Build all nn.Module components (names must match checkpoint keys)
        self._build_geometry_modules()
        self._build_appearance_modules()

        # Compose experts
        self.geometry_expert = GeometryExpert(
            pose_regressor=self.pose_regressor,
            scene_scale=self.scene_scale,
            inference_mode=self.inference_mode,
            camera_mode=self.camera_mode,
        )
        self.appearance_expert = AppearanceExpert(
            image_tokenizer=self.image_tokenizer,
            stage1=self.stage1,
            stage2=self.stage2,
            stage3=self.stage3,
            merge_block1=self.merge_block1,
            resize_block1=self.resize_block1,
            merge_block2=self.merge_block2,
            resize_block2=self.resize_block2,
            attention2=self.attention2,
            attention3=self.attention3,
            dpt_head=self.dpt_head,
            gaussian_decoder=self.gaussian_decoder,
            register_token_init=self.register_token_init,
            patch_size=self.patch_size,
            num_register_tokens=self.num_register_tokens,
            group_size=self.group_size,
            color_dim=self.color_dim,
            opacity_dim=self.opacity_dim,
            sh_degree=self.sh_degree,
            opacity_degree=self.opacity_degree,
            scale_bias=self.scale_bias,
            scale_max=self.scale_max,
            opacity_bias=self.opacity_bias,
            max_dist=self.max_dist,
            inference_mode=self.inference_mode,
        )

        if not self.inference_mode:
            from src.model.loss import LossComputer
            self.loss_computer = LossComputer(config)

    # --- Module builders ---

    def _build_appearance_modules(self):
        """Build all nn.Module components for the appearance expert."""
        self.image_tokenizer = self._create_patch_tokenizer(
            self.in_channels, self.patch_size, self.dim1,
        )
        self.gaussian_decoder = nn.Sequential(
            nn.LayerNorm(self.dim3, bias=False),
            nn.Linear(
                self.dim3,
                (self.patch_size ** 2) * (3 + self.color_dim + 3 + 4 + self.opacity_dim),
                bias=False,
            ),
        )
        self.stage1 = self._build_transformer_stage(self.dim1, self.stage1_nlayer)
        self.stage2 = self._build_transformer_stage(self.dim2, self.stage2_nlayer)
        self.stage3 = self._build_transformer_stage(self.dim3, self.stage3_nlayer)

        self.register_token_init = nn.Parameter(
            torch.randn(1, 1, self.num_register_tokens, self.dim1),
        )
        nn.init.normal_(self.register_token_init, mean=0.0, std=0.02)

        self.attention2 = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=self.width // (self.patch_size * 2),
            patches_y=self.height // (self.patch_size * 2),
            image_width=self.width,
            image_height=self.height,
            num_register_tokens=self.num_register_tokens,
        )
        self.attention3 = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=self.width // (self.patch_size * 4),
            patches_y=self.height // (self.patch_size * 4),
            image_width=self.width,
            image_height=self.height,
            num_register_tokens=self.num_register_tokens,
        )
        self.merge_block1 = nn.Conv2d(
            self.dim1, self.dim2, kernel_size=2, stride=2,
            padding=0, bias=True, groups=self.dim1,
        )
        self.resize_block1 = nn.Linear(self.dim1, self.dim2)
        self.merge_block2 = nn.Conv2d(
            self.dim2, self.dim3, kernel_size=2, stride=2,
            padding=0, bias=True, groups=self.dim2,
        )
        self.resize_block2 = nn.Linear(self.dim2, self.dim3)
        self.dpt_head = DPTHead(
            dim_in=[self.dim1, self.dim2, self.dim3],
            features=self.dim3,
            out_channels=[self.dim1, self.dim2, self.dim3],
        )

        if self.mvp_weights_path != "":
            checkpoint = torch.load(self.mvp_weights_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint
            result = self.load_state_dict(state_dict, strict=False)
            # print(f"{result.missing_keys} missing keys")
            # print(f"{result.unexpected_keys} unexpected keys")
            print(f"Loaded MVP appearance weights from {self.mvp_weights_path}")

    def _build_geometry_modules(self):
        """Build the DA3 pose regressor and optionally load pretrained weights."""
        self.pose_regressor = DepthAnything3(model_name=self.da_model_name)
        if self.da_weights_path != "":
            state_dict = load_file(self.da_weights_path)
            results = self.pose_regressor.load_state_dict(state_dict, strict=False)
            # print(f"{results.missing_keys} missing keys")
            # print(f"{results.unexpected_keys} unexpected keys")
            print(f"Loaded DA3 pose regressor weights from {self.da_weights_path}")
        self.pose_regressor.prune_layers()

    def _build_transformer_stage(self, dim: int, nlayer: int) -> nn.ModuleList:
        """Create a list of identical TransformerBlocks for one transformer stage.

        Args:
            dim: Hidden dimension for all blocks in this stage.
            nlayer: Number of transformer blocks to create.

        Returns:
            An ``nn.ModuleList`` of ``nlayer`` TransformerBlock instances.
        """
        return nn.ModuleList([
            TransformerBlock(dim, False, self.head_dim, self.inter_multi, self.qk_norm)
            for _ in range(nlayer)
        ])

    @staticmethod
    def _create_patch_tokenizer(
        in_channels: int, patch_size: int, d_model: int,
    ) -> nn.Sequential:
        """Build a patch-embedding tokenizer: rearrange → linear projection → LayerNorm.

        Args:
            in_channels: Number of input image channels (e.g. 12 for ray-map images).
            patch_size: Side length of each square patch in pixels.
            d_model: Output embedding dimension.

        Returns:
            An ``nn.Sequential`` that maps ``(B, V, C, H, W)`` to ``(B, V*L, d_model)``.
        """
        return nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=patch_size, pw=patch_size,
            ),
            nn.Linear(in_channels * (patch_size ** 2), d_model, bias=False),
            nn.LayerNorm(d_model, bias=False),
        )

    # --- Training mode override ---

    def train(self, mode: bool = True):
        """Override train() to keep loss modules permanently in eval mode."""
        super().train(mode)
        if not self.inference_mode:
            self.loss_computer.eval()

    # --- Forward helpers ---

    def _build_raymap_input(
        self,
        input_data_dict: dict,
        i_fxfycxcy: torch.Tensor,
        i_c2w: torch.Tensor,
    ):
        """Construct plucker ray-map images: [ray_origin, ray_dir, origin×dir, image]."""
        h, w = input_data_dict["image"].shape[-2:]
        with torch.autocast(device_type="cuda", enabled=False):
            ray_o, ray_d = compute_plucmap(i_fxfycxcy, i_c2w, h, w)
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            i_normalized_image = input_data_dict["image"] * 2.0 - 1.0
            i_raymap_images = torch.concat([ray_o, ray_d, o_cross_d, i_normalized_image], dim=2)
            Ks = fxfycxcy_to_K(i_fxfycxcy)
            i_w2c = invert_SE3(i_c2w)
        return i_raymap_images, Ks, i_w2c

    def _render_and_compute_loss(
        self,
        gaussians: GaussianField,
        cameras: CameraBundle,
        target_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Render Gaussians to all target views and compute photometric + pose losses."""
        h, w = target_images.shape[-2:]
        with torch.autocast(device_type="cuda", enabled=False):
            renderings = GaussianRenderer.apply(
                gaussians.xyz, gaussians.feature, gaussians.scale,
                gaussians.rotation, gaussians.opacity_precompute,
                cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, w, h,
                self.sh_degree, self.near_plane, self.far_plane,
            )
        renderings = renderings.permute(0, 1, 4, 2, 3).contiguous()  # (B, V, 3, H, W)

        cam_info = {
            "pred_fxfycxcy": torch.cat([cameras.pred_i_fxfycxcy, cameras.pred_t_fxfycxcy], dim=1),
            "pred_c2w": torch.cat([cameras.pred_i_c2w, cameras.pred_t_c2w], dim=1),
            "gt_fxfycxcy": torch.cat([cameras.gt_i_fxfycxcy, cameras.gt_t_fxfycxcy], dim=1),
            "gt_c2w": torch.cat([cameras.gt_i_c2w, cameras.gt_t_c2w], dim=1),
        }
        loss_metrics = self.loss_computer(renderings, target_images, cam_info)

        with torch.autocast(device_type="cuda", enabled=False):
            rand_dirs = F.normalize(torch.randn_like(gaussians.xyz), p=2, dim=-1)
            opacity_random = _spherical_harmonics(
                self.opacity_degree, rand_dirs, gaussians.opacity,
            )
            opacity_random = opacity_random.sigmoid().mean()

        loss_metrics["opacity_loss"] = opacity_random * 0.001
        loss_metrics["loss"] = loss_metrics["loss"] + loss_metrics["opacity_loss"]
        return renderings, loss_metrics

    def _render_target_views(
        self,
        gaussians: GaussianField,
        t_c2w: torch.Tensor,
        t_fxfycxcy: torch.Tensor,
        w: int,
        h: int,
    ) -> torch.Tensor:
        """Render each target view sequentially without gradient tracking."""
        t = t_c2w.shape[1]
        xyz = gaussians.xyz[0]
        feature = gaussians.feature[0]
        scale = gaussians.scale[0]
        rotation = gaussians.rotation[0]
        opacity = gaussians.opacity[0]
        renderings = []
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            for i in range(t):
                dir = xyz - t_c2w[0, i:i+1, :3, 3][None, ...]
                opacity_i = _spherical_harmonics(
                    self.opacity_degree, dir, opacity[None, ...],
                )[0]
                rendering = GaussianRenderer.render(
                    xyz, feature, scale, rotation, opacity_i,
                    t_c2w[0, i], t_fxfycxcy[0, i], w, h,
                    self.sh_degree, self.near_plane, self.far_plane,
                )
                renderings.append(rendering)
        renderings = torch.cat(renderings, dim=0)[None, ...]   # (1, T, H, W, 3)
        return renderings.permute(0, 1, 4, 2, 3).contiguous()  # (1, T, 3, H, W)

    def _training_forward(
        self,
        gaussians: GaussianField,
        cameras: CameraBundle,
        target_data_dict: dict,
        input_data_dict: dict,
    ) -> edict:
        """Render Gaussians and compute all training losses.

        Returns:
            An edict with keys ``input``, ``target``, ``loss_metrics``, and ``render``.
        """
        renderings, loss_metrics = self._render_and_compute_loss(
            gaussians, cameras, target_data_dict["image"],
        )
        return edict(
            input=input_data_dict,
            target=target_data_dict,
            loss_metrics=loss_metrics,
            render=renderings,
        )

    def _inference_forward(
        self,
        gaussians: GaussianField,
        cameras: CameraBundle,
        target_data_dict: dict,
        input_data_dict: dict,
    ) -> edict:
        """Render target views at inference time, optionally with pose optimization.

        Returns:
            An edict with keys ``input``, ``target``, and ``render``.
        """
        h, w = target_data_dict["image"].shape[-2:]

        if self.use_pose_optimization:
            prev_grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            renderings, _ = self.pose_optimization(
                gaussians=gaussians,
                t_c2w=cameras.pred_t_c2w,
                t_fxfycxcy=cameras.pred_t_fxfycxcy,
                w=w, h=h,
                target_images=target_data_dict["image"],
            )
            torch.set_grad_enabled(prev_grad_state)
        else:
            renderings = self._render_target_views(
                gaussians, cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, w, h,
            )

        return edict(
            input=input_data_dict,
            target=target_data_dict,
            render=renderings,
        )

    # --- Main forward ---

    def forward(self, input_data_dict: dict, target_data_dict: dict) -> edict:
        """Full forward pass: predict cameras → build ray maps → predict Gaussians → render.

        Args:
            input_data_dict: Batch dict for context views, must contain ``image`` and
                camera ground-truth tensors.
            target_data_dict: Batch dict for novel views to render.

        Returns:
            An edict whose contents depend on the mode (training vs. inference).
        """
        cameras = self.geometry_expert.predict_cameras(input_data_dict, target_data_dict)
        raymap_images, Ks, i_w2c = self._build_raymap_input(
            input_data_dict, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w,
        )
        gaussians = self.appearance_expert.predict_gaussians(
            raymap_images, i_w2c, Ks,
            cameras.pred_i_fxfycxcy, cameras.pred_i_c2w, cameras.pred_t_c2w,
        )

        if not self.inference_mode:
            return self._training_forward(gaussians, cameras, target_data_dict, input_data_dict)
        return self._inference_forward(gaussians, cameras, target_data_dict, input_data_dict)

    # --- Pose optimization (optional inference refinement) ---

    def pose_optimization(
        self,
        gaussians: GaussianField,
        t_c2w: torch.Tensor,
        t_fxfycxcy: torch.Tensor,
        w: int,
        h: int,
        target_images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Refine target poses with 100 steps of Adam (EPA — evaluation-time pose alignment).

        Quaternion and translation parameters are jointly optimised against an MSE
        photometric loss while all Gaussian parameters remain frozen.

        Args:
            gaussians: Frozen Gaussian field from the appearance expert.
            t_c2w: Initial target camera-to-world matrices (1, T, 4, 4).
            t_fxfycxcy: Target camera intrinsics (1, T, 4).
            w: Image width in pixels.
            h: Image height in pixels.
            target_images: Ground-truth target images (1, T, 3, H, W).

        Returns:
            A tuple of (renderings, (refined_c2w, refined_fxfycxcy)) where
            renderings has shape (1, T, 3, H, W).
        """
        num_target_views = t_c2w.shape[1]
        quats = mat_to_quat(t_c2w[..., :3, :3]).clone().detach().requires_grad_(True)
        trans = t_c2w[..., :3, 3].clone().detach().requires_grad_(True)
        t_fxfycxcy = t_fxfycxcy.clone().detach()
        xyz = gaussians.xyz[0].detach()
        feature = gaussians.feature[0].detach()
        scale = gaussians.scale[0].detach()
        rotation = gaussians.rotation[0].detach()
        opacity = gaussians.opacity[0].detach()
        optimizer = torch.optim.Adam([quats, trans], lr=1e-4)

        for _ in range(100):
            R = quat_to_mat(quats / quats.norm(dim=-1, keepdim=True))
            t_c2w_new = torch.zeros_like(t_c2w)
            t_c2w_new[..., :3, :3] = R
            t_c2w_new[..., :3, 3] = trans
            t_c2w_new[..., 3, 3] = 1.0

            optimizer.zero_grad()
            renderings = []
            with torch.autocast(device_type="cuda", enabled=False):
                for i in range(num_target_views):
                    dir = xyz - t_c2w_new[0, i:i+1, :3, 3][None, ...]
                    opacity_i = _spherical_harmonics(
                        self.opacity_degree, dir, opacity[None, ...],
                    )[0]
                    rendering = GaussianRenderer.render(
                        xyz, feature, scale, rotation, opacity_i,
                        t_c2w_new[0, i], t_fxfycxcy[0, i], w, h,
                        self.sh_degree, self.near_plane, self.far_plane,
                    )
                    renderings.append(rendering)
            renderings = torch.cat(renderings, dim=0)[None, ...]

            loss = F.mse_loss(renderings.permute(0, 1, 4, 2, 3).contiguous(), target_images)
            loss.backward()
            optimizer.step()

        refined_c2w = t_c2w_new.detach()
        refined_fxfycxcy = t_fxfycxcy.detach()
        return renderings.permute(0, 1, 4, 2, 3).contiguous(), (refined_c2w, refined_fxfycxcy)

    # --- Checkpoint loading ---

    @torch.no_grad()
    def load_ckpt(self, load_path: str) -> int | None:
        """Load EMA weights from a checkpoint file or the latest .pt in a directory.

        The checkpoint is expected to be a dict with an ``"ema"`` key containing
        the state dict. Loading is done with ``strict=False`` so DA3 weights (loaded
        separately) do not cause missing-key errors.

        Args:
            load_path: Path to a ``.pt`` checkpoint file, or a directory containing
                one or more ``.pt`` files (the lexicographically last is used).

        Returns:
            0 on success, None on failure.
        """
        if os.path.isdir(load_path):
            ckpt_names = sorted(f for f in os.listdir(load_path) if f.endswith(".pt"))
            ckpt_path = os.path.join(load_path, ckpt_names[-1])
        else:
            ckpt_path = load_path
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_path}")
            return None
        result = self.load_state_dict(checkpoint["ema"], strict=False)
        # print(f"{result.missing_keys} missing keys")
        # print(f"{result.unexpected_keys} unexpected keys")
        print(f"Loaded 2Xplat checkpoint in {ckpt_path}")
        return 0
