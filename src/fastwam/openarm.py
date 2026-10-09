"""OpenArm adapters shared by training and deployment."""

import torch
import torchvision.transforms.functional as TF

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class SparseVideoDataset(BaseLerobotDataset):
    # Decode the nine requested video frames, not all 33 state frames.
    presample_images = True


class OpenArmDataset(RobotVideoDataset):
    base_dataset_cls = SparseVideoDataset

    def __getitem__(self, idx):
        return self._get(idx)


class RelativeArmJoints:
    def __init__(self, indices):
        self.indices = list(indices)

    def forward(self, batch):
        if "action" in batch:
            action = batch["action"]["default"].clone()
            action[..., self.indices] -= batch["state"]["default"][..., :1, self.indices]
            batch["action"]["default"] = action
        return batch

    def backward(self, batch):
        action = batch["action"]["default"].clone()
        action[..., self.indices] += batch["state"]["default"][..., :1, self.indices]
        batch["action"]["default"] = action
        return batch


class CropResize:
    def __init__(self, crop_xyxy, size=(240, 320)):
        self.crop_xyxy = crop_xyxy
        self.size = list(size)

    def __call__(self, images):
        x0, y0, x1, y1 = self.crop_xyxy
        return TF.resize(images[..., y0:y1, x0:x1], self.size, antialias=True)


def pack_cameras(images, processor):
    """Same three-camera layout as RobotVideoDataset's robotwin mode."""
    frames = []
    for meta in processor.shape_meta["images"]:
        key = meta["key"]
        frame = torch.as_tensor(images[key])
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != torch.uint8:
            raise ValueError(f"{key} must be a uint8 RGB HWC image")
        frame = frame.permute(2, 0, 1).unsqueeze(0)
        for transform in processor.val_transforms[key]:
            frame = transform(frame)
        frames.append(frame)
    top = TF.resize(frames[0], [256, 320], antialias=True)
    bottom = torch.cat([TF.resize(x, [128, 160], antialias=True) for x in frames[1:]], -1)
    return torch.cat([top, bottom], -2) * 2 - 1


def encode_state(state, processor):
    state = torch.as_tensor(state, dtype=torch.float32).reshape(1, 1, -1)
    return processor.normalizer.normalizers["state"]["default"].forward(state)[:, 0]


def decode_actions(action, state, processor):
    unbatched = action.ndim == 2
    if unbatched:
        action = action.unsqueeze(0)
    action = processor.normalizer.normalizers["action"]["default"].backward(action.float().cpu())
    batch = {"action": {"default": action}, "state": {"default": torch.as_tensor(state).float().reshape(1, 1, -1)}}
    for transform in reversed(processor.action_state_transforms or []):
        batch = transform.backward(batch)
    result = batch["action"]["default"]
    return result[0] if unbatched else result
