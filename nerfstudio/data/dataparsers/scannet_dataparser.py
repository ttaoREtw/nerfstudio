# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for scannet dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class ScanNetDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: ScanNet)
    """target class to instantiate"""
    data: Path = Path("data/scannet")
    """Directory specifying location of data."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    scene_box_min = (-8.0, -8.0, -4.0)
    """The minimum of the scene box."""
    scene_box_max = (8.0, 8.0, 4.0)
    """The maximum of the scene box."""


@dataclass
class ScanNet(DataParser):
    """ScanNet Dataset"""

    config: ScanNetDataParserConfig

    def __init__(self, config: ScanNetDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

    def _get_split_indices(self, num_images, split):
        # Filter image_filenames and poses based on train/eval split percentage
        num_train_images = round(num_images * self.config.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        return indices

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.data / "transforms.json")
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            pose = np.array(frame["transform_matrix"])
            if np.all(np.isfinite(pose)):
                poses.append(pose)
                image_filenames.append(Path(frame["file_path"]))

        poses = np.array(poses)
        # Axis align matrix to align world coordinate with room layout.
        axis_align_mat = np.array(meta["axis_align_matrix"])
        poses = axis_align_mat @ poses

        # Since scannet uses +z as looking-at direction in camera space, flip y and z axes to
        # align nerfstudio. See https://docs.nerf.studio/en/latest/quickstart/data_conventions.html.
        poses[..., :, 1] *= -1
        poses[..., :, 2] *= -1

        # Train / eval split
        indices = self._get_split_indices(len(image_filenames), split)
        image_filenames = [image_filenames[i] for i in indices]
        # Mask out the black border due to some kind of distortion
        mask_filenames = [self.data / "mask_border_10px.png"] * len(image_filenames)
        poses = poses[indices]

        camera_to_world = torch.from_numpy(poses[:, :3].astype(np.float32))
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=meta["fl_x"],
            fy=meta["fl_y"],
            cx=meta["cx"],
            cy=meta["cy"],
            width=meta["w"],
            height=meta["h"],
            camera_type=CameraType.PERSPECTIVE,
        )

        # In x,y,z order
        scene_box = SceneBox(
            aabb=torch.tensor([self.config.scene_box_min, self.config.scene_box_max], dtype=torch.float32)
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )

        return dataparser_outputs
