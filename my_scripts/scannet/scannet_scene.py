"""Process scannet scene to transform.json"""
import argparse
import dataclasses
import json
import pathlib

import imageio.v2 as imageio
import numpy as np
import pandas as pd


def _load_sharpness(fpath):
    table = pd.read_csv(fpath, header=None, names=["frame", "sharpness"])
    return table


def _load_matrix(fpath):
    fpath = pathlib.Path(fpath)
    ext = fpath.suffix
    if ext == ".txt":
        return np.loadtxt(fpath)
    elif ext == ".npy":
        return np.load(fpath)
    else:
        raise NotImplementedError(f"Only support .txt and .npy formats but got {ext}.")


@dataclasses.dataclass
class ScanNetScene:
    """A scene of scannet"""

    scene_dir: pathlib.Path
    """Path to scene directory, e.g., scans/scene0000_00."""
    image_ext: str = "jpg"
    """The extension of images."""
    pose_ext: str = "txt"
    """The extension of poses."""
    intrinsic_filename: pathlib.Path = pathlib.Path("intrinsic.txt")
    """The filename of intrinsic."""
    axis_align_matrix_filename: pathlib.Path = pathlib.Path("axis_align_matrix.npy")
    """The filename of axis align matrix."""
    sharpness_filename: pathlib.Path = pathlib.Path("sharpness.csv")
    """The filename of sharpness table."""
    number_selection_bins: int = 300
    """The number of selection bins."""

    def process(self, output: pathlib.Path):
        """Process scene and dump to output."""
        selelcted_frames = self._select_frames()
        meta = {}
        meta["frames"] = []
        for image_path in self.scene_dir.glob(f"*.{self.image_ext}"):
            if int(image_path.stem) in selelcted_frames:
                pose_path = str(image_path).replace(f".{self.image_ext}", f".{self.pose_ext}")
                pose = _load_matrix(pose_path)
                meta["frames"].append(dict(file_path=str(image_path), transform_matrix=pose.tolist()))

        img_0 = imageio.imread(meta["frames"][0]["file_path"])
        meta["h"] = img_0.shape[0]
        meta["w"] = img_0.shape[1]

        intrinsic = _load_matrix(self.scene_dir / self.intrinsic_filename)
        meta["fl_x"] = intrinsic[0, 0]
        meta["fl_y"] = intrinsic[1, 1]
        meta["cx"] = intrinsic[0, 2]
        meta["cy"] = intrinsic[1, 2]

        axis_align_matrix = _load_matrix(self.scene_dir / self.axis_align_matrix_filename)
        meta["axis_align_matrix"] = axis_align_matrix.tolist()

        with open(output / "transforms.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

    def _select_frames(self):
        table_sharpness = _load_sharpness(self.scene_dir / self.sharpness_filename)
        table_sharpness = table_sharpness.sort_values(by=["frame"])
        num_frames = table_sharpness.shape[0]
        num_bins = min(self.number_selection_bins, num_frames - 1)
        bin_boundaries = np.linspace(0, num_frames - 1, num_bins + 1, dtype=int)
        selected_frames = []
        for i in range(1, len(bin_boundaries)):
            start = bin_boundaries[i - 1]
            end = bin_boundaries[i]
            sorted_bin = table_sharpness.iloc[start:end].sort_values(by=["sharpness"], ascending=False)
            frame, _ = sorted_bin.iloc[0].tolist()
            selected_frames.append(int(frame))
        return set(selected_frames)


def process_scene(scene_dir, output_dir):
    """Process a scene."""
    output_dir.mkdir(exist_ok=True)
    scene = ScanNetScene(scene_dir)
    scene.process(output_dir)


def parse_args():
    """Utility function for argument parsing."""
    parser = argparse.ArgumentParser(description="Transform a scene of scannet to nerfstudio format.")
    parser.add_argument("--output", type=pathlib.Path, default="./output", help="Output path.")
    parser.add_argument("--scene", type=pathlib.Path, required=True, help="Path to scannet's scene")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_scene(args.scene, args.output)
