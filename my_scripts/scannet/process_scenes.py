import argparse
import functools
import multiprocessing
import pathlib

import scannet_scene


def _worker(src_dir, tgt_dir):
    tgt_dir.mkdir(exist_ok=True, parents=True)
    scannet_scene.process_scene(src_dir, tgt_dir)
    print(f"Done {tgt_dir}")


def run(args):
    """Main function."""
    src_dirs = sorted(list(filter(lambda x: x.stem.startswith("scene"), args.scene_root.iterdir())))
    tasks = [(src_dir, args.output / src_dir.stem) for src_dir in src_dirs]
    with multiprocessing.Pool(processes=args.nproc) as pool:
        pool.starmap(_worker, tasks)


def parse_args():
    """Utility function for argument parsing."""
    parser = argparse.ArgumentParser(description="Transform a scene of scannet to nerfstudio format.")
    parser.add_argument("--output", type=pathlib.Path, default="./output", help="Output path.")
    parser.add_argument("--scene-root", type=pathlib.Path, required=True, help="Path to scannet's scenes")
    parser.add_argument("--nproc", type=int, default=8, help="Number processes of pool.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
