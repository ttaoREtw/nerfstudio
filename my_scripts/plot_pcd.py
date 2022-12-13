import argparse
import pathlib

import matplotlib.pyplot as plt

from scene_reconst import pcd


def run(args):
    fpath = args.pcd
    pts = pcd.read_pcd_from_ply(fpath)

    fig = plt.figure(dpi=300)
    ax = plt.axes(projection="3d")
    ax.view_init(45, 45)
    pts.plot(ax)
    fig.savefig(args.output)


def parse_args():
    """Utility function for argument parsing."""
    parser = argparse.ArgumentParser(description="Script for point cloud plotting.")
    parser.add_argument("--pcd", type=pathlib.Path, required=True, help="Input point cloud path.")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output path.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run(parse_args())
