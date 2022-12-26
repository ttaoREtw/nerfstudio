import argparse
import os
import pathlib
import random
import subprocess
import time
from typing import List, Optional

from loguru import logger


exp_name = "v0"


def parse_args():
    """Utility function for argument parsing."""
    parser = argparse.ArgumentParser(description="Script for auto training for all scannet scenes.")
    parser.add_argument(
        "--ns-output-root", type=pathlib.Path, required=True, help="Root to all nerfstudio output files."
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output directory.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--load-step", type=int, default=40000, help="Which checkpoint to use.")
    parser.add_argument("--min-valid-voxels", type=int, default=300000, help="Number of minimum voxels to export.")
    args = parser.parse_args()
    return args


def get_command(
    exp_name: str,
    scene_name: str,
    ns_output_root: pathlib.Path,
    output_dir: pathlib.Path,
    output_log_dir: pathlib.Path,
    min_valid_voxels: int, 
    load_step: int = 50000,
    cuda_id: Optional[int] = None,
):
    """Run command.
    Template:
        COMMAND = (
            "ns-export voxel"
            " --load-config /media/NFS/ttao/scannet/ns_outputs_90k/scene0000_02/instant-ngp/v0/config.yml"
            " --output-dir exp_pcd/scene0000_02"
            " --load-step 50000"
        )
    """

    command = ""
    if cuda_id is not None:
        command += f"CUDA_VISIBLE_DEVICES={cuda_id}"

    command += " ns-export voxel"
    command += f" --load-config {ns_output_root}/{scene_name}/instant-ngp/{exp_name}/config.yml"

    command += f" --output-dir {output_dir / exp_name / scene_name}"
    command += f" --load-step {load_step}"
    command += f" --min-valid-voxels {min_valid_voxels}"
    command += f" > {output_log_dir}/{scene_name}.log"
    return command


class Job:
    ID_COUNTER = 0

    def __init__(self, command: str):
        self._cmd = command
        self._popen = None
        self._id = Job.ID_COUNTER
        Job.ID_COUNTER += 1

    @property
    def command(self):
        return self._cmd

    @command.setter
    def command(self, cmd):
        self._cmd = cmd

    @property
    def id(self):
        return self._id

    @property
    def pid(self):
        if self._popen is None:
            return None
        return self._popen.pid

    def submit(self):
        logger.info(f"Submit job {self.id}: {self.command}")
        self._popen = subprocess.Popen(self._cmd, shell=True)

    def is_done(self):
        if self._popen is None:
            return False
        done = self._popen.poll() is not None
        logger.info(f"Job-{self.id}'s status: " + ("DONE" if done else "WIP"))
        return done


class Device:
    def __init__(self, device_id):
        self._id = device_id
        self._job = None

    def run(self, job: Job):
        logger.info(f"Run job {job.id} on {self}")
        if not self.is_idle:
            logger.warning(f"{self} is not idle")
        job.command = self.to_command_prefix() + " " + job.command
        job.submit()
        self._job = job

    def is_idle(self):
        return self._job is None or self._job.is_done()

    def __repr__(self):
        return f"cuda-device-{self._id}"

    def to_command_prefix(self):
        return f"CUDA_VISIBLE_DEVICES={self._id}"


def _get_available_devices():
    devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if devices is None or not devices:
        return None
    return sorted(list(set(map(int, devices.strip().split(",")))))


class DeviceManager:
    def __init__(self, num_devices):
        devices = _get_available_devices()
        if devices is None:
            devices = [i for i in range(num_devices)]

        self._devices = [Device(i) for i in devices[:num_devices]]

    def assign(self, job: Job):
        logger.info(f"Assign job {job.id}")
        for device in self._devices:
            if device.is_idle():
                device.run(job)
                return
        logger.warning(f"Failed to assign job {job.id}")

    def is_any_idle(self):
        return any(device.is_idle() for device in self._devices)


def read_scene_list(path: pathlib.Path, shuffle):
    with open(path, encoding="utf-8") as f:
        scenes = f.read().splitlines()
    if shuffle:
        random.shuffle(scenes)
    return scenes


def run(args):
    # TEST_COMMAND = "python -c 'import time; time.sleep(2); print(\"done\")'"
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    output_log_dir = output_dir / "logs"
    output_log_dir.mkdir(exist_ok=True)
    num_gpus = args.num_gpus

    ns_output_root = args.ns_output_root

    scenes = sorted(list(map(lambda path: path.stem, ns_output_root.glob("scene*"))))

    device_manager = DeviceManager(num_gpus)
    i = 0
    while i < len(scenes):
        if device_manager.is_any_idle():
            cmd = get_command(
                exp_name,
                scenes[i],
                ns_output_root,
                output_dir,
                output_log_dir,
                args.min_valid_voxels,
                load_step=args.load_step,
            )
            device_manager.assign(Job(cmd))
            i += 1
        else:
            wait_sec = 10
            logger.info(f"Sleep {wait_sec} sec...")
            time.sleep(wait_sec)


if __name__ == "__main__":
    run(parse_args())
