import argparse
import pathlib
import random
import subprocess
import time
from typing import Optional

from loguru import logger


def parse_args():
    """Utility function for argument parsing."""
    parser = argparse.ArgumentParser(description="Script for auto training for all scannet scenes.")
    parser.add_argument(
        "--data-root", type=pathlib.Path, required=True, help="Root to all data containing transforms.json"
    )
    parser.add_argument(
        "--scene-list", type=pathlib.Path, required=True, help="A text file containing scene names to be processed"
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output directory.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--max-num-iterations", type=int, default=90000, help="Number of training iteration per nerf.")
    parser.add_argument("--steps-per-save", type=int, default=10000, help="Number of iteration per checkpoint.")
    args = parser.parse_args()
    return args


def get_ckpt_dir(
    exp_name: str,
    scene_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    return output_dir / scene_dir.stem / "instant-ngp" / exp_name / "nerfstudio_models"


def scene_is_done(
    exp_name: str,
    scene_dir: pathlib.Path,
    output_dir: pathlib.Path,
    max_num_iterations: int = 90000,
) -> bool:
    last_step = max_num_iterations - 1
    return (get_ckpt_dir(exp_name, scene_dir, output_dir) / f"step-{last_step:09d}.ckpt").exists()


def get_command(
    exp_name: str,
    scene_dir: pathlib.Path,
    output_dir: pathlib.Path,
    output_log_dir: pathlib.Path,
    steps_per_save: int = 10000,
    max_num_iterations: int = 90000,
    cuda_id: Optional[int] = None,
):
    """Run command.
    Template:
        COMMAND = (
            "ns-train instant-ngp --data data/scannet/scene0000_00/"
            " --trainer.steps-per-save 5000"
            " --trainer.save-only-latest-checkpoint False"
            " --trainer.max-num-iterations 50000"
            " --experiment-name scene0000_00"
            " --vis tensorboard"
            " --pipeline.model.contraction-type AABB"
            " --pipeline.model.randomize-background False"
            " --optimizers.fields.optimizer.lr 0.01"
            " --timestamp aabb_lr_1e-2_b10px"
            " --output-dir /media/NFS/ttao/scannet/ns_outputs"
            " scannet-data"
        )
    """

    command = ""
    if cuda_id is not None:
        command += f"CUDA_VISIBLE_DEVICES={cuda_id}"

    command += " ns-train instant-ngp"
    command += f" --data {scene_dir}"
    command += f" --trainer.steps-per-save {steps_per_save}"
    command += "  --trainer.save-only-latest-checkpoint False"
    command += f" --trainer.max-num-iterations {max_num_iterations}"
    command += "  --pipeline.model.contraction-type AABB"
    command += "  --pipeline.model.randomize-background False"
    command += "  --optimizers.fields.optimizer.lr 0.01"
    command += f" --experiment-name {scene_dir.stem}"
    command += "  --vis tensorboard"
    command += f" --timestamp {exp_name}"
    command += f" --output-dir {output_dir}"
    command += "  scannet-data"
    command += f" > {output_log_dir}/{scene_dir.stem}.log"
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
        self._id = self._popen.pid

    def is_done(self):
        # import random
        # return random.choice([True, False])
        if self._popen is None:
            return False
        done = self._popen.poll() is not None
        logger.info(f"Job-{self.id}'s status: " + ("DONE" if done else "WIP"))
        return done


class JobPool:
    def __init__(self):
        self._pools = {}

    def __repr__(self):
        return str(self._pools)

    def add(self, job: Job):
        logger.info(f"Add job-{job.id} to pool")
        self._pools[job.id] = job

    def submit(self):
        for job in self._pools.values():
            job.submit()

    def clear(self):
        logger.info("Clear all jobs")
        self._pools.clear()

    def is_done(self):
        return all(job.is_done() for job in self._pools.values())


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
    num_jobs_per_time = args.num_gpus

    exp_name = "v0"
    scenes = read_scene_list(args.scene_list, shuffle=True)
    scenes = [args.data_root / name for name in scenes]
    scenes = list(filter(lambda path: path.exists(), scenes))

    job_pool = JobPool()
    for i in range(0, len(scenes), num_jobs_per_time):
        for j in range(num_jobs_per_time):
            k = i + j
            if k < len(scenes):
                if not scene_is_done(exp_name, scenes[k], output_dir, args.max_num_iterations):
                    cmd = get_command(
                        exp_name,
                        scenes[k],
                        output_dir,
                        output_log_dir,
                        steps_per_save=args.steps_per_save,
                        max_num_iterations=args.max_num_iterations,
                        cuda_id=j,
                    )
                    job_pool.add(Job(cmd))
        job_pool.submit()

        while not job_pool.is_done():
            wait_sec = 30
            logger.info(f"Sleep {wait_sec} sec...")
            time.sleep(wait_sec)

        job_pool.clear()


if __name__ == "__main__":
    run(parse_args())
