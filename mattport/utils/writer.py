"""
Generic Writer class
"""


import datetime
import enum
import os
from abc import abstractmethod
from typing import Dict

import imageio
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

import mattport.utils.writer
from mattport.utils.decorators import check_main_thread, decorate_all

to8b = lambda x: (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)
EVENT_WRITERS = []
EVENT_STORAGE = []
GLOBAL_BUFFER = {}


class EventName(enum.Enum):
    """Names of possible events that can be logged via Local Writer for convenience.
    see config/logging/default_logging.yaml"""

    ITER_LOAD_TIME = "Data Load (time)"
    ITER_TRAIN_TIME = "Train Iter (time)"
    RAYS_PER_SEC = "Rays Per Sec (time)"
    TOTAL_TRAIN_TIME = "Train Total (time)"
    ETA = "ETA (time)"
    CURR_TEST_PSNR = "Test PSNR"


class EventType(enum.Enum):
    """Possible Event types and their associated write function"""

    IMAGE = "write_image"
    SCALAR = "write_scalar"
    DICT = "write_scalar_dict"


@check_main_thread
def put_image(name, image: TensorType["H", "W", "C"], step: int, group: str = None, prefix: str = None):
    """Setter function to place images into the queue to be written out"""
    if isinstance(name, EventName):
        name = name.value

    EVENT_STORAGE.append(
        {"name": name, "write_type": EventType.IMAGE, "event": image, "step": step, "group": group, "prefix": prefix}
    )


@check_main_thread
def put_scalar(name: str, scalar: float, step: int, group: str = None, prefix: str = None):
    """Setter function to place scalars into the queue to be written out"""
    if isinstance(name, EventName):
        name = name.value

    GLOBAL_BUFFER["new_key"] = not name in GLOBAL_BUFFER["events"] or GLOBAL_BUFFER["new_key"]
    GLOBAL_BUFFER["events"][name] = scalar
    EVENT_STORAGE.append(
        {"name": name, "write_type": EventType.SCALAR, "event": scalar, "step": step, "group": group, "prefix": prefix}
    )


@check_main_thread
def put_dict(name: str, scalar_dict: float, step: int, group: str = None, prefix: str = None):
    """Setter function to place a dictionary of scalars into the queue to be written out"""
    EVENT_STORAGE.append(
        {
            "name": name,
            "write_type": EventType.DICT,
            "event": scalar_dict,
            "step": step,
            "group": group,
            "prefix": prefix,
        }
    )


@check_main_thread
def put_time(
    name: str,
    start_time: float,
    end_time: float,
    step: int,
    group: str = None,
    prefix: str = None,
    avg_over_iters: bool = False,
    avg_over_batch: int = None,
    update_eta: bool = False,
):
    """Setter function to place a time element into the queue to be written out

    Processes the time info according to the options:
    avg_over_iters (bool): if True, calculate and record a running average of the times
    avg_over_batch (int): if set, the size of the batch for which we take the average over (batch/second)
    update_eta (bool): if True, update the ETA. should only be set for the training iterations/s
    """
    if isinstance(name, EventName):
        name = name.value

    GLOBAL_BUFFER["step"] = step
    GLOBAL_BUFFER["new_key"] = not name in GLOBAL_BUFFER["events"] or GLOBAL_BUFFER["new_key"]
    val = end_time - start_time
    if avg_over_batch:
        val = avg_over_batch / val

    if avg_over_iters:
        curr_event = GLOBAL_BUFFER["events"].get(name, {"buffer": [], "avg": 0})
        curr_buffer = curr_event["buffer"]
        curr_avg = curr_event["avg"]
        if len(curr_buffer) >= GLOBAL_BUFFER["max_buffer_size"]:
            curr_buffer.pop(0)
        curr_buffer.append(val)
        curr_avg = sum(curr_buffer) / len(curr_buffer)
        put_scalar(name, curr_avg, step, group, prefix)
        GLOBAL_BUFFER["events"][name] = {"buffer": curr_buffer, "avg": curr_avg}
    else:
        GLOBAL_BUFFER["events"][name] = val
        put_scalar(name, val, step, group, prefix)

    if update_eta:
        ## NOTE: eta should be called with avg train iteration time
        remain_iter = GLOBAL_BUFFER["max_iter"] - step
        remain_time = remain_iter * GLOBAL_BUFFER["events"][name]["avg"]
        GLOBAL_BUFFER["events"]["ETA"] = remain_time
        put_scalar(EventName.ETA, remain_time, step, group, prefix)


@check_main_thread
def write_out_storage():
    """Function that writes all the events in storage to all the writer locations"""
    for writer in EVENT_WRITERS:
        for event in EVENT_STORAGE:
            write_func = getattr(writer, event["write_type"].value)
            if event["write_type"] == EventType.DICT:
                write_func(event["event"], event["step"], event["group"], event["prefix"])
                if isinstance(writer, LocalWriter):
                    continue
            else:
                write_func(event["name"], event["event"], event["step"], event["group"], event["prefix"])
                if isinstance(writer, LocalWriter):
                    break
    EVENT_STORAGE.clear()


@check_main_thread
def setup_event_writers(config: DictConfig) -> None:
    """Initialization of all event writers specified in config"""
    logging_configs = config.logging.writer
    for writer_type in logging_configs:
        writer_class = getattr(mattport.utils.writer, writer_type)
        writer_config = logging_configs[writer_type]
        if writer_type == "LocalWriter":
            curr_writer = writer_class(writer_config.save_dir, writer_config.stats_to_track, writer_config.max_log_size)
        else:
            curr_writer = writer_class(writer_config.save_dir)
        EVENT_WRITERS.append(curr_writer)

    ## configure all the global buffer basic information
    GLOBAL_BUFFER["max_iter"] = config.graph.max_num_iterations
    GLOBAL_BUFFER["max_buffer_size"] = config.logging.max_buffer_size
    GLOBAL_BUFFER["steps_per_log"] = config.logging.steps_per_log
    GLOBAL_BUFFER["new_key"] = True
    GLOBAL_BUFFER["events"] = {}


def get_tensorboard_name(name: str, group: str = None, prefix: str = None):
    """Returns a string for tensorboard with an optional group and prefix.
    Where tensorboard_name has the form `group/prefix-name`.
    """
    group_string = f"{group}/" if group else ""
    prefix_string = f"{prefix}" if prefix else ""
    tensorboard_name = f"{group_string}{prefix_string}{name}"
    return tensorboard_name


class Writer:
    """Writer class"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def write_image(
        self, name: str, image: TensorType["H", "W", "C"], step: int, group: str = None, prefix: str = None
    ) -> None:
        """_summary_

        Args:
            name (str): data identifier
            image (TensorType["H", "W", "C"]): rendered image to write
            step (int): the time step to log
            group (str): the group e.g., "Loss", "Accuracy", "Time"
            prefix (str): the prefix e.g., "train-", "test-"
        """
        raise NotImplementedError

    @abstractmethod
    def write_scalar(self, name: str, scalar: float, step: int, group: str = None, prefix: str = None) -> None:
        """Required method to write a single scalar value to the logger

        Args:
            name (str): data identifier
            step (int): the time step to log
            group (str): the group e.g., "Loss", "Accuracy", "Time"
            prefix (str): the prefix e.g., "train-", "test-"
        """
        raise NotImplementedError

    @check_main_thread
    def write_scalar_dict(
        self, scalar_dict: Dict[str, float], step: int, group: str = None, prefix: str = None
    ) -> None:
        """Function that writes out all scalars from a given dictionary to the logger

        Args:
            scalar_dict (dict): dictionary containing all scalar values with key names and quantities
            step (int): the time step to log
            group (str): the group e.g., "Loss", "Accuracy", "Time"
            prefix (str): the prefix e.g., "train-", "test-"
        """
        for name, scalar in scalar_dict.items():
            self.write_scalar(name, scalar, step, group=group, prefix=prefix)


@decorate_all([check_main_thread])
class TensorboardWriter(Writer):
    """Tensorboard Writer Class"""

    def __init__(self, save_dir: str):
        super().__init__(save_dir)
        self.tb_writer = SummaryWriter(log_dir=self.save_dir)

    def write_image(
        self, name: str, image: TensorType["H", "W", "C"], step: int, group: str = None, prefix: str = None
    ) -> None:
        """_summary_

        Args:
            name (str): data identifier
            image (TensorType["H", "W", "C"]): rendered image to write
        """
        image = to8b(image)
        tensorboard_name = get_tensorboard_name(name, group=group, prefix=prefix)
        self.tb_writer.add_image(tensorboard_name, image, step, dataformats="HWC")

    def write_scalar(self, name: str, scalar: float, step: int, group: str = None, prefix: str = None) -> None:
        """Tensorboard method to write a single scalar value to the logger

        Args:
            name (str): data identifier
            x (float): x value to write
            y (float): y value to write
            group (str)): a prefix to group tensorboard scalars
        """
        tensorboard_name = get_tensorboard_name(name, group=group, prefix=prefix)
        self.tb_writer.add_scalar(tensorboard_name, scalar, step)


def _cursorup(x: int):
    """utility tool to move the cursor up on the terminal

    Args:
        x (int): amount of lines to move cursor upward
    """
    print(f"\r\033[{x}A", end="\x1b[1K\r")


def _format_time(v):
    if v < 1.0:
        ms = v * 1e3
        v = f"{ms:0.4f} ms"
    else:
        v = str(datetime.timedelta(seconds=v))
    return v


def _consolidate_events():
    latest_map = {}
    for event in EVENT_STORAGE:
        name = event["name"]
        latest_map[name] = event["event"]
    return latest_map


@decorate_all([check_main_thread])
class LocalWriter(Writer):
    """Local Writer Class"""

    def __init__(self, save_dir: str, stats_to_track: ListConfig, max_log_size: int = 0):
        """
        Args:
            stats_to_track (ListConfig): the names of stats that should be logged.
            max_log size (int): max number of lines that will be logged to teminal.
        """
        super().__init__(save_dir)
        self.stats_to_track = [EventName[name].value for name in stats_to_track]
        self.max_log_size = max_log_size
        self.past_stats = []
        self.max_mssg_len = 0

    def write_image(
        self, name: str, image: TensorType["H", "W", "C"], step: int, group: str = None, prefix: str = None
    ) -> None:
        if name in self.stats_to_track:
            image = to8b(image)
            image_path = os.path.join(self.save_dir, f"{name}.jpg")
            imageio.imwrite(image_path, np.uint8(image.cpu().numpy() * 255.0))

    def write_scalar(self, name: str, scalar: float, step: int, group: str = None, prefix: str = None) -> None:
        latest_map = _consolidate_events()
        self.handle_header(latest_map)
        self.handle_stats(latest_map)
        GLOBAL_BUFFER["new_key"] = False

    def handle_header(self, latest_map):
        """helper to handle the printing of the header labels"""
        if (not self.max_log_size and self.max_mssg_len == 0) or (
            self.max_log_size and (len(self.past_stats) == 0 or GLOBAL_BUFFER["new_key"])
        ):
            mssg = f"{'Step (% Done)':<20}"
            for name, _ in latest_map.items():
                if name in self.stats_to_track:
                    mssg += f"{name:<20} "
            if self.max_log_size:
                if len(self.past_stats) > 1:
                    _cursorup(len(self.past_stats) - 1)
            print(mssg)
            print("-" * len(mssg))
            if self.max_log_size:
                if len(self.past_stats) > 0:
                    for mssg in self.past_stats:
                        print(mssg)
            self.max_mssg_len = len(mssg)

    def handle_stats(self, latest_map, padding=" "):
        """helper to print out the stats in a readable format"""
        step = GLOBAL_BUFFER["step"]
        fraction_done = step / GLOBAL_BUFFER["max_iter"]
        curr_mssg = f"{step} ({fraction_done*100:.02f}%)"
        curr_mssg = f"{curr_mssg:<20}"
        for name, v in latest_map.items():
            if name in self.stats_to_track:
                if "(time)" in name:
                    v = _format_time(v)
                else:
                    v = f"{v:0.4f}"
                curr_mssg += f"{v:<20} "

        # update the history buffer
        if self.max_log_size:
            if len(self.past_stats) > self.max_log_size:
                self.past_stats.pop(0)
                _cursorup(len(self.past_stats))
            elif len(self.past_stats) > 1:
                _cursorup(len(self.past_stats) - 1)

            for mssg in self.past_stats:
                print(f"{mssg:{padding}<{self.max_mssg_len}}")
            self.past_stats.append(curr_mssg)
        else:
            print(curr_mssg)
