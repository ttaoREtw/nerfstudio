import pathlib
import shutil

import tqdm

src = pathlib.Path("/media/NFS/ttao/scannet/scannet_instance_data")
tgt = pathlib.Path("/media/NFS/ttao/scannet/posed_images")


for fpath in tqdm.tqdm(sorted(list(src.glob("*axis_align_matrix.npy")))):
    scene_name = fpath.stem.replace("_axis_align_matrix", "")
    # print(f'{str(fpath)} -> {tgt / scene_name / "axis_align_matrix.npy"}')
    shutil.copy(str(fpath), str(tgt / scene_name / "axis_align_matrix.npy"))
