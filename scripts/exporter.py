"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import tyro
from rich.console import Console
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (
    generate_point_cloud,
    get_mesh_from_filename,
)
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


@dataclass
class ExportVoxelDensity(Exporter):
    """Export occupancy Density."""

    number_voxel: Tuple[int, int, int] = (256, 256, 128)
    """Output voxel size."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Maximum of the bounding box."""
    sample_points_per_voxel: int = 128
    """Number of sample points to determine the density of each voxels."""
    vis_alpha_thres: float = 0.99
    """Alpha threshold."""
    eval_chunk_size: int = 409600

    def __post_init__(self):
        number_voxel, voxel_size, xyz_min, xyz_max = self._compute_world_meta()
        self.number_voxel = number_voxel
        self.voxel_size = voxel_size
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

    def main(self) -> None:
        """Export occupancy grid."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)
        # (H * W * D, 3)
        pts = self._compute_voxel_grids()
        # (H * W * D, # of samples, 3)
        positions, views = self._create_voxel_samples(pts)

        _, num_sample_points_per_voxel, _ = positions.shape
        alpha_all = []
        color_all = []
        with torch.no_grad():
            position_chunks = torch.split(positions, self.eval_chunk_size, dim=0)
            views_chunks = torch.split(views, self.eval_chunk_size, dim=0)
            for i, (pos, vw) in enumerate(zip(position_chunks, views_chunks)):
                CONSOLE.print(f"Extracting chunk {i+1}/{len(position_chunks)}...")
                density, color = pipeline.model.field.get_outputs_from_position(
                    pos.to(pipeline.model.device), vw.to(pipeline.model.device)
                )
                density = density.reshape(-1, num_sample_points_per_voxel)
                alpha = 1.0 - torch.exp(-density)
                color = color.reshape(-1, num_sample_points_per_voxel, 3)
                alpha_all.append(alpha.mean(dim=-1).cpu())
                color_all.append(color.mean(dim=-2).cpu())

        alpha = torch.cat(alpha_all, dim=0)
        color = torch.cat(color_all, dim=0)

        voxel_position = pts.float().numpy()
        voxel_alpha = alpha.float().numpy()
        voxel_color = color.float().numpy()

        mask = voxel_alpha > self.vis_alpha_thres
        voxel_position = voxel_position[mask]
        voxel_alpha = voxel_alpha[mask]
        voxel_color = voxel_color[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_position)
        pcd.colors = o3d.utility.Vector3dVector(voxel_color)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving voxels...")
        o3d.io.write_point_cloud(str(self.output_dir / "voxel.ply"), pcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving voxel")

    def _compute_world_meta(self):
        number_voxel = torch.tensor(self.number_voxel).long()
        xyz_min = torch.tensor(self.bounding_box_min).float()
        xyz_max = torch.tensor(self.bounding_box_max).float()
        scene_length = xyz_max - xyz_min
        voxel_size = scene_length / number_voxel
        return (number_voxel, voxel_size, xyz_min, xyz_max)

    def _compute_sample_offsets(self):
        torch.random.manual_seed(1028)
        # [0, 1)
        offs = torch.rand(self.sample_points_per_voxel, 3)
        # [-1, 1)
        offs = offs * 2 - 1
        # Sample around center
        offs = 0.5 * offs
        # [-voxel_size/2, voxel_size/2)
        offs = offs * self.voxel_size / 2.0
        return offs

    def _compute_voxel_grids(self):
        x, y, z = torch.meshgrid(
            torch.linspace(0, self.number_voxel[0] - 1, self.number_voxel[0]),
            torch.linspace(0, self.number_voxel[1] - 1, self.number_voxel[1]),
            torch.linspace(0, self.number_voxel[2] - 1, self.number_voxel[2]),
            indexing="ij",
        )
        # (H * W * D, 3)
        pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        # Scale the points to fit the scene
        pts = pts * self.voxel_size
        # Shift the min point to xyz_min
        pts = pts + self.xyz_min
        # Shift each point to the center of voxels
        pts = pts + self.voxel_size / 2.0
        return pts

    @torch.no_grad()
    def _create_voxel_samples(self, pts):
        # Make samples for each voxel
        offs = self._compute_sample_offsets()
        # (H * W * D, # samples, 3)
        pts = pts.unsqueeze(-2) + offs
        views = -offs
        views = views / views.norm(dim=-1, keepdim=True)
        return pts, views.expand(pts.shape)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    estimate_normals: bool = False
    """Estimate normals for the point cloud."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            estimate_normals=self.estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def validate_pipeline(self, pipeline: Pipeline) -> None:
        """Check that the pipeline is valid for this exporter."""
        if self.normal_method == "model_output":
            CONSOLE.print("Checking that the pipeline has a normal output.")
            origins = torch.zeros((1, 3), device=pipeline.device)
            directions = torch.ones_like(origins)
            pixel_area = torch.ones_like(origins[..., :1])
            camera_indices = torch.zeros_like(origins[..., :1])
            ray_bundle = RayBundle(
                origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
            )
            outputs = pipeline.model(ray_bundle)
            if self.normal_output_name not in outputs:
                CONSOLE.print(
                    f"[bold yellow]Warning: Normal output '{self.normal_output_name}' not found in pipeline outputs."
                )
                CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
                CONSOLE.print(
                    "[bold yellow]Warning: Please train a model with normals "
                    "(e.g., nerfacto with predicted normals turned on)."
                )
                CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
                CONSOLE.print("[bold yellow]Exiting early.")
                sys.exit(1)

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)
        self.validate_pipeline(pipeline)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """
    NOT YET IMPLEMENTED
    Export a mesh using marching cubes.
    """

    def main(self) -> None:
        """Export mesh"""
        raise NotImplementedError("Marching cubes not implemented yet.")


Commands = Union[
    Annotated[ExportVoxelDensity, tyro.conf.subcommand(name="voxel")],
    Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
    Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
    Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
    Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[Commands]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
