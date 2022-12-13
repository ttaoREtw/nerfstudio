#!/bin/bash

# Use viewer will hang after training
ns-train instant-ngp --data data/nerfstudio/poster/ --vis tensorboard
# scannet scene0000_00
ns-train instant-ngp --data data/scannet/scene0000_00/ \
                     --trainer.steps-per-save 5000 \
                     --trainer.save-only-latest-checkpoint False \
                     --trainer.max-num-iterations 50000 \
                     --experiment-name scene0000_00 \
                     --vis tensorboard \
                     --pipeline.model.contraction-type AABB \
                     --pipeline.model.randomize-background False \
                     --optimizers.fields.optimizer.lr 0.01 \
                     --timestamp "aabb_lr_1e-2_b10px" \
                     --output-dir "/media/NFS/ttao/scannet/ns_outputs" \
                     scannet-data

# Use viewer to see the training result
LOAD_DIR="outputs/data-nerfstudio-poster/instant-ngp/2022-12-08_172125/nerfstudio_models/"
ns-train instant-ngp --data data/scannet/scene0000_00/ \
                     --trainer.load-dir ${LOAD_DIR} \
                     --trainer.load-step 28000 \
                     --viewer.start-train False \
                     --pipeline.model.contraction-type AABB \
                     scannet-data

# Export
ns-export voxel --load-config exp_pcd/aabb.yml --output-dir exp_pcd/aabb  --bounding-box-min -4 -4 -2 --bounding-box-max 4 4 2
