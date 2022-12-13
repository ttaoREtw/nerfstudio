Implementation of Instant-NGP. Recommended real-time model for bounded 
synthetic data.

╭─ arguments ────────────────────────────────────────────────────────────────╮
│ -h, --help                                                                 │
│     show this help message and exit                                        │
│ --output-dir PATH                                                          │
│     relative or absolute output directory to save all checkpoints and      │
│     logging (default: outputs)                                             │
│ --method-name {None}|STR                                                   │
│     Method name. Required to set in python or via cli (default:            │
│     instant-ngp)                                                           │
│ --experiment-name {None}|STR                                               │
│     Experiment name. If None, will automatically be set to dataset name    │
│     (default: None)                                                        │
│ --timestamp STR                                                            │
│     Experiment timestamp. (default: '{timestamp}')                         │
│ --vis {viewer,wandb,tensorboard}                                           │
│     Which visualizer to use. (default: viewer)                             │
│ --data {None}|PATH                                                         │
│     Alias for --pipeline.datamanager.dataparser.data (default: None)       │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ machine arguments ────────────────────────────────────────────────────────╮
│ Machine configuration                                                      │
│ ────────────────────────────────────────────────────────────────────────── │
│ --machine.seed INT                                                         │
│     random seed initilization (default: 42)                                │
│ --machine.num-gpus INT                                                     │
│     total number of gpus available for train/eval (default: 1)             │
│ --machine.num-machines INT                                                 │
│     total number of distributed machines available (for DDP) (default: 1)  │
│ --machine.machine-rank INT                                                 │
│     current machine's rank (for DDP) (default: 0)                          │
│ --machine.dist-url STR                                                     │
│     distributed connection point (for DDP) (default: auto)                 │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ logging arguments ────────────────────────────────────────────────────────╮
│ Logging configuration                                                      │
│ ────────────────────────────────────────────────────────────────────────── │
│ --logging.relative-log-dir PATH                                            │
│     relative path to save all logged events (default: .)                   │
│ --logging.steps-per-log INT                                                │
│     number of steps between logging stats (default: 10)                    │
│ --logging.max-buffer-size INT                                              │
│     maximum history size to keep for computing running averages of stats.  │
│     e.g. if 20, averages will be computed over past 20 occurances.         │
│     (default: 20)                                                          │
│ --logging.enable-profiler {True,False}                                     │
│     whether to enable profiling code; prints speed of functions at the end │
│     of a program. profiler logs run times of functions and prints at end   │
│     of training (default: True)                                            │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ logging.local-writer arguments ───────────────────────────────────────────╮
│ if provided, will print stats locally. if None, will disable printing      │
│ ────────────────────────────────────────────────────────────────────────── │
│ --logging.local-writer.enable {True,False}                                 │
│     if True enables local logging, else disables (default: True)           │
│ --logging.local-writer.stats-to-track                                      │
│ {ITER_TRAIN_TIME,TOTAL_TRAIN_TIME,ITER_VIS_TIME,ETA,TRAIN_RAYS_PER_SEC,TE… │
│ [...]                                                                      │
│     specifies which stats will be logged/printed to terminal (default:     │
│     ITER_TRAIN_TIME TRAIN_RAYS_PER_SEC CURR_TEST_PSNR VIS_RAYS_PER_SEC     │
│     TEST_RAYS_PER_SEC)                                                     │
│ --logging.local-writer.max-log-size INT                                    │
│     maximum number of rows to print before wrapping. if 0, will print      │
│     everything. (default: 10)                                              │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ viewer arguments ─────────────────────────────────────────────────────────╮
│ Viewer configuration                                                       │
│ ────────────────────────────────────────────────────────────────────────── │
│ --viewer.relative-log-filename STR                                         │
│     Filename to use for the log file. (default: viewer_log_filename.txt)   │
│ --viewer.start-train {True,False}                                          │
│     whether to immediately start training upon loading viewer if False,    │
│     will just visualize dataset but you can toggle training in viewer      │
│     (default: True)                                                        │
│ --viewer.zmq-port {None}|INT                                               │
│     The zmq port to connect to for communication. If None, find an         │
│     available port. (default: None)                                        │
│ --viewer.launch-bridge-server {True,False}                                 │
│     whether or not to launch the bridge server (default: True)             │
│ --viewer.websocket-port {None}|INT                                         │
│     the default websocket port to connect to (default: 7007)               │
│ --viewer.ip-address STR                                                    │
│     the ip address where the bridge server is running (default: 127.0.0.1) │
│ --viewer.num-rays-per-chunk INT                                            │
│     number of rays per chunk to render with viewer (default: 64000)        │
│ --viewer.max-num-display-images INT                                        │
│     Maximum number of training images to display in the viewer, to avoid   │
│     lag. This does not change which images are actually used in            │
│     training/evaluation. If -1, display all. (default: 512)                │
│ --viewer.quit-on-train-completion {True,False}                             │
│     Whether to kill the training job when it has completed. Note this will │
│     stop rendering in the viewer. (default: False)                         │
│ --viewer.skip-openrelay {True,False}                                       │
│     Avoid using openrelay to communicate with the viewer. Try disabling if │
│     you have trouble connecting to the viewer (default: False)             │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ trainer arguments ────────────────────────────────────────────────────────╮
│ Trainer configuration                                                      │
│ ────────────────────────────────────────────────────────────────────────── │
│ --trainer.steps-per-save INT                                               │
│     Number of steps between saves. (default: 2000)                         │
│ --trainer.steps-per-eval-batch INT                                         │
│     Number of steps between randomly sampled batches of rays. (default:    │
│     500)                                                                   │
│ --trainer.steps-per-eval-image INT                                         │
│     Number of steps between single eval images. (default: 500)             │
│ --trainer.steps-per-eval-all-images INT                                    │
│     Number of steps between eval all images. (default: 25000)              │
│ --trainer.max-num-iterations INT                                           │
│     Maximum number of iterations to run. (default: 30000)                  │
│ --trainer.mixed-precision {True,False}                                     │
│     Whether or not to use mixed precision for training. (default: True)    │
│ --trainer.relative-model-dir PATH                                          │
│     Relative path to save all checkpoints. (default: nerfstudio_models)    │
│ --trainer.save-only-latest-checkpoint {True,False}                         │
│     Whether to only save the latest checkpoint or all checkpoints.         │
│     (default: True)                                                        │
│ --trainer.load-dir {None}|PATH                                             │
│     Optionally specify a pre-trained model directory to load from.         │
│     (default: None)                                                        │
│ --trainer.load-step {None}|INT                                             │
│     Optionally specify model step to load from; if none, will find most    │
│     recent model in load_dir. (default: None)                              │
│ --trainer.load-config {None}|PATH                                          │
│     optional parameters if we want to resume training (default: None)      │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.datamanager arguments ───────────────────────────────────────────╮
│ specifies the datamanager config                                           │
│ ────────────────────────────────────────────────────────────────────────── │
│ --pipeline.datamanager.train-num-rays-per-batch INT                        │
│     Number of rays per batch to use per training iteration. (default:      │
│     8192)                                                                  │
│ --pipeline.datamanager.train-num-images-to-sample-from INT                 │
│     Number of images to sample during training iteration. (default: -1)    │
│ --pipeline.datamanager.train-num-times-to-repeat-images INT                │
│     When not training on all images, number of iterations before picking   │
│     new images. If -1, never pick new images. (default: -1)                │
│ --pipeline.datamanager.eval-num-rays-per-batch INT                         │
│     Number of rays per batch to use per eval iteration. (default: 1024)    │
│ --pipeline.datamanager.eval-num-images-to-sample-from INT                  │
│     Number of images to sample during eval iteration. (default: -1)        │
│ --pipeline.datamanager.eval-num-times-to-repeat-images INT                 │
│     When not evaluating on all images, number of iterations before picking │
│     new images. If -1, never pick new images. (default: -1)                │
│ --pipeline.datamanager.eval-image-indices {None}|{INT [INT ...]}           │
│     Specifies the image indices to use during eval; if None, uses all.     │
│     (default: 0)                                                           │
│ --pipeline.datamanager.camera-res-scale-factor FLOAT                       │
│     The scale factor for scaling spatial data such as images, mask,        │
│     semantics along with relevant information about camera intrinsics      │
│     (default: 1.0)                                                         │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.datamanager.camera-optimizer arguments ──────────────────────────╮
│ Specifies the camera pose optimizer used during training. Helpful if poses │
│ are noisy, such as for data from Record3D.                                 │
│ ────────────────────────────────────────────────────────────────────────── │
│ --pipeline.datamanager.camera-optimizer.mode {off,SO3xR3,SE3}              │
│     Pose optimization strategy to use. If enabled, we recommend SO3xR3.    │
│     (default: off)                                                         │
│ --pipeline.datamanager.camera-optimizer.position-noise-std FLOAT           │
│     Noise to add to initial positions. Useful for debugging. (default:     │
│     0.0)                                                                   │
│ --pipeline.datamanager.camera-optimizer.orientation-noise-std FLOAT        │
│     Noise to add to initial orientations. Useful for debugging. (default:  │
│     0.0)                                                                   │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.datamanager.camera-optimizer.optimizer arguments ────────────────╮
│ ADAM parameters for camera optimization.                                   │
│ ────────────────────────────────────────────────────────────────────────── │
│ --pipeline.datamanager.camera-optimizer.optimizer.lr FLOAT                 │
│     (default: 0.0006)                                                      │
│ --pipeline.datamanager.camera-optimizer.optimizer.eps FLOAT                │
│     (default: 1e-15)                                                       │
│ --pipeline.datamanager.camera-optimizer.optimizer.weight-decay FLOAT       │
│     (default: 0)                                                           │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.datamanager.camera-optimizer.scheduler arguments ────────────────╮
│ Learning rate scheduler for camera optimizer..                             │
│ ────────────────────────────────────────────────────────────────────────── │
│ --pipeline.datamanager.camera-optimizer.scheduler.lr-final FLOAT           │
│     (default: 5e-06)                                                       │
│ --pipeline.datamanager.camera-optimizer.scheduler.max-steps INT            │
│     (default: 10000)                                                       │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.model arguments ─────────────────────────────────────────────────╮
│ specifies the model config                                                 │
│ ────────────────────────────────────────────────────────────────────────── │
│ --pipeline.model.enable-collider {True,False}                              │
│     Whether to create a scene collider to filter rays. (default: False)    │
│ --pipeline.model.collider-params {None}|{STR FLOAT [STR FLOAT ...]}        │
│     Instant NGP doesn't use a collider. (default: None)                    │
│ --pipeline.model.eval-num-rays-per-chunk INT                               │
│     specifies number of rays per chunk during eval (default: 8192)         │
│ --pipeline.model.max-num-samples-per-ray INT                               │
│     Number of samples in field evaluation. (default: 24)                   │
│ --pipeline.model.grid-resolution INT                                       │
│     Resolution of the grid used for the field. (default: 128)              │
│ --pipeline.model.contraction-type {AABB,UN_BOUNDED_TANH,UN_BOUNDED_SPHERE} │
│     Resolution of the grid used for the field. (default:                   │
│     UN_BOUNDED_SPHERE)                                                     │
│ --pipeline.model.cone-angle FLOAT                                          │
│     Should be set to 0.0 for blender scenes but 1./256 for real scenes.    │
│     (default: 0.004)                                                       │
│ --pipeline.model.render-step-size FLOAT                                    │
│     Minimum step size for rendering. (default: 0.01)                       │
│ --pipeline.model.near-plane FLOAT                                          │
│     How far along ray to start sampling. (default: 0.05)                   │
│ --pipeline.model.far-plane FLOAT                                           │
│     How far along ray to stop sampling. (default: 1000.0)                  │
│ --pipeline.model.use-appearance-embedding {True,False}                     │
│     Whether to use an appearance embedding. (default: False)               │
│ --pipeline.model.randomize-background {True,False}                         │
│     Whether to randomize the background color. (default: True)             │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.model.loss-coefficients arguments ───────────────────────────────╮
│ parameters to instantiate density field with                               │
│ ────────────────────────────────────────────────────────────────────────── │
│ --pipeline.model.loss-coefficients.rgb-loss-coarse FLOAT                   │
│     (default: 1.0)                                                         │
│ --pipeline.model.loss-coefficients.rgb-loss-fine FLOAT                     │
│     (default: 1.0)                                                         │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline arguments ───────────────────────────────────────────────────────╮
│ Pipeline configuration                                                     │
│ ────────────────────────────────────────────────────────────────────────── │
│ --pipeline.target-num-samples INT                                          │
│     The target number of samples to use for an entire batch of rays.       │
│     (default: 262144)                                                      │
│ --pipeline.max-num-samples-per-ray INT                                     │
│     The maximum number of samples to be placed along a ray. (default:      │
│     1024)                                                                  │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.fields.optimizer arguments ────────────────────────────────────╮
│ Basic optimizer config with Adam                                           │
│ ────────────────────────────────────────────────────────────────────────── │
│ --optimizers.fields.optimizer.lr FLOAT                                     │
│     (default: 0.01)                                                        │
│ --optimizers.fields.optimizer.eps FLOAT                                    │
│     (default: 1e-15)                                                       │
│ --optimizers.fields.optimizer.weight-decay FLOAT                           │
│     (default: 0)                                                           │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.fields arguments ──────────────────────────────────────────────╮
│ --optimizers.fields.scheduler {None}                                       │
│     (default: None)                                                        │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ optional subcommands ─────────────────────────────────────────────────────╮
│ Specifies the dataparser used to unpack the data.  (default:               │
│ nerfstudio-data)                                                           │
│ ────────────────────────────────────────────────────────────────────────── │
│ [{nerfstudio-data,blender-data,friends-data,instant-ngp-data,nuscenes-dat… │
│     nerfstudio-data                                                        │
│     blender-data                                                           │
│     friends-data                                                           │
│     instant-ngp-data                                                       │
│     nuscenes-data                                                          │
│     record3d-data                                                          │
│     dnerf-data                                                             │
│     phototourism-data                                                      │
│     scannet-data                                                           │
╰────────────────────────────────────────────────────────────────────────────╯
