# Source map

- `opengait/modeling/models/pace_net.py`: PACE-Net and existing control switches.
- `opengait/modeling/phase_modeling.py`: temporal assignments and mass balancing.
- `opengait/modeling/losses/phase.py`: assignment regularization.
- `opengait/data/`: aligned loading, sampling, and transforms.
- `opengait/modeling/backbones/resnet.py`: ResNet-9 encoders.
- `opengait/modeling/base_model.py`, `modules.py`, `loss_aggregator.py`: OpenGait runtime.
- `configs/default.yaml`: base settings merged by the config loader.
- `configs/pace_net/pace_net_scoliosis1k.yaml`: primary configuration.
- `configs/pace_net/dmr_controls/`: historical control configurations.
- `datasets/pretreatment_heatmap.py`: pose-to-map conversion.
- `datasets/Scoliosis1K/Scoliosis1K_118.json`: official sequence-entry split.
- `train_pace_net_single_gpu.py`, `evaluate_pace_net_single_gpu.py`: launchers.
- `live/scripts/`: LIVE preprocessing, cohort selection, models, CV, aggregation.
- `reproducibility/`: existing aggregate reference metrics.
- `tests/`: existing CPU tokenizer, regularization, and sampler tests.

The public `PACENet` class is a naming alias of the original `DRFPlusPhase` implementation. Network computation and checkpoint keys are retained. Only portable data paths and the exported class name differ in the released control configurations.
