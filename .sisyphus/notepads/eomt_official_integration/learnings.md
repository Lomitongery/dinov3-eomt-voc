# Learnings - EoMT Official Integration

## datasets/ - Base Dataset Files

### Files Created (exact copies from github.com/tue-mps/eomt)
- `datasets/dataset.py` — Base zip-based `Dataset(torch.utils.data.Dataset)` class
  - Reads images/targets from zip files via `zipfile.ZipFile`
  - Supports semantic, instance, and panoptic targets via `target_parser` callable
  - `__getitem__` returns `(img: tv_tensors.Image, target: dict)` with "masks", "labels", "is_crowd"
  - Worker-safe zip loading via `get_worker_info()`
  - Supports nested zips, separate image/target zips, COCO-style JSON annotations
- `datasets/lightning_data_module.py` — Base `LightningDataModule(lightning.LightningDataModule)`
  - Constructor: path, batch_size, num_workers, img_size, num_classes, check_empty_targets, ignore_idx
  - `train_collate()` — stacks images into batch tensor, keeps targets as list of dicts
  - `eval_collate()` — returns `tuple(zip(*batch))`
  - Shared `dataloader_kwargs` dict
- `datasets/transforms.py` — `Transforms(nn.Module)` augmentation pipeline
  - Color jitter (brightness, contrast, saturation, hue) with random ordering
  - Random horizontal flip, scale jitter, pad to img_size, random crop
  - Filters out `is_crowd` masks and empty masks after augmentation
  - Retries with original image if no valid masks remain
  - Uses `tv_tensors` (torchvision >= 0.16)

### Key Observations
- VOCSemantic (next task) will NOT inherit from Dataset (zip-based)
- VOCSemantic WILL inherit from LightningDataModule
- transforms.py expects (img, target_dict) format with tv_tensors
- `lightning` package (not `pytorch_lightning`) is the import used

## configs/ - Configuration Files

### Created: `configs/dinov3/voc/semantic/eomt_small_512.yaml`
- VOC semantic config for EoMT (ViT-Small, 512×512)
- `num_classes: 21` (PASCAL VOC has 21 classes including background)
- `num_q: 100` (semantic uses 100 queries, panoptic uses 200)
- `num_blocks: 4` (default for semantic)
- `backbone_name: "vit_small_patch16_dinov3.lvd1689m"` (ViT-Small DINOv3)
- `masked_attn_enabled: True` (EoMT's masked attention)
- `attn_mask_annealing` with 4 blocks, calculated for 50 epochs at 732 steps/epoch (1464 images / batch_size 2)
  - Block 0: start=0, end=8784 (12 epochs, ~25%)
  - Block 1: start=5856 (8 epochs), end=14640 (20 epochs)
  - Block 2: start=10248 (14 epochs), end=20496 (28 epochs)
  - Block 3: start=14640 (20 epochs), end=26352 (36 epochs)
- No `delta_weights` (training from scratch, no EoMT checkpoint)
- TensorBoard logger (not wandb)
- ModelCheckpoint monitors `metrics/val_iou_all`, saves top 3 + last
- LearningRateMonitor logs at every step
- `precision: 16-mixed`, `devices: 1` (single RTX 5070)
- `data.path` and `data.batch_size` overridable via CLI
- Verified: `python main.py -c configs/dinov3/voc/semantic/eomt_small_512.yaml --print_config` parses correctly

## Code Quality Review (F2)

### Results Summary
- **Syntax**: PASS (all 16 .py files compile cleanly)
- **Hardcoded paths**: NONE found (all paths are configurable via constructor params or CLI)
- **TODO/FIXME/HACK**: NONE found
- **features_only=True**: NONE found
- **wandb imports**: FOUND in `training/lightning_module.py` line 26 — used for logging visualizations
- **File sizes**: All reasonable (max 908 lines in lightning_module.py, rest under 310 lines)
- **Model forward pass**: Shapes verified correct through full trace
- **VERDICT**: APPROVE (with minor note about wandb dependency)

## Documentation - Official Integration Guide

### Created: `docs/04_官方EoMT仓库集成说明.md`
- 815 lines, 5270 Chinese characters, comprehensive technical document
- 8 chapters + appendix covering all aspects of the integration
- Chapter 1: Project background and decision history (original PSPNet -> simplified EoMT -> official repo)
- Chapter 2: Official repo overview (architecture, modules, dependencies, data flow)
- Chapter 3: 8 unchanged official files listed with descriptions
- Chapter 4: 5 modified files with detailed diff explanations (eomt.py RoPE fix is the core change)
- Chapter 5: 2 new custom files (VOCSemantic dataset, YAML config)
- Chapter 6: 6 deleted old files
- Chapter 7: 5 preserved files (docs + __init__.py)
- Chapter 8: Training and evaluation guide with commands and FAQ
- Appendix: Complete file inventory table with statistics

### Key Documentation Decisions
- All code examples use `env -i HOME=$HOME PATH=$PATH` to handle proxy issues
- Training commands include both fast_dev_run and full training variants
- Annealing step calculations are explained in detail with epoch equivalents
- The RoPE fix in _attn() is documented with the full token layout diagram
- VOCSemantic design decisions (no normalization, tv_tensors, empty target handling) are explained
