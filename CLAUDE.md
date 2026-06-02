# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`py3dcal` is a PyPI package for calibrating tactile sensors by automatically probing them with a 3D printer, then training neural networks to convert raw sensor output into depth/contact maps. It supports two families of sensors:

- **Vision-based** (DIGIT, GelSight Mini): camera images → surface gradient maps → depth maps, via the `TouchNet` CNN.
- **Magnetic** (ReSkin): magnetometer readings → contact predictions, via the `MagNet` MLP.

There is no test suite, no linter config, and no build step beyond `setup.py`. Users normally `pip install py3dcal`; for local development install editable:

```bash
pip install -e .
```

The only console entry point is `list-com-ports` (maps to `py3DCal.list_com_ports`), used to find the serial port of an attached printer.

Project was tested with Python 3.10.4. Full usage docs live at https://rohankotanu.github.io/3DCal/ — that site, not the README, is the source of truth for the user-facing API.

## The end-to-end pipeline

`examples/full_pipeline.py` shows the canonical flow. The package is built around four sequential stages, each surfaced as a top-level function re-exported in `py3DCal/__init__.py`:

1. **Data collection** — `Calibrator(printer, sensor).probe(...)` (vision) or `.probe_reskin(...)` (magnetic). Drives the printer to each calibration point, presses the probe into the gel, and records sensor output. Writes a `sensor_calibration_data/` directory.
2. **Annotation** — `annotate(dataset_path, probe_radius_mm)`. An interactive Matplotlib GUI (keys `w/a/s/d` move, `r/f` resize, `q` advances, `1/2/3` switch views) where the user fits circles to two probe images to compute a `px_per_mm` scale. Produces `annotations/annotations.csv` and `annotations/metadata.json`. Vision pipeline only.
3. **Training** — `train_model(model, dataset, ...)`. Splits the dataset, runs the training loop, and writes `weights.pth` and `losses.csv` to the current working directory.
4. **Inference** — `get_depthmap` / `save_2d_depthmap` / `show_2d_depthmap` (vision) or `get_reskin_contact` (magnetic).

## Architecture and key abstractions

### Hardware drivers use abstract base classes

`data_collection/printers/Printer.py` and `data_collection/sensors/Sensor.py` are ABCs. To add hardware, subclass them:

- A **Printer** must implement `connect`, `disconnect`, `send_gcode`, `get_response`, `initialize`. The base class provides `go_to(x, y, z)`. `Ender3` is the only concrete printer; it talks G-code over `pyserial` at 115200 baud and homes by waiting for four `ok` responses.
- A **Sensor** must implement `connect`, `disconnect`, `capture_image`. The base provides `flush_frames`. Each sensor sets calibration geometry as instance attributes: `x_offset`, `y_offset`, `z_offset` (the sensor surface height), `z_clearance`, `max_penetration`, and `default_calibration_file`. `Calibrator` reads these attributes to plan printer moves — getting them wrong drives the probe into the gel too hard or misses it.

Concrete sensors live in `data_collection/sensors/<Name>/` alongside a `default.csv` of calibration points. `capture_image()` returns different things per sensor family: vision sensors return an RGB `numpy` image (DIGIT flips its frame horizontally via `cv2.flip`); ReSkin returns a flat list of magnetometer channels (`Bx/By/Bz/T` × 5). Hardware-import guards (e.g. `from digit_interface import Digit` wrapped in `try/except`) let the package import on machines without the hardware libraries installed.

### Calibration data format

`probe()` creates this layout under `data_save_path/sensor_calibration_data/`:

```
annotations/probe_data.csv     # img_name, x_mm, y_mm, penetration_depth_mm
blank_images/blank.png         # reference image with no contact
probe_images/                  # one PNG per probe, named <idx>_X<x>Y<y>Z<z>.png
```

Calibration point CSVs (the `default.csv` files and any custom file passed as `calibration_file_path`) have columns `x_mm, y_mm, penetration_depth_mm, num_images` and a header row that is skipped. Points whose depth exceeds the sensor's `max_penetration` are skipped at probe time. `probe_reskin()` instead writes a single flat `probe_data.csv` (+ `no_contact_data.csv`) with the magnetometer channels inline — no images, no separate annotation step.

### Vision model input is a 5-channel tensor

`TouchNet` (`model_training/models/touchnet.py`) is a fully-convolutional 9-layer CNN: input **5 channels**, output **2 channels** (x/y surface gradients). The 5 channels are RGB **plus two coordinate-embedding channels** appended by `add_coordinate_embeddings` (per-pixel column and row indices). This is applied consistently in both training (`TactileSensorDataset`) and inference (`get_depthmap`) — any new path that feeds TouchNet must replicate it, or the channel count won't match.

`get_depthmap` subtracts the blank image, adds coordinate embeddings, runs the model to get gradients, then integrates them into a depth map via `fast_poisson` (Poisson surface reconstruction in `lib/fast_poisson.py`).

Pretrained weights are downloaded on demand from Zenodo when `TouchNet(load_pretrained=True, sensor_type=...)` is used; `SensorType` (DIGIT / GELSIGHTMINI) selects which weight file. Files are cached under `root` and skipped if already present.

### Datasets and training

`model_training/datasets/` has one `Dataset` per sensor: `TactileSensorDataset` (the generic vision dataset, used by DIGIT and GelSightMini subclass datasets) and `ReSkinDataset`. `TactileSensorDataset.__getitem__` returns `(image_with_embeddings, gradient_map_target)`; the gradient map target is generated from the annotated contact circle by `precompute_gradients` / `get_gradient_map`. `subtract_blank` and `add_coordinate_embeddings` default to `True`.

`train_model` enforces the model↔dataset pairing: `TouchNet` requires a `TactileSensorDataset`, `MagNet` requires a `ReSkinDataset` (see `_validate_model_and_dataset`). It always writes outputs to the **current working directory**, not the dataset directory.

### Validation conventions

Argument validation is centralized in `model_training/lib/validate_parameters.py` (`validate_device`, `validate_root`, `validate_dataset`) and called at the top of the public functions. Follow this pattern — validate inputs up front and raise `ValueError`/`TypeError` with an explanatory message rather than failing deep in the call stack.

## Gotchas

- Several modules contain a stray `from pyexpat import model` import (e.g. `depthmaps.py`, `reskin_prediction.py`) — it is unused; don't rely on it or propagate it.
- `Calibrator.disconnect_sensor` and parts of the `probe` flow call `input()` and print to stdout — the collection pipeline is interactive and assumes a human at a terminal with hardware attached. It cannot run unattended in CI.
- The annotation and visualization GUIs require an interactive Matplotlib backend (a display).
- `default_calibration_file` paths are resolved relative to each sensor module's own directory via `os.path.dirname(os.path.abspath(__file__))`; the CSVs are bundled into the package through `MANIFEST.in` (`recursive-include py3DCal/data_collection *.csv`).
