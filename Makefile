.PHONY: install test lint format scan-data preprocess-debug preprocess-gas run-fastsurfer segment-debug plot-etco2-qc fit-real-debug pool-debug simulate-debug train-debug evaluate-debug infer-real-debug

install:
	python -m pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

scan-data:
	hybrid-cvr scan-data --bids-root data/raw --out data/processed/manifest.csv

preprocess-debug:
	hybrid-cvr preprocess-real --config configs/preprocessing.yaml --manifest data/processed/manifest.csv

preprocess-gas:
	hybrid-cvr preprocess-gas --config configs/preprocessing.yaml --manifest data/processed/manifest.csv --processed-root data/processed

run-fastsurfer:
	hybrid-cvr run-fastsurfer --config configs/segmentation.yaml --manifest data/processed/manifest.csv --out data/derivatives/fastsurfer/fastsurfer_summary.csv

segment-debug:
	hybrid-cvr segment-real --config configs/segmentation.yaml --manifest data/processed/manifest.csv

plot-etco2-qc:
	hybrid-cvr plot-etco2-qc --manifest data/processed/manifest.csv --processed-root data/processed

fit-real-debug:
	hybrid-cvr fit-real-cvr --config configs/real_cvr_fit.yaml --manifest data/processed/manifest.csv

pool-debug:
	hybrid-cvr pool-distributions --config configs/simulation.yaml --real-fit-dir data/derivatives/real_cvr --seg-dir data/derivatives/segmentation --out data/distributions/tissue_parameter_samples.parquet

simulate-debug:
	hybrid-cvr simulate --config configs/simulation.yaml --out data/simulated

train-debug:
	hybrid-cvr train --config configs/train_unet_pinn.yaml --sim-root data/simulated --out data/models/unet_pinn

evaluate-debug:
	hybrid-cvr evaluate --checkpoint data/models/unet_pinn/best.pt --sim-root data/simulated/test --out data/derivatives/evaluation

infer-real-debug:
	hybrid-cvr infer-real --checkpoint data/models/unet_pinn/best.pt --config configs/train_unet_pinn.yaml --subject sub-01 --session ses-01 --bids-root data/raw --out data/derivatives/real_inference
