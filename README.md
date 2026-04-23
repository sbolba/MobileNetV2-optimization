# MobileNetV2 Optimization for Mobile and IoT Deployment

This project optimizes MobileNetV2 for inference on smaller devices using pruning and quantization.

The goal is to keep strong image-classification performance while reducing model size and improving deployment efficiency on resource-constrained hardware.

## Method

**Pruning** removes weights near zero via a binary mask. Structured pruning is used for efficient sparse-matrix operations on mobile processors. Pruning is applied before quantization to maximize compression.

**Quantization** reduces weight precision (e.g., float32 → int8), cutting memory usage by ~75% while preserving accuracy.

## Dataset

[tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) — 5 classes: daisy, dandelion, roses, sunflowers, tulips. Images resized to 224×224.

## Model

MobileNetV2 base with a custom classification head (GlobalAveragePooling2D → Dense(128,relu) → Dense(5,softmax)).

Transfer learning workflow:
1. Freeze base, train head (5 epochs)
2. Unfreeze last 20 layers, fine-tune end-to-end (10 epochs, lr=1e-5)

It applies patterns from the initial fitting to optimize new training sessions.

## Project Workflow

Run scripts in this order:
1. `python model_download.py`
2. `python prep_data.py`
3. `python prefitting.py`
4. `python test_model.py`

`test_model.py` loads `MobileNetV2_fitted.keras`, preprocesses a sample image from the dataset, and prints predicted class plus confidence.

## Usage

```bash
python model_download.py   # Download and compile base model
python prep_data.py      # Prepare dataset
python prefitting.py    # Fine-tune and evaluate
python test_model.py    # Run single-image inference sanity check
```

## Setup

```bash
pip install tensorflow tensorflow-datasets numpy
```

Optional (recommended): use a virtual environment before installing dependencies.

## Files

| File | Description |
|------|-------------|
| `model_download.py` | Load MobileNetV2 + compile |
| `prep_data.py` | Download and split dataset |
| `prefitting.py` | Fine-tune on training data |
| `test_model.py` | Run inference on one image and print class/confidence |
| `pruning.py` | Pruning stage for model compression pipeline |
| `quant.py` | Quantization stage for deployment optimization |

## Results

*Base Model*
| Metric | Value |
|--------|-------|
| Test accuracy | 92.15% |
| Model size | 19.02 MB |
| Latency | 0.07 s |

*Optimized Model*
| Metric | Value |
|--------|-------|
| Test accuracy | — |
| Model size | — |
| Latency | — |

## Requirements

- TensorFlow
- TensorFlow Datasets
- NumPy