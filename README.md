# MobileNetV2 Optimization for Mobile and IoT Deployment

This project optimizes MobileNetV2 for inference on smaller devices using pruning and quantization.

The goal is to keep strong image-classification performance while reducing model size and improving deployment efficiency on resource-constrained hardware.

The goal of this project is to see the differences between the only fitted model and the optimized model.

## Method

**Pruning** removes weights near zero via a binary mask. Structured pruning is used for efficient sparse-matrix operations on mobile processors. Pruning is applied before quantization to maximize compression.
(prune_magnitude_low (with pruning_settings) -> compile -> fit (with UpdatePruningSteps() as callback) -> strip_pruning)

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

`test_model.py` loads `MobileNetV2_fitted.h5`, preprocesses a sample image from the dataset, and prints predicted class plus confidence.

## Usage

```bash
python model_download.py   # Download and compile base model
python prep_data.py      # Prepare dataset
python prefitting.py    # Fine-tune and evaluate
python test_model.py    # Optional, run single-image inference sanity check
```

**Note:**
Pruning with `tensorflow-model-optimization` is not compatible with Keras 3. If you need pruning, use a separate legacy TF 2.13 environment and generate the base and fitted models in that same environment before running pruning.

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
| Test accuracy | 91.88% |
| Model size | 18.98 MB |
| Latency | 0.06 s |

*Optimized Model*
| Metric | Value |
|--------|-------|
| Test accuracy | — |
| Model size | — |
| Latency | — |

## Requirements
**Only fitted model**
- TensorFlow
- TensorFlow Datasets
- NumPy
```bash
pip install tensorflow tensorflow-datasets numpy
```

**Fitted model + pruning + quantization**
- Python 3.8-3.11 (64-bit)
- TensorFlow 2.13
- tensorflow_model_optimization
- protobuf<4.21
- Tensorflow Datasets
- NumPy
```bash
pip install "tensorflow==2.13.*" "tensorflow-model-optimization==0.7.5" "protobuf<4.21" tensorflow-datasets numpy
```