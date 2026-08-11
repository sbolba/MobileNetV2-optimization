# MobileNetV2 Optimization

An experimental study investigating the **accuracy–efficiency trade-offs of MobileNetV2** through systematic model compression and optimization techniques.

Rather than aiming solely to produce the smallest possible file footprint, this project evaluates how specific optimization methods impact:

* Predictive performance (**Accuracy**, **Precision**, **Recall**, **F1-score**)
* Model complexity (**Total / Trainable parameters**, **Model size**)
* Deployment efficiency (**Inference latency**, **Throughput**)

---

## 1. Environment and Compatibility

This project relies on the **TensorFlow Model Optimization Toolkit (TF-MOT)** for weight pruning and optimization workflows. 

Standard `tensorflow.keras` (Keras 3) lacks native support for legacy graph-level transformation tools like TF-MOT pruning wrappers and specialized export routines. To ensure full compatibility with pruning schedules and post-training quantization pipelines, this project strictly utilizes **`tf_keras`** rather than vanilla `tensorflow.keras`.

**Implementation requirements:**
* `prefitting.py` explicitly constructs, trains, and serializes models using `tf_keras`.
* `benchmark.py` evaluates models strictly via the `tf_keras` runtime.
* Optimization modules interact directly with `tf_keras` layer wrappers.
* Saved models must be created and restored within an identical `tf_keras` environment.

> **Warning:** Mixing standard `tf.keras` with `tf_keras` runtimes will cause deserialization failures or unsupported pruning layer exceptions.

### Tested Environment
```text
Python        3.11
TensorFlow    2.21.0
tf-keras      2.21.0
TF-MOT        0.8.1
scikit-learn  1.7.2
```

### Environment Setup
Create and activate a dedicated virtual environment (using uv or venv):
```bash 
uv venv
```
```bash
source .venv/bin/activate 
```
Install required dependencies directly: 
```bash
pip install tensorflow==2.21.0 tf-keras==2.21.0 tensorflow-model-optimization==0.8.1 scikit-learn==1.7.2
```

**Note**: Avoid installing dependencies globally, as conflicting system-level Keras packages can break TF-MOT pruning masks.

## 2. Motivation

Deploying deep neural networks to edge and resource-constrained hardware requires balancing model capacity against hardware limits.

**MobileNetV2** uses inverted residuals and depthwise separable convolutions specifically designed for lightweight computer vision. This makes it an ideal architecture for testing low-level model compression strategies.

This study addresses a core research question:
   _How much inference efficiency can be gained through MobileNetV2 optimization before predictive accuracy degrades significantly?_

### Evaluated Techniques
- Unstructured Magnitude Pruning: $10\%$ to $50\%$ target sparsity
- Structured Pruning: $2:4$ block pattern sparsity
- Post-Training Quantization (PTQ): FP16 and INT8 precision via TensorFlow Lite

## 3. Core Objectives
1. Establish a reliable, unrolled FP32 baseline.
2. Unroll nested MobileNetV2 layers to allow uniform, layer-wise weight pruning.
3. Construct a standardized, hardware-consistent benchmarking pipeline.
4. Evaluate predictive drop-off across increasing unstructured sparsity levels.
5. Test structured $2:4$ block pruning behavior on general-purpose CPU hardware.
6. Evaluate FP16 and INT8 quantization using the TFLite execution engine.
7. Conduct an ablation study isolating structural vs. precision-driven contributions.
8. Perform a Pareto analysis to identify optimal deployment candidates.

## 4. Experimental Pipeline   
```plaintext
               Original MobileNetV2
                        │
                        ▼
            Model Structure Unrolling
                   (prefitting.py)
                        │
                        ▼
                  FP32 Baseline
                        │
                        ▼
                 Reliable Benchmark
                        │
     ┌──────────────────┼──────────────────┐
     ▼                  ▼                  ▼
Unstructured        Structured        Quantization
  Pruning            Pruning         (TFLite Engine)
 10% - 50%             2:4              ┌────┴────┐
     │                  │               ▼         ▼
     └──────────────────┼────────────► FP16     INT8
                        │
                        ▼
                 Ablation Study
                        │
                        ▼
                 Pareto Analysis
```
## 5. Dataset
Models are trained, fine-tuned, and benchmarked on a 5-class flower image classification dataset (daisy, dandelion, roses, sunflowers, tulips).
### Data Split
- Total Images: $3{,}670$
- Training Set ($80\%$): $2{,}936$ images
- Validation Set ($10\%$): $367$ images
- Test Set ($10\%$): $367$ images

The test partition is strictly held out and remains isolated from training, fine-tuning, pruning, and quantization calibration (representative dataset) steps.

## 6. Model Prefitting & Architecture Unrolling (prefitting.py)
Standard pre-trained implementations (e.g., tf.keras.applications.MobileNetV2) encapsulate the internal backbone into a single nested Functional block. Applying TF-MOT pruning directly to a wrapped model treats the entire backbone as a single layer, preventing weight-masking algorithms from accessing inner depthwise and point-wise convolutions.

To solve this, prefitting.py **unrolls the model architecture**:
1. Iterates through the native MobileNetV2 backbone and extracts every inner layer (Conv2D, DepthwiseConv2D, BatchNormalization, ReLU).
2. Reconstructs a flat, unrolled functional execution graph mapping inputs directly to classification heads.
3. Exposes every weight tensor to TF-MOT, enabling fine-grained, global magnitude pruning across all convolutional layers.

## 7. Baseline Performance
Performance of the unrolled FP32 baseline on an x86_64 CPU:
- Accuracy: $94.55\%$
- F1-Score: $0.9456$
- Model Size: $26.26\text{ MB}$ (.h5 container)
- Mean Latency: $529.59\text{ ms}$
- Throughput: $60.36\text{ images/sec}$ 
## 8. Experimental Results Summary
Benchmarking results collected across all optimization variants on CPU hardware:
| Model Variant       | Sparsity | Accuracy | F1-Score | Model Size | Mean Latency | Throughput    |
|---------------------|----------|----------|----------|------------|--------------|---------------|
| Baseline (FP32)     | 0.00%    | 94.55%   | 0.9456   | 26.26 MB   | 529.59 ms    | 60.36 img/s   |
| Pruning 10%         | 9.79%    | 94.55%   | 0.9457   | 9.13 MB    | 560.79 ms    | 55.67 img/s   |
| Pruning 20%         | 19.58%   | 89.65%   | 0.8979   | 9.13 MB    | 599.22 ms    | 55.06 img/s   |
| Pruning 30%         | 29.37%   | 66.49%   | 0.6504   | 9.13 MB    | 570.95 ms    | 53.43 img/s   |
| Pruning 40%         | 39.16%   | 71.66%   | 0.7186   | 9.13 MB    | 564.16 ms    | 52.66 img/s   |
| Pruning 50%         | 48.96%   | 31.34%   | 0.2571   | 9.13 MB    | 607.17 ms    | 54.63 img/s   |
| Structured (2:4)    | 49.99%   | 17.71%   | 0.0533   | 9.13 MB    | 597.65 ms    | 52.55 img/s   |
| FP16 Quantized      | N/A      | 94.55%   | 0.9456   | 4.27 MB    | 8.07 ms      | 122.49 img/s  |
| INT8 Quantized      | N/A      | 94.82%   | 0.9483   | 2.58 MB    | 10.14 ms     | 98.34 img/s   |
## 9. Ablation Study
### 1. Capacity Constraints under Unstructured Pruning
- **$10\%$ Sparsity**: MobileNetV2 maintains full baseline accuracy ($94.55\%$) with zero loss in performance.
- **$\ge 20\%$ Sparsity**: Accuracy drops rapidly (falling to $31.34\%$ at $50\%$ sparsity). Because MobileNetV2's depthwise separable convolutions are already parameter-efficient, aggressively removing weights severely degrades channel capacity.
### 2. Theoretical Sparsity vs. CPU Hardware Execution
- **File Size vs. Latency**: Pruning reduces serialized file size from $26.26\text{ MB}$ to $9.13\text{ MB}$ (due to zero-value array compression in zip storage), but does not lower real-time CPU latency. Standard CPU execution kernels process zeros as standard floating-point operations, adding sparse indexing overhead that slightly increases latency ($529\text{ ms} \to 560\text{--}600\text{ ms}$).
- **Structured $2:4$ Pruning on CPU**: Without hardware backends designed for sparse acceleration (e.g., NVIDIA Ampere Sparse Tensor Cores), $2:4$ block sparsity acts like standard zeroing on CPU. This causes severe accuracy collapse ($17.71\%$) while failing to deliver speedups ($597.65\text{ ms}$).
### 3. Quantization and TFLite Runtime Efficiency
- **Transitioning to TensorFlow Lite (XNNPACK delegate)** provides massive performance gains over standard Keras CPU execution.
- **FP16 Quantization**: Retains $100\%$ baseline accuracy ($94.55\%$) while reducing storage size by $83.7\%$ ($26.26\text{ MB} \to 4.27\text{ MB}$) and cutting inference latency from $529.59\text{ ms}$ to $8.07\text{ ms}$ ($\sim 65.6\times$ speedup).
- **INT8 Quantization**: Delivers the best overall efficiency. Full integer quantization acts as a slight regularizer, improving accuracy to $94.82\%$, shrinking file size to $2.58\text{ MB}$ ($10.16\times$ compression), and reducing latency to $10.14\text{ ms}$ ($\sim 52.2\times$ speedup).
## 10. Pareto Analysis
Evaluating candidates along the Pareto frontier (balancing high Accuracy against low Model Size and low Latency):
| Modello | Accuratezza | Dimensione | Latenza | Note |
|---------|-------------|------------|---------|------|
| **INT8 Quantized** | **94.82%** | 2.58 MB | 10.14 ms | Miglior compromesso |
| **FP16 Quantized** | 94.55% | 4.27 MB | **8.07 ms** | Più veloce |
| Baseline (FP32) | 94.55% | 26.26 MB | 529.59 ms | Riferimento |
| Pruning 20% | 89.65% | 9.13 MB | 599.22 ms | Calo prestazionale |

```plaintext
Accuracy (%)
   95% |                                    ● INT8 (94.82%, 2.58MB, 10.14ms)
       |                            ● FP16 (94.55%, 4.27MB, 8.07ms)
   94% |    ● FP32 Baseline (94.55%, 26.26MB, 529.59ms)
       |
   90% |               ● Pruned 20% (89.65%)
       |
    0% +------------------------------------------------------------─→
       0 ms                10 ms                 500 ms          Latency
```
```plaintext
Accuracy (%)
  95% │                                                  ● INT8 (94.82%)
      │                                          ● FP16 (94.55%)
  94% │    ● FP32 (94.55%)
      │
  90% │               ● Pruning 20% (89.65%)
      │
  70% │                         ● Pruning 40% (71.66%)
      │
  30% │                                   ● Pruning 50% (31.34%)
      │
  17% │                                            ● Structured (17.71%)
      └─────────────────────────────────────────────────────────────→
        0           2.58          9.13          26.26        Model Size (MB)
```

### Pareto Optimal Models
1. **INT8 Quantized (Best Accuracy & Storage Footprint)**: 
   - Highest accuracy overall ($94.82\%$) and smallest file size ($2.58\text{ MB}$). Ideal for edge environments with tight RAM and storage limits.
2. **FP16 Quantized (Best Latency & Throughput)**:
   - Lowest inference latency ($8.07\text{ ms}$) and highest throughput ($122.49\text{ img/s}$) while matching the exact FP32 baseline accuracy ($94.55\%$).
   
### Non-Pareto / Dominated Models
- **Unstructured & Structured Pruned Variants ($10\%\text{--}50\%$)**: Strictly dominated by INT8 and FP16 quantization. Pruning introduces computational overhead on standard CPUs while causing accuracy to drop beyond minimal sparsity levels.
   
## 11. Repository Structure
```plaintext
MobileNetV2-optimization/
│
├── models/
│   └── MobileNetV2_fitted.h5
│
├── optimization/
│   ├── pruning.py
│   └── quant.py
│
├── results/
│   ├── json_results.json
│   └── experiments.csv
│
├── prep_data.py
├── prefitting.py
├── benchmark.py
│
├── README.md
└── .gitignore
```
## 12. Conclusions
1. **Framework Alignment & Model Unrolling**: Optimization toolkits require full visibility into a network's structure. Using tf_keras and explicitly unrolling nested MobileNetV2 architecture layers were necessary steps to enable global, layer-wise weight pruning.
2. **Architecture Sensitivity**: Compact architectures with depthwise separable convolutions are sensitive to weight removal. Pruning beyond $10\%$ sparsity causes rapid accuracy loss unless paired with extensive fine-tuning or distillation.
3. **Hardware Engine Dependency**: Unstructured and structured weight sparsity without specialized hardware support results in "theoretical efficiency"—reducing non-zero parameter counts on paper without translating to speedups on standard CPU runtimes.
4. **Quantization Dominance**: Post-training quantization paired with optimized runtime delegates (TFLite + XNNPACK) proved to be the most effective optimization path, yielding up to $10\times$ smaller model sizes and $\sim 65\times$ faster inference speeds without loss of accuracy.