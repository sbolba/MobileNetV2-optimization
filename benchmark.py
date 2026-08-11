from pathlib import Path
import json
import csv
import os
import time
import math

import numpy as np
import tensorflow as tf
import tf_keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from prep_data import train_ds, val_ds, test_ds, train_size
from optimization.pruning import (
    unstructured_pruning,
    structured_pruning,
)

from optimization.quant import (
    convert_to_fp16,
    convert_to_int8,
    create_representative_dataset,
)

BASELINE_MODEL_PATH = "models/MobileNetV2.h5"
RESULTS_DIR = Path("results")

BATCH_SIZE = 32
WARMUP_RUNS = 20
BENCHMARK_RUNS = 100

PRUNING_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

#Utility benchmark functions
def get_model_size_mb(model): #Save the model temporarily and get his file size.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    temporary_path = os.path.join(RESULTS_DIR, "_temporary_model.keras")
    model.save(temporary_path)

    size_bytes = os.path.getsize(temporary_path)

    os.remove(temporary_path)

    return {
        "model_size_bytes": int(size_bytes),
        "model_size_mb": float(
            size_bytes / (1024 ** 2)
        ),
    }

def count_params(model): #Trainable and non-trainable parameters
    trainable = np.sum([
        np.prod(variable.shape) 
        for variable in model.trainable_variables
    ])

    non_trainable = np.sum([
        np.prod(variable.shape)
        for variable in model.non_trainable_variables
    ])

    return {
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(non_trainable),
        "total_parameters": int(
            trainable + non_trainable
        ),
    }

def calc_metrics(model, dataset):
    #Calculate accuracy, precision, recall and F1 score
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predictions = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_test_samples": int(len(y_true)),
    }

def measure_latency(model, dataset, warmup_runs=WARMUP_RUNS, benchmark_runs=BENCHMARK_RUNS):
    """
    - warm-up runs
    - 100 measured runs
    - mean
    - standard deviation
    - p50
    - p95
    """

    images, _ = next(iter(dataset))

    #Warm-up
    for _ in range(warmup_runs):
        model(images, training=False)

    #Benchmark
    latencies = []

    for _ in range(benchmark_runs):
        start = time.perf_counter()
        model(images, training=False)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        latencies.append(latency_ms)

    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }

def measure_throughput(model, dataset, benchmark_runs=BENCHMARK_RUNS):
    images, _ = next(iter(dataset))

    batch_size = images.shape[0]
    
    #Warm-up
    for _ in range(WARMUP_RUNS):
        model(images, training=False)

    start = time.perf_counter()
    for _ in range(benchmark_runs):
        model(images, training=False)
    end = time.perf_counter()

    total_time = end-start
    total_images = batch_size * benchmark_runs

    return float(total_images / total_time)

'''
now the tflite functions !
'''

def get_tflite_size_mb(model_bytes):
    size_bytes = len(model_bytes)

    return {
        "model_size_bytes": int(size_bytes),
        "model_size_mb": float(size_bytes / (1024 ** 2)),
    }


def create_tflite_interpreter(model_bytes):
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    return interpreter

#Private util functions
def prepare_tflite_input(image, input_details):
    input_dtype = input_details[0]["dtype"]

    if input_dtype in (np.int8, np.uint8):
        input_scale, input_zero_point = input_details[0]["quantization"]
        image = (image / input_scale + input_zero_point).round().astype(input_dtype)
    else:
        image = image.astype(input_dtype)

    return image

def prepare_tflite_output(output, output_details):
    output_dtype = output_details[0]["dtype"]

    if output_dtype in (np.int8, np.uint8):
        output_scale, output_zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - output_zero_point) * output_scale

    return output

def count_tflite_params(interpreter):
    total_params = 0
    for detail in interpreter.get_tensor_details():
        shape = detail['shape']
        if len(shape) > 0:
            total_params += int(np.prod(shape))
    return total_params

def calc_tflite_metrics(interpreter, dataset):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    y_true = []
    y_pred = []

    for images, labels in dataset:
        for image, label in zip(images, labels):
            image = np.expand_dims(image.numpy(), axis=0)

            image = prepare_tflite_input(image, input_details)

            interpreter.set_tensor(
                input_index,
                image
            )

            interpreter.invoke()

            output = interpreter.get_tensor(
                output_index
            )

            output = prepare_tflite_output(output, output_details)

            prediction = np.argmax(output, axis=1)[0]

            y_true.append(int(label.numpy()))
            y_pred.append(int(prediction))

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "num_test_samples": int(
            len(y_true)
        ),
    }


def measure_tflite_latency(interpreter, dataset, warmup_runs=WARMUP_RUNS, benchmark_runs=BENCHMARK_RUNS,):
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]

    images, _ = next(iter(dataset))

    image = images[0].numpy()
    image = np.expand_dims(image, axis=0)

    image = prepare_tflite_input(image, input_details)

    interpreter.set_tensor(
        input_index,
        image
    )

    # Warm-up
    for _ in range(warmup_runs):
        interpreter.invoke()

    latencies = []

    for _ in range(benchmark_runs):
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()

        latencies.append((end - start) * 1000.0)

    return {
        "latency_mean_ms": float(
            np.mean(latencies)
        ),
        "latency_std_ms": float(
            np.std(latencies)
        ),
        "latency_p50_ms": float(
            np.percentile(latencies, 50)
        ),
        "latency_p95_ms": float(
            np.percentile(latencies, 95)
        ),
    }

def measure_tflite_throughput(interpreter, dataset, benchmark_runs=BENCHMARK_RUNS,):
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]

    images, _ = next(iter(dataset))

    image = images[0].numpy()
    image = np.expand_dims(image, axis=0)

    image = prepare_tflite_input(image, input_details)

    interpreter.set_tensor(
        input_index,
        image
    )

    # Warm-up
    for _ in range(WARMUP_RUNS):
        interpreter.invoke()

    start = time.perf_counter()

    for _ in range(benchmark_runs):
        interpreter.invoke()

    end = time.perf_counter()

    total_time = end - start

    return float(benchmark_runs / total_time)

def run_benchmark(model, name, sparsity=None):
    print("\n" + "=" * 60)
    print(f"Benchmark: {name}")
    print("=" * 60)

    classification = calc_metrics(model, test_ds)
    parameters = count_params(model)
    size = get_model_size_mb(model)
    latency = measure_latency(model, test_ds)
    throughput = measure_throughput(model, test_ds)

    results = {
        "model": name,
        "sparsity": sparsity,
        **classification,
        **parameters,
        **size,
        **latency,
        "throughput_images_per_second": throughput,
    }

    print("\nResults:")
    for key, value in results.items():
        print(f"{key}: {value}")

    return results

def run_tflite_benchmark(model_bytes, name, sparsity=None):
    print("\n" + "=" * 60)
    print(f"Benchmark: {name}")
    print("=" * 60)

    interpreter = create_tflite_interpreter(model_bytes)

    classification = calc_tflite_metrics(interpreter, test_ds)

    size = get_tflite_size_mb(model_bytes)

    latency = measure_tflite_latency(interpreter, test_ds)

    throughput = measure_tflite_throughput(interpreter, test_ds)

    results = {
        "model": name,
        "sparsity": sparsity,
        **classification,
        "trainable_parameters": None,
        "non_trainable_parameters": None,
        "total_parameters": count_tflite_params(interpreter),
        **size,
        **latency,
        "throughput_images_per_second": throughput,
    }

    print("\nResults:")

    for key, value in results.items():
        print(f"{key}: {value}")

    return results

def save(all_results, dir_path):
    json_path = dir_path / "json_results.json"
    with open(json_path, "w") as file:
            json.dump(all_results, file, indent=4)

    csv_path = dir_path / "experiments.csv"
    fieldnames = [
        "model",
        "sparsity",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "num_test_samples",
        "trainable_parameters",
        "non_trainable_parameters",
        "total_parameters",
        "model_size_bytes",
        "model_size_mb",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "throughput_images_per_second",
    ]
    
    with  open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    #Baseline
    print("\n" + "=" * 60)
    print("Starting baseline model benchmark")
    print("=" * 60)
    baseline_model = tf_keras.models.load_model(BASELINE_MODEL_PATH)
    baseline_result = run_benchmark(baseline_model, "baseline")
    all_results.append(baseline_result)
    del baseline_model

    #Pruning experiments
    for sparsity in PRUNING_LEVELS:
        print("\n" + "=" * 60)
        print(f"Starting pruning experiments | sparsity: {sparsity}")
        print("=" * 60)

        #Reload baseline model
        model = tf_keras.models.load_model(BASELINE_MODEL_PATH)
        steps_per_epoch = math.ceil(train_size/BATCH_SIZE)
        pruned_model, actual_sparsity = unstructured_pruning(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            sparsity=sparsity,
            steps_per_epoch=steps_per_epoch,
            epochs=3
        )
        print(f"Requested sparsity: {sparsity:.0%} | Actual sparsity: {actual_sparsity:.2%}")
        result = run_benchmark(pruned_model, f"pruning_{sparsity:.0%}", actual_sparsity)
        all_results.append(result)

        del model
        del pruned_model

    #Structured pruning
    print("\n" + "=" * 60)
    print("Starting structured pruning experiment")
    print("=" * 60)

    model = tf_keras.models.load_model(BASELINE_MODEL_PATH)
    steps_per_epoch = math.ceil(train_size / BATCH_SIZE)
    structured_model, actual_sparsity = structured_pruning(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=3
    )
    print(f"2:4 sparsity | Actual sparsity: {actual_sparsity:.2%}")
    result = run_benchmark(structured_model, "structured_pruning_2_4", actual_sparsity)
    all_results.append(result)

    del model
    del structured_model

    # FP16 Quantization
    print("\n" + "=" * 60)
    print("Starting FP16 quantization experiment")
    print("=" * 60)

    model = tf_keras.models.load_model(BASELINE_MODEL_PATH)
    fp16_model = convert_to_fp16(model)
    result = run_tflite_benchmark(fp16_model, "fp16")
    all_results.append(result)

    del model
    del fp16_model

    # INT8 Quantization
    print("\n" + "=" * 60)
    print("Starting INT8 quantization experiment")
    print("=" * 60)

    model = tf_keras.models.load_model(BASELINE_MODEL_PATH)
    representative_dataset = create_representative_dataset(train_ds=train_ds, num_samples=100)
    int8_model = convert_to_int8(model, representative_dataset)
    result = run_tflite_benchmark(int8_model, "int8")
    all_results.append(result)

    del model
    del int8_model

    save(all_results=all_results, dir_path=RESULTS_DIR)

if __name__ == "__main__":
    main()