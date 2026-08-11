import tensorflow as tf
import numpy as np

def convert_to_fp16(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    return converter.convert()

'''
Representative dataset is needed for calibration because INT8 has significantly smaller intervals
'''
def create_representative_dataset(train_ds, num_samples=100):
    samples = []

    for images, _ in train_ds:
        for image in images:
            samples.append(
                np.expand_dims(image.numpy(), axis=0)
            )

            if len(samples) >= num_samples:
                break

        if len(samples) >= num_samples:
            break

    def representative_dataset():
        for sample in samples:
            yield [sample.astype(np.float32)]

    return representative_dataset

def convert_to_int8(model, representative_dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()