import pathlib
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Configuration
DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/flower_photos.tgz"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

# Dataset download
data_dir = tf.keras.utils.get_file( # Save them in C/Users/USER/.keras/datasets/
    origin=DATASET_URL,
    fname="flower_photos",
    untar=True
)

data_dir = pathlib.Path(data_dir) / "flower_photos"


# Load the complete dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=IMAGE_SIZE,
    batch_size=None,
    shuffle=True,
    seed=SEED
)

class_names = full_dataset.class_names

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"  {index}: {class_name}")

dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
print(f"\nTotal images: {dataset_size}")

# Split sizes
train_size = int(dataset_size * TRAIN_SPLIT)
val_size = int(dataset_size * VAL_SPLIT)
test_size = dataset_size - train_size - val_size

print(f"Train images: {train_size}")
print(f"Validation images: {val_size}")
print(f"Test images: {test_size}")

# Split dataset
train_ds = full_dataset.take(train_size)
remaining_ds = full_dataset.skip(train_size)
val_ds = remaining_ds.take(val_size)
test_ds = remaining_ds.skip(val_size)

# MobileNetV2 preprocessing
def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels


train_ds = train_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Batch
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

# Performance
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)