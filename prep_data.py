import tensorflow_datasets as tfds
import tensorflow as tf

#import data from tensorflow_datasets for model fitting
(train_ds, test_ds), info = tfds.load(
                                        'tf_flowers', 
                                        split=['train[:80%]', 'train[80%:]'],
                                        with_info=True, 
                                        as_supervised=True
                                    )

'''
or
import tensorflow as tf
import os

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True) #extract the compressed file into a directory and returns the dir name
data_dir = os.path.join(os.path.dirname(data_dir), 'flower_photos') #enter the directory in order to use image_dataset_from_directory

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
'''

def prepare_data(image, label):
    # MobileNetV2 needs 244x244 images with pixels between -1 and 1 (or 0 and 1)
    image = tf.image.resize(image, (244, 244))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(prepare_data).batch(32)
test_ds = test_ds.map(prepare_data).batch(32)

for (image_batch, label_batch) in train_ds.take(1):
    print("image_shape: ", image_batch.shape)
    print("batch_label: ", label_batch.numpy())
    break

