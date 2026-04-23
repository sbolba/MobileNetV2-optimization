import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir + r'\flower_photos')

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

def preprocess(images, labels):
    #use preprocess_input function from keras.application.MobileNetV2
    #the model expects values between -1 and 1
    images = preprocess_input(images)
    return images, labels

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

card = tf.data.experimental.cardinality(val_ds)

val_ds_full = val_ds  # save it for later, we will split it into val_ds and test_ds

val_ds = val_ds_full.take(card // 2)          # 80-90%
test_ds = val_ds_full.skip(card // 2)          # 90-100%

#cache and prefetch for better performance
#train_ds and train_ds get consumed each epoch, so if we repeat them they will be always available
#I add it on test_ds for consistency, but it won't make a difference because we only evaluate once on it
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE).repeat()
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE).repeat() 
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE).repeat()


roses = list(data_dir.glob('roses/*.jpg'))

daisy = list(data_dir.glob('daisy/*.jpg'))

dandelion = list(data_dir.glob('dandelion/*.jpg'))

sunflowers = list(data_dir.glob('sunflowers/*.jpg'))

tulips = list(data_dir.glob('tulips/*.jpg'))