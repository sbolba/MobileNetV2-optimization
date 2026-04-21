import tensorflow as tf
from tensorflow.keras.models import load_model
from prep_data import train_ds, val_ds

model = load_model('MobileNetV2.keras')

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)
