import tensorflow as tf
from prep_data import train_ds, val_ds, test_ds
import time
import numpy as np
import os
import math

model = tf.keras.models.load_model("models/MobileNetV2.h5")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#train_steps and val_steps because if I don't specify it the fitting will run forever because of .repeat() in the dataset
train_steps = math.ceil(2936 / 32) #(80% of 3670 images) // batch size
val_steps = math.ceil(367 / 32) #(10% of 3670 images) // batch size
test_steps = math.ceil(367 / 32) #(10% of 3670 images) // batch size

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=train_steps,
    validation_steps=val_steps
)

for layer in model.layers[-20:]:
    layer.trainable = True #unfreezing last 20 layers

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), #use this loss because we have 5 sets: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
  metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    steps_per_epoch=train_steps,
    validation_steps=val_steps
)

results = model.evaluate(test_ds, steps=test_steps)
print(f"Results: final loss {results[0]} | final accuracy {results[1]}")

model.save("models/MobileNetV2_fitted.h5")

# Size calculation
model_size = os.path.getsize("models/MobileNetV2_fitted.h5") / (1024 * 1024) # size in MB
print(f"Model size: {model_size:.2f} MB")

# Latency calculation
input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
model.predict(input_data) #warm up the model

start_time = time.time()
prediction = model.predict(input_data)
end_time = time.time()

latency = end_time - start_time
print(f"Latency: {latency:.2f} seconds")