import tf_keras as keras
from prep_data import (train_ds, val_ds, test_ds)

# 1. Base MobileNetV2
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze base_model initially
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)

nested_model = keras.Model(inputs=inputs, outputs=outputs)

print("\n========================================")
print("Stage 1 — Training classification head")
print("========================================\n")

nested_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

nested_model.fit(
    train_ds, 
    validation_data=val_ds,
    epochs=5
)

print("\n========================================")
print("Stage 2 — Fine-tuning")
print("========================================\n")

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

nested_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

nested_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

print("\n========================================")
print("Unrolling model structure and unfreezing all layers")
print("========================================\n")

# Unfreeze all layers
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = True

x = inputs
x = base_model.call(x)

for layer in nested_model.layers[2:]:
    layer.trainable = True
    x = layer(x)

flat_model = keras.Model(inputs=inputs, outputs=x)

flat_model.set_weights(nested_model.get_weights())

print("\n========================================")
print("Evaluation")
print("========================================\n")

flat_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

test_loss, test_accuracy = flat_model.evaluate(test_ds, verbose=1)
print(f"Test accuracy: {test_accuracy:.4%}")

flat_model.save("models/MobileNetV2.h5")
print("Model saved to models/MobileNetV2.h5")