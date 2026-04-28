import sys
import math
import tensorflow as tf
from prep_data import train_ds, val_ds, test_ds

def _keras_major_version():
    version = getattr(tf.keras, "__version__", "")
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, IndexError):
        return 0

if _keras_major_version() >= 3:
    print("Pruning is not supported with Keras 3. Use a legacy TF/Keras 2 environment with tfmot.")
    sys.exit(0)

try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    print("tensorflow_model_optimization is required for pruning in a legacy TF/Keras 2 environment.")
    sys.exit(0)

# Pruning API
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
strip_pruning = tfmot.sparsity.keras.strip_pruning
UpdatePruningStep = tfmot.sparsity.keras.UpdatePruningStep
PolynomialDecay = tfmot.sparsity.keras.PolynomialDecay

base_model = tf.keras.models.load_model("models/MobileNetV2_fitted.h5")

#train_steps and val_steps because of .repeat() in the dataset
train_steps = math.ceil(2936 / 32) #(80% of 3670 images) // batch size
val_steps = math.ceil(367 / 32) #(10% of 3670 images) // batch size
test_steps = math.ceil(367 / 32) #(10% of 3670 images) // batch size

'''
I tried to spread the pruning across the whole process but I encountered a better stability when I only pruned within the 
first epoch although it's not intuitive so I kept it that way.
'''
pruning_settings = PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,
    begin_step=0,
    end_step=train_steps
)

pruning_model = prune_low_magnitude(base_model, pruning_schedule=pruning_settings)

pruning_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
)

pruning_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[UpdatePruningStep()] #important when pruning
)

pruning_model.evaluate(test_ds, steps=test_steps)

clean_sparse_model = strip_pruning(pruning_model)

clean_sparse_model.save("models/MobileNetV2_pruned.h5")