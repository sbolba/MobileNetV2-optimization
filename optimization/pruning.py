import tensorflow_model_optimization as tfmot
import tf_keras as keras
import numpy as np

'''
This module uses tf_keras instead of tf.keras because the
TensorFlow Model Optimization Toolkit version used by this
project is not compatible with the standalone Keras 3 API.
'''

def _apply_pruning_to_layer(layer, **pruning_kwargs):
    """
    Applies prune_low_magnitude to layers containing learnable weight matrices.
    """
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D, keras.layers.Dense)):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_kwargs)
    return layer

def make_full_model_prunable(model, **pruning_kwargs):
    """
    Ensures all layers are trainable and applies pruning wrappers 
    across all eligible layers in the unrolled model.
    """
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True

    return keras.models.clone_model(
        model,
        clone_function=lambda l: _apply_pruning_to_layer(l, **pruning_kwargs)
    )

def unstructured_pruning(model, train_ds, val_ds, sparsity, steps_per_epoch, epochs=3):
    """
    Performs unstructured magnitude-based pruning across the entire unrolled architecture.
    """
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=sparsity,
        begin_step=0,
        end_step=steps_per_epoch * epochs,
    )

    pruning_model = make_full_model_prunable(model, pruning_schedule=pruning_schedule)

    pruning_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    pruning_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=1,
    )

    # Calculate actual global sparsity
    total = 0
    zeros = 0
    for layer in pruning_model.layers:
        if hasattr(layer, "pruning_vars"):
            for weight, mask, threshold in layer.pruning_vars:
                mask_np = mask.numpy()
                total += mask_np.size
                zeros += np.count_nonzero(mask_np == 0)

    actual_sparsity = zeros / total if total > 0 else 0.0
    print(f"\nOverall model sparsity achieved: {actual_sparsity:.2%}\n")

    stripped_model = tfmot.sparsity.keras.strip_pruning(pruning_model)
    return stripped_model, actual_sparsity

def structured_pruning(model, train_ds, val_ds, steps_per_epoch, epochs=3):
    """
    Applies 2:4 structured sparsity across all eligible layers in the network.
    """
    pruning_params = {
        "sparsity_m_by_n": (2, 4),
    }

    pruning_model = make_full_model_prunable(model, **pruning_params)

    pruning_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    pruning_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=1,
    )

    total = 0
    zeros = 0
    for layer in pruning_model.layers:
        if hasattr(layer, "pruning_vars"):
            for weight, mask, threshold in layer.pruning_vars:
                mask_np = mask.numpy()
                total += mask_np.size
                zeros += np.count_nonzero(mask_np == 0)

    actual_sparsity = zeros / total if total > 0 else 0.0
    print(f"\nOverall 2:4 structured sparsity achieved: {actual_sparsity:.2%}\n")

    stripped_model = tfmot.sparsity.keras.strip_pruning(pruning_model)
    return stripped_model, actual_sparsity