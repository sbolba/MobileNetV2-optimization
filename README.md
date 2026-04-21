# MobileNetV2-optimization

This project has a simple base idea: make possible to run on a mobile or IoT device big models (i. e. MobileNetV2)

For this purpose I will use two techniques: pruning + quantization

# Pruning

Pruning is basically eliminating all the values in the weight matrix near to zero.
It consists in creating a binary mask with zeros and ones that we want to multiply with the weight matrix.

Structured pruning is the method I will use, because it's optimized for OS calculations.

The last part of pruning would be deleting all the zeros from the matrix, but due I wanna add quantization I will do this step only at the end of the project for better accuration.

Compriming a pruned model would create a sparse model and it would already be super fast because zeros are very well-compressed.

Model Wrap -> Sceduling -> Pruning-Aware Training

# Quantization

Quantization means take all the values and converting them all in "lighter" values.
Doing that, you can free a lot of memory and the model could run on smaller device too.
(i. e. [float32 -> int8]: 75% memory saved)

# PCQ

1) Pruning
2) While the model does still have his binary masks simulate the 8 bit low precision during training (fitting)
3) Export final model