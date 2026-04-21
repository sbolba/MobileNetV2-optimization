from keras.applications import MobileNetV2
import tensorflow as tf

model = MobileNetV2()

#compiling (this loss and optimizer are suggested in the dataset url)
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), #use this loss because we have 5 sets: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
  metrics=['accuracy'])

print(model.summary(expand_nested=True, show_trainable=True))
print("\n")
weights = model.layers[1].get_weights()
print(weights[0].dtype)

model.save('MobileNetV2.keras')