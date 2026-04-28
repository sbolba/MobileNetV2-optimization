import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


#I need a base model, I chose MobileNetV2
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#MobileNetV2 doesn't have the final dense layer with 5 nodes
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
dropout = tf.keras.layers.Dropout(0.2)(x) #adding dropout because I had 1 accuracy and 0.9 val_accuracy, so I was overfitting
outputs = Dense(5, activation='softmax')(dropout)

model = Model(inputs=base.input, outputs=outputs)

#for better fitting I freeze base layers initially (standard transfer learning workflow)
for layer in base.layers:
    layer.trainable = False

#compiling
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(), #use this loss because we have 5 sets: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
  metrics=['accuracy']
)

print(model.summary(expand_nested=True, show_trainable=True))

model.save("models/MobileNetV2.h5") #saving in h5 format for consistent IO across scripts