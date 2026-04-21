from keras.applications import MobileNetV2

model = MobileNetV2()

#compiling
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision'])

print(model.summary(expand_nested=True, show_trainable=True))
print("\n")
weights = model.layers[1].get_weights()
print(weights[0].dtype)

model.save('MobileNetV2.keras')