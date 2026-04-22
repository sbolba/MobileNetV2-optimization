import tensorflow as tf
from tensorflow.keras.models import load_model
from prep_data import train_ds, val_ds, test_ds
model = load_model('MobileNetV2.keras')

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

'''
for image, label in test_ds.take(1):
    test_pred_im = image
    test_pred_lab = label
    break

model.predict(
    test_pred_im,
    treshold=0.5
)

class_names = train_ds.class_names

print("should be:")
print(class_names[test_pred_lab])
'''

results = model.evaluate(test_ds)
print(f"Results: final loss {results[0]} | final accuracy {results[1]}")