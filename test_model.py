from prep_data import class_names, daisy, dandelion, roses, sunflowers, tulips
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import numpy as np

def process_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224)) 
    img_array = tf.keras.utils.img_to_array(img) # (224, 224, 3)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0) # add batch dimension: (1, 224, 224, 3)
    return img_array

model = load_model('models/MobileNetV2_fitted.keras')

# you can choose whatever image you want
sunflowers_image = sunflowers[0]
img = process_image(sunflowers_image)

result = model.predict(img)

predicted = np.argmax(result, axis=1)
confidence = np.max(result, axis=1)

print(f"Image predicted was {class_names[predicted[0]]}")
print(f"The confidence was {confidence[0]:.2%}")