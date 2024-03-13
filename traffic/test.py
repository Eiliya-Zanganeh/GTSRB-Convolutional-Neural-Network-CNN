import tensorflow as tf
import numpy as np


def test(path) -> str:
    model = tf.keras.models.load_model('../Model')
    img = tf.keras.preprocessing.image.load_img(path, target_size=(30, 30))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    predictions = model.predict(img)
    predictions = np.argmax(predictions, axis=-1)
    return predictions

print(test('../img.jpg'))