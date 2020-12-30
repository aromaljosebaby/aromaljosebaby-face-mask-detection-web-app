from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import numpy as np
import streamlit as st
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def get_model():

    model = tf.saved_model.load('saved_model/face_mask_model_with_no_data_aug_on_resnet')
    return model


def predict(image):
    loaded_model = get_model()

    prediction_dictionary = {0: 'with_mask', 1: 'without_mask'}
    colour_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

    ans = loaded_model(image, training=False)

    label = np.argmax(ans)
    final_prediction = prediction_dictionary[np.argmax(ans)]

    return label,final_prediction











