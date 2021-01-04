import streamlit as st
#from PIL import Image
import cv2
#from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
#import tensorflow as tf
#import classify

'''def main():

    html_temp = """
        <div style="background-color:#e63946 ;padding:10px">
        <h2 style="color:white;text-align:center;">Deep Learning Web App</h2>
        </div><br><br>
        """

    st.markdown(html_temp, unsafe_allow_html=True)

    st.title("Mask Detection")

    st.title(" ")



    st.title(" ")
    st.title(" ")

    variable=0


    def load_haarcascade(variable):
        haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return haarcascade

    haarcascade=load_haarcascade(variable)




    rect_size = 4
    prediction_dictionary={0: 'with_mask', 1: 'without_mask'}
    colour_dict={0:(0,255,0),1:(0,0,255)}

    button=st.button('Start Video')

    if button:

        cap = cv2.VideoCapture(0)
        while True:
            (rval, im) = cap.read()
            im = cv2.flip(im, 1, 1)



            rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
            faces = haarcascade.detectMultiScale(rerect_size)
            for f in faces:
                (x, y, w, h) = [v * rect_size for v in f]

                face_img = im[y: y + h, x: x + w]

                img = cv2.resize(im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

                img = img_to_array(img)

                img = img / 255.0

                lol = np.array([img])

                #label,final_prediction=classify.predict(lol)

                #ans = loaded_model(lol, training=False)



                #cv2.rectangle(im, (x, y), (x + w, y + h), colour_dict[label], 2)
                #cv2.rectangle(im, (x, y - 40), (x + w, y), colour_dict[label], -1)
                #cv2.putText(im, final_prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('LIVE', im)
            key = cv2.waitKey(10)

            if key == 27:
                break

        cap.release()

        cv2.destroyAllWindows()

    st.title(" ")
    st.subheader('Press Escape key to close Video')


if __name__=='__main__':'''
    main()
