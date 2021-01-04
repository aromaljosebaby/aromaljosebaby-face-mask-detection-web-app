import streamlit as st
import streamlit as st
import cv2 as cv
import tempfile
import classify
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img
def load_haarcascade():
    haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    return haarcascade

haarcascade = load_haarcascade()


rect_size = 4
prediction_dictionary = {0: 'with_mask', 1: 'without_mask'}
colour_dict = {0: (0, 255, 0), 1: (0, 0, 255)}



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


start_button = st.button('Start Video')



#f = st.file_uploader("Upload file")

if start_button:

    #tfile = tempfile.NamedTemporaryFile(delete=False)
    #tfile.write(f.read())

    print('In')
    #vf = cv.VideoCapture(tfile.name)

    stframe = st.empty()
    end_button = st.button('End Video')

    cap = cv.VideoCapture(0)
    while True:

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #stframe.image(gray)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #frame=cv.rotate(frame,cv.cv2.ROTATE_90_COUNTERCLOCKWISE)









        rerect_size = cv.resize(frame, (frame.shape[1] // rect_size, frame.shape[0] // rect_size))
        faces = haarcascade.detectMultiScale(rerect_size)
        for f in faces:
            (x, y, w, h) = [v * rect_size for v in f]

            face_img = frame[y: y + h, x: x + w]

            img = cv.resize(frame, dsize=(224, 224), interpolation=cv.INTER_CUBIC)

            img = img_to_array(img)

            img = img / 255.0

            lol = np.array([img])

            label,final_prediction=classify.predict(lol)

            #ans = loaded_model(lol, training=False)



            cv.rectangle(frame, (x, y), (x + w, y + h), colour_dict[label], 2)
            cv.rectangle(frame, (x, y - 40), (x + w, y), colour_dict[label], -1)
            cv.putText(frame, final_prediction, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        #cv.imshow('LIVE', frame)
        stframe.image(frame)


        if end_button:
            stframe = st.empty()
            cap.release()
            break



    #cv.destroyAllWindows()
    print('done')


