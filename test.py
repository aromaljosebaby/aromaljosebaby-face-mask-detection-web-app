import tensorflow as tf
import cv2
import numpy as np

#model =tf.keras.models.load_model("face_mask_detection_alert_system.h5")
loaded_model=tf.saved_model.load('saved_model\\face_mask_model_with_no_data_aug_on_resnet')
from tensorflow.keras.preprocessing.image import img_to_array, load_img


prediction_dictionary={0: 'with_mask', 1: 'without_mask'}

colour_dict={0:(0,255,0),1:(0,0,255)}



rect_size = 4
cap = cv2.VideoCapture(1)



haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()
    im =cv2.flip(im ,1 ,1)


    '''img = cv2.resize(im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    img = img_to_array(img)

    img = img / 255.0

    lol = np.array([img])


    ans = loaded_model(lol, training=False)



    print(prediction_dictionary[np.argmax(ans)])'''


    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y: y +h, x: x +w]

        img = cv2.resize(im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        img = img_to_array(img)

        img = img / 255.0

        lol = np.array([img])

        ans = loaded_model(lol, training=False)



        label =np.argmax(ans)
        final_prediction=prediction_dictionary[np.argmax(ans)]

        cv2.rectangle(im ,(x ,y) ,( x +w , y +h) ,colour_dict[label] ,2)
        cv2.rectangle(im ,(x , y -40) ,( x +w ,y) ,colour_dict[label] ,-1)
        cv2.putText(im, final_prediction, (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()



img=cv2.imread('332.jpg')
img=cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

img = img_to_array(img)

img=img/255.0



lol=np.array([img])
print(lol.shape)

ans = loaded_model(lol, training=False)

print(ans)



print(prediction_dictionary[np.argmax(ans)])