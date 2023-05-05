import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model(r'model/model.h5')

def img_preprocess(img):
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    return img

cap = cv2.VideoCapture('Idiot Driver.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    img = img_preprocess(frame)
    steering_angle = model.predict(np.array([img]))[0][0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Steering angle: {:.2f}'.format(steering_angle), (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
