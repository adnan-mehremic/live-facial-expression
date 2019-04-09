# Live facial expression

import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

web_cam = cv2.VideoCapture(0)

while web_cam.isOpened():

	ret, frame = web_cam.read()
	frame = cv2.flip(frame, 1)
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

	for (x, y, width, height) in faces:
		cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
		
		face = frame[int(y):int(y+height), int(x):int(x+width)]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		face = cv2.resize(face, (48, 48))
		
		img = image.img_to_array(face)
		img = np.expand_dims(img, axis=0)
		img /= 255

		predictions = model.predict(img)
		max_index = np.argmax(predictions[0])
		emotion = emotions[int(max_index)]

		cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

	cv2.imshow('Real time facial expression', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


web_cam.release()
cv2.destroyAllWindows()
