from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

# Load face detector and emotion model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

# Emojis folder
emoji_folder = 'emojis'

# Load emoji images
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
emojis = {}
for label in emotion_labels:
    path = os.path.join(emoji_folder, f"{label}.png")
    emojis[label] = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keep alpha channel

def overlay_emoji(frame, emoji_img, x, y, w, h):
    # Resize emoji width relative to face width
    scale_factor = 0.6  # adjust for smaller emoji
    new_w = int(w * scale_factor)
    new_h = int(emoji_img.shape[0] * (new_w / emoji_img.shape[1]))
    emoji_resized = cv2.resize(emoji_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Position: slightly above the detected face
    y_offset = max(0, y - int(new_h*0.8))  # shift up above head
    x_offset = x + w//2 - new_w//2

    # Ensure within frame
    y1, y2 = y_offset, y_offset + new_h
    x1, x2 = x_offset, x_offset + new_w
    if y1 < 0: y1, y2 = 0, new_h
    if x1 < 0: x1, x2 = 0, new_w
    if y2 > frame.shape[0]: y2 = frame.shape[0]; y1 = y2 - new_h
    if x2 > frame.shape[1]: x2 = frame.shape[1]; x1 = x2 - new_w

    alpha_s = emoji_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0,3):
        frame[y1:y2, x1:x2, c] = (alpha_s * emoji_resized[:, :, c] +
                                  alpha_l * frame[y1:y2, x1:x2, c])
    return frame

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        max_index = int(np.argmax(prediction))
        emotion_label = emotion_labels[max_index]

        frame = overlay_emoji(frame, emojis[emotion_label], x, y, w, h)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

    cv2.imshow('Emotion Detector with Emojis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
