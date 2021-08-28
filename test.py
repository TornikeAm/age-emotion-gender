import cv2 as cv
import torch
from torchvision import models
print('Emotion Task')
exec(open('emotion.py').read())
print('Gender Task')
exec(open('gender.py').read())
print('age Task')
exec(open('age.py').read())


emotion_model = models.resnet18()
num_ftrs=emotion_model.fc.in_features
emotion_model.fc=nn.Linear(num_ftrs,7)
emotion_model.load_state_dict(torch.load('./models/Emotion_model.pt'))
emotion_model.eval()


gender_model=models.resnet18()
gen_num_ftrs=gender_model.fc.in_features
gender_model.fc=nn.Linear(gen_num_ftrs,2)
gender_model.load_state_dict(torch.load('./models/Gendermodel.pt'))
gender_model.eval()

age_model=models.resnet50()
age_num_ftrs=age_model.fc.in_features
age_model.fc=nn.Linear(age_num_ftrs,2)
age_model.load_state_dict(torch.load('./models/AgeModel.pt'))
age_model.eval()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

emotions={0:'anger',1:'contempt',2:'disgust',3:'fear',4:'happy',5:'sadness',6:'surprise'}
ages={0:'0-12',1:'12-18',2:'18-30',3:'30-50',4:'50-70',5:'70+'}

# For webcam input:
cap = cv.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)



    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        height, width,c = image.shape
        mp_drawing.draw_detection(image, detection)
        bbox = detection.location_data.relative_bounding_box
        x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
        boundbox=int(x*width), int(y*height),int(w*width),int(h*height)

        cropped_face = image[int(y * height):int((y + h) * height), int(x * width):int((x + w) * width)]
        cropped_face=cv.resize(cropped_face,(112,112))

        emotion=emotion_model(torch.tensor(cropped_face).view(1,3,112,112)/255)
        gender=gender_model(torch.tensor(cropped_face).view(1,3,112,112)/255)
        age=age_model(torch.tensor(cropped_face).view(1,3,112,112)/255)

        print(gender)
        cv.putText(image,f"{['Male' if torch.argmax(gender)==0 else 'Female']}",(boundbox[0],boundbox[1]-20),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv.putText(image,f"{list(emotions.values())[list(emotions.keys()).index(torch.argmax(emotion).item())] }",(boundbox[0],boundbox[1]+50),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        cv.putText(image,f"{list(ages.values())[list(ages.keys()).index(torch.argmax(age).item())] }",(boundbox[0],boundbox[1]-40),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.imshow('MediaPipe Face Detection', image)
    if cv.waitKey(5) & 0xFF == ord("q"):
      break
cap.release()