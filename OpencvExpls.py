import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "harcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

while True:
    success,frame  = camera.read()
    if not success :
        break
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0))


    cv2.imshow("Pencere",frame)
    if cv2.waitKey(1) == ord('q'):
        break
        
    

        
