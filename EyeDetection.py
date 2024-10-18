import cv2
eye_classifier = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
video = cv2.VideoCapture("./Videos/SwaggerGoneNet9.mp4")
eyeCascade = cv2.CascadeClassifier()
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_classifier.detectMultiScale(gray_image)
    for x,y,w,h in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow("Eyes", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
