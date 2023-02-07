import cv2 as cv
vid = cv.VideoCapture(0)
# address = "https://192.168.1.66:8080/video"
# vid.open(address)

def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier("haar_facedetect.xml")
    faces = haar_cascade.detectMultiScale(gray, scaleFactor= 2.3, minNeighbors=3 )
    for (x,y,a,b) in faces:
        cv.rectangle(img,(x,y),(x+a,y+b),(0,255,0),2)
    return img


while True:
    isTrue, frame = vid.read()

    flipped=cv.flip(frame,1)
    cv.imshow("face_detection",face_detect(flipped))
    key = cv.waitKey(1)

    if key == ord("c") or (cv.getWindowProperty("face_detection",cv.WND_PROP_VISIBLE)<1):
        break
vid.release()
cv.destroyAllWindows()
