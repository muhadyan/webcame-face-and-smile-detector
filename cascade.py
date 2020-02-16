import cv2

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('models/haarcascade_smile.xml')


scaleFactor = 1.2
minNeighbor = 8
smile_scaleFactor = 1.7
smile_minNeighbor = 10
color = (0,0,255) # Square Color
smile_color = (0,255,0) # Smile detection color
line = 3 # Line width

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor, minNeighbor)

    for (x, y, width, height) in faces:
        img = cv2.rectangle(img, (x,y), (x+width, y+height), color, line)
        roi_gray = img_gray[y:y+height, x:x+width]
        roi_color = img[y:y+height, x:x+width]
        smiles = smile_cascade.detectMultiScale(roi_gray, smile_scaleFactor, smile_minNeighbor)
        for (s_x, s_y, s_w, s_h) in smiles:
            cv2.rectangle(roi_color,(s_x, s_y),(s_x+s_w, s_y+s_h), smile_color, line)
    cv2.imshow("Live Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # pres q to EXIT
        break
        
        
cap.release()
cv2.destroyAllWindows()