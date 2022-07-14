import cvzone
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np



cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
textList = ['This is Computer Vision Project',
'We are studying','Dynamic Text Reader']
 

while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img,draw=False)
    if faces:
        face = faces[0] 
        pointLeft = face[145]
        pointRight = face[374]
        # cv2.line(img,pointLeft,pointRight,(0,200,0),3)
        # cv2.circle(img,pointLeft,5,(255,0,255),cv2.FILLED)
        # cv2.circle(img,pointRight,5,(255,0,255),cv2.FILLED)
        w, _ = detector.findDistance(pointLeft,pointRight)
        W  = 6.3
        # d = 50
        f = 300
        
        d = (W*f)/w
        print(d)

        cvzone.putTextRect(img, f'Depth is {int(d)}cm',
        (face[10][0]-75,face[10][1]-50),
        scale= 2)
        
        for i, text in enumerate(textList):
            singleheight = 20 * int(d/6)
            scale = 0.5 + int((d/20)*10)/80
            cv2.putText(imgText,text,(50,50+(i*singleheight)),cv2.FONT_ITALIC,scale,(255,255,255),2)

    imgStatcked = cvzone.stackImages([img,imgText],2,1)   
    cv2.imshow("Image", imgStatcked)
    cv2.waitKey(1)