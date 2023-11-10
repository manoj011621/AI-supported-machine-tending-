import cv2
import pickle





width , height = 194, 270
try:
    with open('freespacepos','wb') as f:
            posList = pickle.load(f)
except:
    posList =[]
        


def mouseClick(events,x,y,flags,params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x,y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList) :
            x1,y1 = pos
            if x1<x<x1+width and y1<y1+height:
                posList.pop(i)
   

while True:
    #cv2.rectangle(img,(20,35),(214,305),(255,0,255),2)

    img = cv2.imread('test.jpg')
    for pos in posList:
        cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),(255,0,255),2)
    cv2.imshow("image",img)
    cv2.setMouseCallback("image",mouseClick)
    cv2.waitKey(1)