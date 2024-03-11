import cv2
import cvzone
import numpy as np
from ultralytics import  YOLO
import math
from sort import Sort



class eyes:
    def __init__(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        classNames = ["person", "bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light"
                      ,"fire hydrant", "stop sign", "parking meter", "bench","bird","cat", "dog","horse","sheep",
                      "cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
                      "frisbee","skis","snowboard","sports ball", "kite","baseball bat", "baseball glove","skateboard",
                      "sulfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
                      "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
                      "pottedplant","bed","dining table","toilet","tv "," laptop","mouse","remote","keyboard","phone ",
                      "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
                        "hair drier" ,"toothbrush"]

        model = YOLO('yolov8n.pt')

        tracker = Sort(max_age=20 , min_hits=3,iou_threshold=0.3)
        while True:
            success, img = cap.read()
            result = model(img, stream=True)

            detections = np.empty((0, 5))
            #add bounding boxes to all the objects detected
            for r in result:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    #x1, y1, w, h = box.xywh[0]
                    #bbox = int(x1), int(y1), int(w), int(h)
                    x1, y1 ,x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(x1, y1, x2, y2)
                    w, h = x2-x1, y2-y1
                    cvzone.cornerRect(img,(x1, y1, w, h))
                    bbox = (x1, y1, w, h)

                    conf = math.ceil((box.conf[0]*100))/100
                    cls = int(box.cls[0])
                    name = classNames[cls]
                    cvzone.putTextRect(img,f'{name} {conf}', (max(0, x1),max(35, y1)))
                    if conf > 0.3 :
                        currentArray = np.array([x1,y1,x2,y2, conf])
                        detections = np.vstack((detections, currentArray))


            resultsTracker = tracker.update(detections )

            for result in resultsTracker:
                x1,y1,x2,y2,id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(result)
               # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255.0,0))
                #cvzone.putTextRect(img, f'{name } {conf}', (max(0, x1), max(35, y1)))


            cv2.imshow("image", img)
            cv2.waitKey(1)


    def check(area):
        try:
            pass
        except :
            print("error")


