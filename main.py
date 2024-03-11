import threading
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import math
import capture
from sort import Sort
import recognition
import pyscreenshot
from pyzbar.pyzbar import decode






class eyes:

    def __init__(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        # list of classification names that can be detected
        classNames = ["person", "bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light"
                      ,"fire hydrant", "stop sign", "parking meter", "bench","bird","cat", "dog","horse","sheep",
                      "cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
                      "frisbee","skis","snowboard","sports ball", "kite","baseball bat", "baseball glove","skateboard",
                      "sulfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
                      "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
                      "pottedplant","bed","dining table","toilet","tv "," laptop","mouse","remote","keyboard","phone ",
                      "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
                        "hair drier","toothbrush"]

        model = YOLO('yolov8n.pt')

        tracker = Sort(max_age=20 , min_hits=3,iou_threshold=0.3)
        while True:
            success, img = cap.read()
            result = model(img, stream=True)
            barcodes = decode(img)
            #barcode and QR-code reading from live video footage and decoding
            #provide a bounding box for the detected
            for barcode in decode(img):
                barcodeData = barcode.data.decode('utf-8')
                pts = np.array([barcode.polygon], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(img,[pts], True, (255,0,255),5)
                pts2 = barcode.rect
                #cv2.putText(img, barcodeData,(pts2[0]),(pts2[1]),0.9,(255,0,255), 2)
                print(barcode.data)


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

                    # only track object with the confidence of above 0.3
                    if conf > 0.3 :
                        currentArray = np.array([x1,y1,x2,y2, conf])
                        detections = np.vstack((detections, currentArray))
                    # only when the identified object is a person with the confidence of above 8.0 should the face recognition be started

                    if name == "person" and conf > 7.0:
                        self.check(((x1, y1, w, h)))
                        image = capture.capture(cap)
                        username = recognition.recognition(img)
                        if username != None:
                            try:
                                cvzone.putTextRect(img, f'{username} {conf}', (max(0, x1), max(35, y1)))
                            except:
                                pass
                        else:
                            pass
                            cvzone.putTextRect(img, f'{name} {conf}', (max(0, x1), max(35, y1)))

                        


            resultsTracker = tracker.update(detections )
            # tracking each of the identified objects
            for result in resultsTracker:
                x1,y1,x2,y2,id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(result)
               # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255.0,0))
                #cvzone.putTextRect(img, f'{name } {conf}', (max(0, x1), max(35, y1)))


            cv2.imshow("image", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                img = self.AddFace(cap)
            else:
                pass

    # compares  the provide face image  with the faces in the database
    def check_face(area):
        result = recognition.recognition.testRecognition(area , "/photos/oscar/oscar.jpg")
        if result == True :
            print("oscar found "+ area)

    # creates  a thread for every  face from every frame of the capture
    def check(area):
        image = pyscreenshot.grab(bbox=area)
        try:
            threading.Thread(target=eyes.check_face, args=(image.copy(),)).start()
        except ValueError:
            print("error")


    def AddFace(self, frame):
        image = capture.capture(frame)
        return image


eyes()

