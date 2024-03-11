import cv2
import pyscreenshot

class capture:

    #function to open the image captured
    def open_png(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        cv2.imshow(image, "image")
        cv2.waitKey(0)


    #the default capture function to capture an image of a frame
    def __init__(self,cap):
        # Capture a frame from the webcam
        ret, frame = cap.read()

        if ret:
           # image = cv2.imwrite('/path/to/image.jpg', frame)
            image = cv2.imshow("image ", frame)
            print("Image captured!")
            cv2.waitKey(0)
        else:
            print("Error capturing image")

        return image


