from deepface import DeepFace
class recognition:
    def __init__(image):
        pass

    # check  if the image is a  database
    def recognise( image):
        dfs = DeepFace.find(img_path=image, db_path="C:/workspace/my_db")
        return dfs
    # function to test if two images are of the same person
    def testRecognition(image1, image2):
        dfs = DeepFace.verify(img1_path=image1, img2_path=image2)
        return dfs
    # analyse then image and get details like emotion, gender, age estimate and race
    def analyse(image_path):
        objs = DeepFace.analyze(img_path= image_path,
                                actions=['age', 'gender', 'race', 'emotion']
                                )
        return objs
    def stream(self):
        try:
            DeepFace.stream()
        except :
            print(" failed to stream ")

    
result = recognition.analyse("oscar1.jpg")
#print(result)