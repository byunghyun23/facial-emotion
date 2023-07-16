# pip install mtcnn
import cv2
from mtcnn.mtcnn import MTCNN


class Mtcnn:
    def __init__(self):
        self.mtcnn = MTCNN()
        self.required_size = (224, 224)

    def get_face(self, image):
        # Detect
        results = self.mtcnn.detect_faces(image)
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        image = image[y1:y2, x1:x2]

        # Resize
        image = cv2.resize(image, self.required_size)

        return image