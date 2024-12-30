import numpy as np
import cv2
def image2texture(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    texture_data = image.ravel().astype('float32') / 255
    return texture_data