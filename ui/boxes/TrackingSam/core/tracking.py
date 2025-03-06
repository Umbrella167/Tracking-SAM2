from ui.boxes.TrackingSam.core.SAM2 import SAM2Image
import cv2
class TrackingSAM2(SAM2Image):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def tracking_obj(self,image,points):
        self.load_image(image)
        mask, scores, logits = self.add_positive_point(points)
        
    def orb_get_points(self,image):
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors
        