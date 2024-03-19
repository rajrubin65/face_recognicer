from deepface import DeepFace
from image_enhancer import ImageEnchancer
import tensorflow as tf
from image_preprocessing import *
from PIL import Image


img_enchancer = ImageEnchancer()


class FaceRecognisation():
    def __init__(self,max_perc):
        self.max_perc = max_perc
    
    def get_percentage(self,distance):
        similarity =  1 - distance
        percentage = similarity * 100
        return percentage
    
    def face_recg_deep(self,img1_path,img2_path):
        try:
            img1_path = enhance_image(img1_path)
            img2_path = enhance_image(img2_path)
            result = DeepFace.verify(img1_path,
                            img2_path,
                            model_name='Facenet512',
                            enforce_detection = False)
            print('result>>',result)
            perc = self.get_percentage(result['distance'])
            return perc
        except Exception as e:
            print('Error occured in face_recg_deep method ::',e)
            return None
    
    def face_recg_with_mask(self,img1_path,img2_path):
        try:
            result = DeepFace.verify(img1_path = img1_path,
                            img2_path = img2_path,
                            model_name='VGG-Face',
                            detector_backend='mtcnn',
                            enforce_detection=False)
            perc = self.get_percentage(result['distance'])
            return perc
        except Exception as e:
            print('Error occured in face_recg_deep method ::',e)
            return None
    
    def handle_super_resolution(self,image_path,filename=''):
        try:
            enchanced_image = img_enchancer.get_enchanced_image(image_path)
            # self.save_image(enchanced_image,filename)
            return enchanced_image
        except Exception as e:
            print('Error occured in handle_super_resolution method ::',e)
    
    def save_image(self,image, filename):
        """
            Saves unscaled Tensor Images.
            Args:
            image: 3D image tensor. [height, width, channels]
            filename: Name of the file to save.
        """
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save("%s.jpg" % filename)
        print("Saved as %s.jpg" % filename)

    def match_comparer(self,perc,img1,img2):
        if perc >= self.max_perc:
            return perc,'matched'
        return perc, 'not matched', img1, img2
    
    def executor(self,img1_path,img2_path):
        match_score = self.face_recg_deep(img1_path,img2_path)
        if match_score and match_score >= self.max_perc:
            return self.match_comparer(match_score,img1_path,img2_path)
        else:
            # enchance the images
            print('get less matching >>>>>>>',match_score)
            print('Enchancing the images....')
            img1_path = self.handle_super_resolution(img1_path)
            img2_path = self.handle_super_resolution(img2_path)
            match_score = self.face_recg_deep(img1_path,img2_path)
            if match_score:
                if match_score >= self.max_perc:
                    return self.match_comparer(match_score,img1_path,img2_path)
                else:
                    return self.match_comparer(match_score,img1_path,img2_path)
            else:
                return 0,'not matched',img1_path,img2_path
            
            
    
