import glob
import os
from image_enhancer import ImageEnchancer
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from numpy import asarray
from image_preprocessing import *
import mediapipe as mp


img_enchancer = ImageEnchancer()


class FaceRecognisationSent50():
    def __init__(self,max_perc):
        self.max_perc = max_perc
        self.vgg_model = VGGFace(model='senet50',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg')
        self.face_detector = MTCNN()
    

    def opencv_cascade_face_detector(self,image_path):
        # Load the pre-trained face detector (Haar cascades)
        face_cascade = cv2.CascadeClassifier('models\opencv_cascade_conf\haarcascade_frontalface_default.xml')
        # Read the input image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        output = [{'box':face}for face in faces]
        return output
    
    def detect_faces_blazeface(self,image):
        # Initialize MediaPipe face detection
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Run BlazeFace face detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image_rgb)
            # Draw face detection annotations on the image
            output = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                int(bboxC.width * iw), int(bboxC.height * ih)
                    output.append({'box':[x, y, w, h]})
        return output
        
    
    def extract_face_from_image(self,image_path, required_size=(224, 224)):
        # load image and detect faces
        # apply image preprocessing
        image = enhance_image(image_path)
        # image = blur_background(image)
        # image = plt.imread(image_path)
        detector = self.face_detector
        faces = detector.detect_faces(image)
        # if mtcnn can't detect
        if not faces:
            print("MTCNN can't detect the face\ntry with faces_blazeface")
            faces = self.detect_faces_blazeface(image)
        face_images = []
        for face in faces:
            # extract the bounding box from the requested face
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face_boundary = image[y1:y2, x1:x2]
            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            face_images.append(face_array)
        return face_images
    
    def get_model_scores(self,img1_path,img2_path):
        try:
            faces = [self.extract_face_from_image(img1_path)[0],self.extract_face_from_image(img2_path)[0]]
            samples = asarray(faces, 'float32')
            # prepare the data for the model
            samples = preprocess_input(samples, version=2)
            # create a vggface model object
            model = self.vgg_model
            # perform prediction
            model_scores = model.predict(samples)
            perc = (1 - cosine(model_scores[0],model_scores[1])) *100
            return perc
        except Exception as e:
            print('Error occured in get_model_scores ::',e)
            return None
        
    def match_comparer(self,perc):
        if perc >= self.max_perc:
            return perc,'matched'
        return perc, 'not matched'
    
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
    
    def remove_files_from_folder(folder_path):
        files = glob.glob(os.path.join(folder_path, '*'))
        for file in files:
            try:
                os.remove(file)
                print(f"Removed file: {file}")
            except OSError as e:
                print(f"Error occurred while removing {file}: {e}")
    
    def executor(self,img1_path,img2_path):
        print('>>>>>>>>>>>>>>Image Matching Starts>>>>>>>>>>>>>>')
        print('processing.............')
        match_score = self.get_model_scores(img1_path,img2_path)
        if match_score:
            return self.match_comparer(match_score)
        else:
            return (0,'not matched')
            
            
    
