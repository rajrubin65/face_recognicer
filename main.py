from face_recognizer import FaceRecognisation
from face_recognizer_vgg_sent50 import FaceRecognisationSent50
import pandas as pd
import cv2


customer_main = 'D:/RubinRaj/Emsyne/Development/face_recognization/images/Nelson_customer/'
customer_otp = 'D:/RubinRaj/Emsyne/Development/face_recognization/images/Nelson_Otp/'

img1 = '030060000003673_Photo.jpg'
img2 = '18360779_30-07-2020-11-56-46-17-AM.jpg'




if __name__ == '__main__':
    # fr = FaceRecognisation(max_perc=70)
    fr = FaceRecognisation(max_perc=70)
    img_path1 = ''.join([customer_main,img1])
    image1 = cv2.imread(img_path1)
    img_path2 = ''.join([customer_otp,img2])
    # img_path2 = 'wrong.jpg'
    image2 = cv2.imread(img_path2)
    
    print(img_path1)
    print(img_path2)
    res = fr.executor(img1_path=image1,
                img2_path=image2)
    print(res)

    # df = pd.read_excel('output.xlsx')
    # df['KERAS_VGGFACE_SENT50_mediapipe_detc'] = None

    # for count,(cur_img_path, new_img_path) in enumerate(zip(df['PHOTO_PATH'],df['CUST_PHOTO_PATH_RENEWAL'])):
    #     cur_img_paths = ''.join([customer_main,cur_img_path])
    #     new_img_paths = ''.join([customer_otp,new_img_path])
    #     print(cur_img_paths,new_img_paths)
    #     try:
    #         df['KERAS_VGGFACE_SENT50_mediapipe_detc'][count] = fr.executor(cur_img_paths,new_img_paths)
    #     except Exception as e:
    #         df['KERAS_VGGFACE_SENT50_mediapipe_detc'][count] = "ERROR"
    #     finally:
    #         print('Match_score >>>>>>>>> ',df['KERAS_VGGFACE_SENT50_mediapipe_detc'][count])
    # df.to_excel('output1803.xlsx')