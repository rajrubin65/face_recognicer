import pandas as pd
import cv2
from face_recognizer import FaceRecognisation
from face_recognizer_vgg_sent50 import FaceRecognisationSent50

customer_main = 'D:/RubinRaj/Emsyne/Development/face_recognization/images/Nelson_customer/'
customer_otp = 'D:/RubinRaj/Emsyne/Development/face_recognization/images/Nelson_Otp/'


def image_matcher(img1,img2,match_threshold):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    output = dict()
    fr = FaceRecognisation(max_perc=match_threshold)
    match_score = fr.executor(image1,image2)
    if 25 <= match_score[0] <= match_threshold:
        print(f'get less match score {match_score[:2]} try vgg-face sent50')
        fr_vgg = FaceRecognisationSent50(max_perc=match_threshold)
        match_score = fr_vgg.executor(match_score[2],match_score[3])
    output['match_threshold'] = match_threshold
    output['match_score'] = match_score[0]
    output['match_status'] = match_score[1]
    return output



df = pd.read_excel('output.xlsx')
df['match'] = None
df['match_score'] = None
df.drop(columns=['KERAS_VGGFACE_MATCH_SCORE','KERAS_VGGFACE_SENT50'],inplace=True)
for count,(cur_img_path, new_img_path) in enumerate(zip(df['PHOTO_PATH'],df['CUST_PHOTO_PATH_RENEWAL'])):
    cur_img_paths = ''.join([customer_main,cur_img_path])
    new_img_paths = ''.join([customer_otp,new_img_path])
    print(cur_img_paths,'\n',new_img_paths)
    try:
        data= image_matcher(cur_img_paths,new_img_paths,70)
        df['match'][count]  = data['match_status']
        df['match_score'][count] = data['match_score']
    except Exception as e:
        df['match'] = 'Error'
        df['math_score'] = 'Error'
    finally:
        print('Match_score >>>>>>>>> ',df.loc[count])
df.to_excel('output1903.xlsx')