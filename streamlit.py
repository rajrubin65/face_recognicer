import cv2
import numpy as np
from face_recognizer import FaceRecognisation
from face_recognizer_vgg_sent50 import FaceRecognisationSent50
import streamlit as st
from PIL import Image

st.title("Face Matcher")

st.write("Upload two images to compare")

uploaded_files = st.file_uploader("Choose two images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

find_match = st.button("Find Match Score")


def read_image(file):
    contents = file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

if uploaded_files is not None and len(uploaded_files) == 2:
    image1 = cv2.imdecode(np.fromstring(uploaded_files[0].read(), np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.fromstring(uploaded_files[1].read(), np.uint8), cv2.IMREAD_COLOR)
    fr = FaceRecognisation(max_perc=70)
    match_score = fr.executor(image1,image2)
    if 35 <= match_score[0] <= 70:
        print(f'get less match score {match_score[:2]} try vgg-face sent50')
        fr_vgg = FaceRecognisationSent50(max_perc=70)
        match_score = fr_vgg.executor(match_score[2],match_score[3])

    st.write(f'Match Status ::',match_score[0])
    st.write(f'Match Score ::',match_score[1])