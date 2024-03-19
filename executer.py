import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from face_recognizer_vgg_sent50 import FaceRecognisationSent50
from face_recognizer import FaceRecognisation



app = FastAPI()


def read_image(file):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.post("/face_matcher/")
async def face_matcher(match_threshold:int = 70,file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Check if uploaded files are images
    if not (file1.content_type.startswith('image/') and file2.content_type.startswith('image/')):
        raise HTTPException(status_code=400, detail="Uploaded files must be images")
    image1 = read_image(file1)
    image2 = read_image(file2)
    output = dict()
    fr = FaceRecognisation(max_perc=match_threshold)
    match_score = fr.executor(image1,image2)
    if 35 <= match_score[0] <= match_threshold:
        print(f'get less match score {match_score[:2]} try vgg-face sent50')
        fr_vgg = FaceRecognisationSent50(max_perc=match_threshold)
        match_score = fr_vgg.executor(match_score[2],match_score[3])
    output['match_threshold'] = match_threshold
    output['match_score'] = match_score[0]
    output['match_status'] = match_score[1]
    return output




if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)