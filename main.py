from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import os
from deepface import DeepFace
import easyocr
import cv2
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
reader = easyocr.Reader(['en', 'id'], gpu=False)

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
def upload_image(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(file_path)

    try:
        face_result = DeepFace.analyze(img_path=file_path, actions=['age', 'gender'], enforce_detection=False)
        face_data = face_result[0]
    except:
        face_data = {"age": "Unknown", "gender": "Unknown"}

    result_text = reader.readtext(file_path)
    extracted_text = " ".join([res[1] for res in result_text])

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": file.filename,
        "face_data": face_data,
        "extracted_text": extracted_text
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
