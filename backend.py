from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import time

from PIL import Image
from utils import image_to_bits, check_bits_alteration
from embedding import embed_data
from extraction import extract_data
from config import TWO_BIT_QIM

app = FastAPI()

UPLOAD_DIR = Path("uploads")
STEGOFOLDER = Path("stego_files")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STEGOFOLDER.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/embed-secret")
async def embed_secret(audio: UploadFile = File(...), image: UploadFile = File(...)):
    audio_filename = UPLOAD_DIR / audio.filename
    if audio_filename.exists():
        audio_filename.unlink()  # remove existing file
    try:
        with open(audio_filename, "wb") as audio_file:
            shutil.copyfileobj(audio.file, audio_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving audio file: {str(e)}")

    image_filename = UPLOAD_DIR / image.filename
    if image_filename.exists():
        image_filename.unlink()  # remove existing file
    try:
        with open(image_filename, "wb") as image_file:
            shutil.copyfileobj(image.file, image_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image file: {str(e)}")

    try:
        secret_data, secret_shape = image_to_bits(path=image_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    files = [
        {
            "file_type": "jpg",
            "height": secret_shape[0],
            "width": secret_shape[1],
            "channels": secret_shape[2],
            "payload_length": len(secret_data)
        }
    ]

    # Generate a unique stego audio filename using the original audio filename
    stego_audio_filename = STEGOFOLDER / f"{audio.filename.split('.')[0]}_stego.wav"
    if stego_audio_filename.exists():
        stego_audio_filename.unlink()  # remove existing stego file

    start_time = time.time()
    try:
        frames_count = embed_data(secret_data=secret_data, files=files, audio_path=audio_filename, stego_path=stego_audio_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding secret: {str(e)}")
    elapsed_time = time.time() - start_time

    return JSONResponse(content={
        "message": "Secret data embedded successfully!",
        "stego_audio_file": stego_audio_filename.name,
        "frames_count":frames_count,
        "elapsed_time": elapsed_time
    })

@app.post("/api/extract-secret")
async def extract_secret(stego: UploadFile = File(...)):
    # Save the uploaded stego audio file into the UPLOAD_DIR folder.
    stego_audio_filename = UPLOAD_DIR / stego.filename
    if stego_audio_filename.exists():
        stego_audio_filename.unlink()  # remove existing file
    try:
        with open(stego_audio_filename, "wb") as file_out:
            shutil.copyfileobj(stego.file, file_out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving stego audio file: {str(e)}")

    # Generate a unique extracted image filename based on stego filename
    extracted_image_path = STEGOFOLDER / f"{stego.filename.split('.')[0]}_extracted.png"
    if extracted_image_path.exists():
        extracted_image_path.unlink()  # remove previous extraction

    start_time = time.time()
    try:
        extracted_data = extract_data(audio_path=stego_audio_filename, extracted_data_path=extracted_image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting secret data: {str(e)}")
    elapsed_time = time.time() - start_time

    return JSONResponse(content={
        "message": "Secret data extracted successfully!",
        "extracted_image_file": extracted_image_path.name,
        "elapsed_time": elapsed_time
    })


@app.get("/stego_files/{filename}")
async def get_stego_file(filename: str):
    stego_file_path = STEGOFOLDER / filename
    if stego_file_path.exists() and stego_file_path.is_file():
        return FileResponse(stego_file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")
