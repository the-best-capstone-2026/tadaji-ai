# src/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.inference import predict_from_bytes

app = FastAPI(title="Tandanji AI Server")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_from_bytes(image_bytes)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
