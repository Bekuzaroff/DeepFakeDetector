import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import uvicorn
import os
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from model.models.multi_model_inference import EnhancedDeepfakeDetector
app = FastAPI(title="Deepfake Detector API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:5501", "http://localhost:5500"],  # addresses of frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

def load_model():
    global detector

    detector = EnhancedDeepfakeDetector()
    
    # create architecture of models
    detector.build_models()
    
    # save weights
    models_loaded = False
    
    if os.path.exists("model/best_efficientnet_b0.pth"):
        detector.models['efficientnet_b0'].load_state_dict(
            torch.load("model/best_efficientnet_b0.pth", map_location='cuda')
        )
        detector.models['efficientnet_b0'].eval()
        print("Loaded EfficientNet model")
        models_loaded = True
    
    if os.path.exists("model/best_resnet50.pth"):
        detector.models['resnet50'].load_state_dict(
            torch.load("model/best_resnet50.pth", map_location='cuda')
        )
        detector.models['resnet50'].eval()
        print("Loaded ResNet50 model")
        models_loaded = True
    
    if not models_loaded:
        print("No model checkpoints found. Please train models first:")
        print("   - best_efficientnet_b0.pth")
        print("   - best_resnet50.pth")
        exit(1)
    
    # ensemble method
    detector.config['ensemble']['method'] = 'simple_average'

# ML model predict method
def predict_deepfake(image_path):
    result = detector.predict_ensemble(image_path).get("Ensemble") # will return a dict with prediction and confidence {}

    is_fake = result.get("prediction", 0) == 1
    confidence = result.get("confidence", 0.5)
    
    return is_fake, confidence


@app.get("/") # root method, returns message only
async def root():
    return {"message": "Deepfake Detector API", "status": "running"}

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    """Анализ изображения на дипфейк"""
    try:
        # file type check
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Загрузите изображение (jpg, png, jpeg)")
        
        # save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploads/{timestamp}_{file.filename}"
        
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # ml prediction
        is_fake, confidence = predict_deepfake(filename)
        
        # remove temp file
        os.remove(filename)
        
        # return json response
        return JSONResponse({
            "success": True,
            "is_fake": is_fake,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 1),
            "result": "deepfake" if is_fake else "real",
            "message": "Изображение похоже на дипфейк" if is_fake else "Изображение выглядит реальным"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    # uvcorn server runner
    load_model() # load our detector
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000
    )




