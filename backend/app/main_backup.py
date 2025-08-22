import io, base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from .models_loader import (
    load_efficientnet, load_resnet18, load_xray_model,
    EFFICIENTNET_CLASSES, RESNET18_CLASSES, XRAY_CLASSES
)
from .utils.gradcam import GradCAM, overlay_heatmap
from .schemas import PredictionResponse, XrayAnalysisResponse
from .medical_analyzer import MedicalAnalyzer

app = FastAPI(title="SkinSight AI Backend")

# Enable CORS for frontend-backend connection during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- image pre-processing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- helper to run inference ----------
def run_inference(image: Image.Image, model_name: str):
    if model_name == "efficientnet":
        model      = load_efficientnet()
        class_list = EFFICIENTNET_CLASSES
        target_layer = model.features[-1]
    elif model_name == "resnet18":
        model      = load_resnet18()
        class_list = RESNET18_CLASSES
        target_layer = model.layer4[-1]
    else:
        raise ValueError("model_name must be 'efficientnet' or 'resnet18'")

    input_tensor = transform(image).unsqueeze(0)
    gradcam      = GradCAM(model, target_layer)
    heatmap      = gradcam.generate(input_tensor)

    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = class_list[pred_idx]
    confidence = float(probs[pred_idx])

    # overlay heatmap
    img_np     = np.array(image.resize((224, 224)))
    cam_overlay= overlay_heatmap(heatmap, img_np)
    buf        = io.BytesIO()
    Image.fromarray(cam_overlay).save(buf, format="PNG")
    gradcam_b64= base64.b64encode(buf.getvalue()).decode()

    prob_dict  = {cls: float(p) for cls, p in zip(class_list, probs)}

    return PredictionResponse(
        model=model_name,
        predicted_class=pred_class,
        confidence=confidence,
        probabilities=prob_dict,
        gradcam_png=gradcam_b64
    )

# ---------- helper to run text inference for X-ray model ----------
def run_text_inference(text: str, model_name: str):
    if model_name != "xray":
        raise ValueError("Text inference is only supported for 'xray' model")
    
    model, tokenizer = load_xray_model()
    class_list = XRAY_CLASSES
    
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    
    pred_idx = int(np.argmax(probs))
    pred_class = class_list[pred_idx]
    confidence = float(probs[pred_idx])
    
    prob_dict = {cls: float(p) for cls, p in zip(class_list, probs)}
    
    # For text models, we don't have GradCAM, so we'll return an empty string
    return PredictionResponse(
        model=model_name,
        predicted_class=pred_class,
        confidence=confidence,
        probabilities=prob_dict,
        gradcam_png=""
    )

# ---------- enhanced X-ray analysis with medical insights ----------
def run_enhanced_xray_analysis(text: str, model_name: str):
    if model_name != "xray":
        raise ValueError("Enhanced analysis is only supported for 'xray' model")
    
    model, tokenizer = load_xray_model()
    class_list = XRAY_CLASSES
    
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    
    pred_idx = int(np.argmax(probs))
    pred_class = class_list[pred_idx]
    confidence = float(probs[pred_idx])
    
    prob_dict = {cls: float(p) for cls, p in zip(class_list, probs)}
    
    # Initialize medical analyzer
    analyzer = MedicalAnalyzer()
    
    # Perform comprehensive medical analysis
    analysis = analyzer.analyze_report(text, pred_class, confidence)
    
    return XrayAnalysisResponse(
        model=model_name,
        predicted_class=pred_class,
        confidence=confidence,
        probabilities=prob_dict,
        key_findings=analysis['key_findings'],
        disease_risks=analysis['disease_risks'],
        medical_suggestions=analysis['medical_suggestions'],
        severity_assessment=analysis['severity_assessment'],
        follow_up_recommendations=analysis['follow_up_recommendations'],
        report_summary=analysis['report_summary'],
        clinical_significance=analysis['clinical_significance']
    )

# ---------- routes ----------
@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, file: UploadFile = File(...)):
    model_name = model_name.lower()
    
    if model_name in ["efficientnet", "resnet18"]:
        # Image-based models
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        try:
            return run_inference(image, model_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    elif model_name == "xray":
        # Text-based X-ray model
        try:
            content = await file.read()
            text = content.decode('utf-8')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid text file or encoding")
        
        try:
            return run_text_inference(text, model_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    else:
        raise HTTPException(status_code=400, detail="model_name must be 'efficientnet', 'resnet18', or 'xray'")

@app.post("/predict-text/xray", response_model=PredictionResponse)
async def predict_text(text: str = Form(...)):
    """Alternative endpoint for direct text input for X-ray model"""
    try:
        return run_text_inference(text, "xray")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/xray", response_model=XrayAnalysisResponse)
async def analyze_xray_file(file: UploadFile = File(...)):
    """Enhanced X-ray analysis with medical insights from file upload"""
    try:
        content = await file.read()
        text = content.decode('utf-8')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid text file or encoding")
    
    try:
        return run_enhanced_xray_analysis(text, "xray")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-text/xray", response_model=XrayAnalysisResponse)
async def analyze_xray_text(text: str = Form(...)):
    """Enhanced X-ray analysis with medical insights from direct text input"""
    try:
        return run_enhanced_xray_analysis(text, "xray")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
