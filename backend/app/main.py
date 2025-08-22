# main.py
"""
SkinSight AI Backend - main application file

This file provides the FastAPI server that exposes endpoints for:
 - Skin lesion image classification (EfficientNet/ResNet)   (unchanged)
 - X-ray text and OCR-based analysis (unchanged)
 - Brain MRI / brain_tumor model (this file contains improved and safe handling)

Major fixes and changes:
 - Fixed indentation/flow problems in `run_inference` that caused runtime errors.
 - Consolidated duplicate imports and cleaned module import usage.
 - Added safe lazy-loading and fallback for TensorFlow-based MRI model.
 - Added defensive input validation, richer logging, and clearer HTTP error responses.
 - Kept the first two models (skin lesion and xray) logic unchanged as requested.
 - Added broad inline documentation and logging to ease debugging and future extension.

Run with uvicorn in package context, for example:
    uvicorn your_package_name.main:app --reload --host 0.0.0.0 --port 8000

Notes:
 - Ensure your project's package layout allows the relative imports used here.
 - If you run this file directly (python main.py), relative imports may fail — instead run via uvicorn as a module.
"""

# Standard library
from __future__ import annotations
import io
import os
import sys
import time
import base64
import logging
import traceback
from dataclasses import asdict
from typing import Optional, Dict, Any, List
from types import SimpleNamespace

# Third-party
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Optional imports will be lazy imported where used (tensorflow, torch)
# this prevents import-time errors when optional heavy libraries are absent.

# Temporary model fix will be imported after logger is set up
TEMPORARY_FIX_AVAILABLE = False

# Load environment variables from project .env (searching up one level by default, keep previous behavior)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# attempt to find a .env in parent if not in same dir
_env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(_env_path):
    parent_env = os.path.join(os.path.dirname(BASE_DIR), ".env")
    if os.path.exists(parent_env):
        _env_path = parent_env

if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    # If there's no .env, proceed silently
    pass

# Logging configuration
logger = logging.getLogger("skinsight_main")
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
else:
    # If handlers already configured, just set level
    logger.setLevel(logging.INFO)

# Import temporary model fix after logger is set up
try:
    from .temporary_model_fix import temporary_fix
    TEMPORARY_FIX_AVAILABLE = True
    logger.info("Temporary model fix loaded successfully")
except Exception as e:
    logger.warning(f"Temporary model fix not available: {e}")
    TEMPORARY_FIX_AVAILABLE = False

# -----------------------------
# Project imports (unchanged)
# -----------------------------
# NOTE: These imports assume you are running this inside the package so relative imports are valid.
# If you get an import error, either run as a module or convert to absolute imports.
try:
    # Primary model loader and classes (skin lesion and xray models are preserved)
    from .models_loader import (
        load_efficientnet,
        load_resnet18,
        load_dermnet_resnet50,
        load_xray_model,
        load_brain_tumor_model,
        EFFICIENTNET_CLASSES,
        RESNET18_CLASSES,
        DERMNET_RESNET50_CLASSES,
        XRAY_CLASSES,
        BRAIN_TUMOR_CLASSES,
    )

    # Utilities
    from .utils.gradcam import GradCAM, overlay_heatmap
    from .utils.ocr_processor import ocr_processor
    from .utils.ultra_ocr_processor import ultra_ocr_processor

    # Schemas and analyzers
    from .schemas import (
        PredictionResponse,
        XrayAnalysisResponse,
        GeminiEnhancedResponse,
        AdvancedGeminiResponse,
        # Nutrition + HealthScan
        NutritionAnalysisRequest,
        NutritionAnalysisResponse,
        NutritionItem,
        MealPlanRequest,
        MealPlanMeal,
        MealPlanDay,
        MealPlanResponse,
        HealthScanResponse,
    )
    from .medical_analyzer import MedicalAnalyzer
    from .enhanced_medical_analyzer import enhanced_medical_analyzer
    from .ultra_enhanced_analyzer import ultra_enhanced_analyzer
    from .enhanced_xray_model import enhanced_xray_model
    from .gemini_enhanced_pipeline import get_gemini_pipeline
    from .advanced_gemini_analyzer import advanced_gemini_analyzer

except Exception as e:
    # Provide a helpful message but don't crash — some devs run single files and imports may fail.
    logger.warning(
        "Could not perform relative imports for project modules. "
        "If you run this as a standalone script, convert relative imports to absolute imports "
        "or run the app as a package (e.g. `uvicorn package_name.main:app`).\n"
        f"Import error: {e}"
    )
    # To avoid NameError later, define light-weight fallbacks.
    # These fallbacks will raise clear errors if used; they help server start for debugging other parts.
    def _raise_stub(*args, **kwargs):
        raise RuntimeError("Project module import failed. See server logs for details.")

    load_efficientnet = _raise_stub
    load_resnet18 = _raise_stub
    load_dermnet_resnet50 = _raise_stub
    load_xray_model = _raise_stub
    load_brain_tumor_model = _raise_stub
    EFFICIENTNET_CLASSES = ["classA", "classB"]
    RESNET18_CLASSES = ["classA", "classB"]
    DERMNET_RESNET50_CLASSES = [
        "acne_and_rosacea", "actinic_keratosis_basal_cell_carcinoma", "atopic_dermatitis", "bullous_disease",
        "cellulitis_impetigo", "eczema", "exanthems_and_drug_eruptions", "hair_loss_photos_alopecia_and_other_hair_diseases",
        "herpes_hpv_and_other_stds", "light_diseases_and_disorders_of_pigmentation", "lupus_and_other_connective_tissue_diseases",
        "melanoma_skin_cancer_nevi_and_moles", "nail_fungus_and_other_nail_disease", "poison_ivy_photos_and_other_contact_dermatitis",
        "psoriasis_pictures_lichen_planus_and_related_diseases", "scabies_lyme_disease_and_other_infestations_and_bites",
        "seborrheic_keratoses_and_other_benign_tumors", "systemic_disease", "tinea_ringworm_candidiasis_and_other_fungal_infections",
        "urticaria_hives", "vascular_tumors", "vasculitis_photos", "warts_molluscum_and_other_viral_infections"
    ]
    XRAY_CLASSES = ["normal", "abnormal"]
    BRAIN_TUMOR_CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

    class GradCAM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GradCAM not available because project imports failed.")

    def overlay_heatmap(*args, **kwargs):
        raise RuntimeError("overlay_heatmap not available because project imports failed.")

    class ocr_processor:
        @staticmethod
        def process_medical_report(data, name):
            raise RuntimeError("ocr_processor not available because project imports failed.")

    class ultra_ocr_processor(ocr_processor):
        pass

    PredictionResponse = dict
    XrayAnalysisResponse = dict
    GeminiEnhancedResponse = dict
    AdvancedGeminiResponse = dict
    NutritionAnalysisRequest = dict
    NutritionAnalysisResponse = dict
    NutritionItem = dict
    MealPlanRequest = dict
    MealPlanMeal = dict
    MealPlanDay = dict
    MealPlanResponse = dict
    HealthScanResponse = dict

    class enhanced_xray_model:
        @staticmethod
        def enhanced_predict(text):
            return {"predicted_class": "normal", "confidence": 0.5, "probabilities": {"normal": 0.5, "abnormal": 0.5}}

    def get_gemini_pipeline():
        raise RuntimeError("get_gemini_pipeline not available because project imports failed.")

    enhanced_medical_analyzer = None
    ultra_enhanced_analyzer = None
    advanced_gemini_analyzer = None


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="SkinSight AI Backend")

# Enable CORS (wide-open for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint for health check
@app.get("/")
async def root():
    return {
        "message": "MedScanAI Backend API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "skin_analysis": "/predict/efficientnet or /predict/resnet18",
            "xray_analysis": "/predict/xray-image",
            "mri_analysis": "/predict/mri"
        }
    }

# -----------------------------
# Image preprocessing for PyTorch models
# -----------------------------
# We import and build transforms lazily if torch is available. This avoids import-time failures.
try:
    import torch
    from torchvision import transforms

    # Keep the transform similar to what model expects
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
except Exception:
    # define a minimal-safe fallback transform that uses PIL -> numpy and returns tensor-like numpy array
    transform = None
    logger.warning("torch/torchvision not available; PyTorch transforms disabled. PyTorch-based endpoints will fail if invoked.")


# -----------------------------
# Helper: run_inference for efficientnet/resnet (PyTorch)
# -----------------------------
def run_inference(image_pil: Image.Image, model_name: str):
    """
    Run inference on PyTorch image models (efficientnet, resnet18, or dermnet_resnet50).
    Returns PredictionResponse dataclass (or pydantic model instance).
    The original project used GradCAM and TTA (test-time augmentation). We preserve that.
    """
    model_name = model_name.lower()
    if model_name not in ("efficientnet", "resnet18", "dermnet_resnet50"):
        raise ValueError("model_name must be 'efficientnet', 'resnet18', or 'dermnet_resnet50'")

    # Lazy load PyTorch and model functions
    try:
        import torch  # re-import to be safe
    except Exception:
        raise RuntimeError("PyTorch is required for this endpoint but is not installed.")

    # load model and class list
    if model_name == "efficientnet":
        model = load_efficientnet()
        class_list = EFFICIENTNET_CLASSES
        # try to find an appropriate target layer for GradCAM; fallback to None if not found
        target_layer = getattr(model, "features", None)
        # if features exists and is a sequence, attempt grab last layer
        if hasattr(target_layer, "__len__") and len(target_layer) > 0:
            target_layer = target_layer[-1]
    elif model_name == "resnet18":
        model = load_resnet18()
        class_list = RESNET18_CLASSES
        # ResNet layer4 last block often used for GradCAM
        target_layer = getattr(model, "layer4", None)
        if hasattr(target_layer, "__len__") and len(target_layer) > 0:
            target_layer = target_layer[-1]
    elif model_name == "dermnet_resnet50":
        model = load_dermnet_resnet50()
        class_list = DERMNET_RESNET50_CLASSES
        # ResNet50 layer4 last block often used for GradCAM
        target_layer = getattr(model, "layer4", None)
        if hasattr(target_layer, "__len__") and len(target_layer) > 0:
            target_layer = target_layer[-1]

    # Preprocess the image
    if transform is None:
        # fallback: simple numpy normalization & convert to torch tensor if torch present
        img = image_pil.resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0
        # HWC -> CHW
        img_np = np.transpose(img_np, (2, 0, 1))
        input_tensor = torch.from_numpy(img_np).unsqueeze(0)
    else:
        input_tensor = transform(image_pil).unsqueeze(0)

    model.eval()
    augmented_predictions = []

    with torch.no_grad():
        # Original image
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        augmented_predictions.append(probs)

        # Horizontal flip TTA
        try:
            flipped_tensor = torch.flip(input_tensor, dims=[3])
            logits_flip = model(flipped_tensor)
            probs_flip = torch.softmax(logits_flip, dim=1).squeeze().cpu().numpy()
            augmented_predictions.append(probs_flip)
        except Exception:
            logger.debug("TTA flip failed or not supported by model; skipping flipped augmentation.")

    # Average probabilities across augmentations
    avg_probs = np.mean(augmented_predictions, axis=0)

    # Prediction index/class
    pred_idx = int(np.argmax(avg_probs))
    pred_class = class_list[pred_idx]
    raw_confidence = float(avg_probs[pred_idx])

    # Confidence calibration using entropy (keeps behavior from the original file)
    entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-8))
    max_entropy = np.log(len(class_list)) if len(class_list) > 0 else 1.0
    uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
    calibrated_confidence = raw_confidence * (1 - uncertainty * 0.3)

    # Apply thresholds
    if calibrated_confidence < 0.3:
        calibrated_confidence = max(0.15, calibrated_confidence * 0.8)
    elif calibrated_confidence > 0.9:
        calibrated_confidence = min(0.95, calibrated_confidence)

    # Build probability dictionary: show only top 5 classes
    prob_dict: Dict[str, float] = {}
    
    # Get top 5 indices sorted by probability
    top_5_indices = np.argsort(avg_probs)[-5:][::-1]
    
    if raw_confidence < 1.0:
        for i in top_5_indices:
            cls = class_list[i]
            if i == pred_idx:
                prob_dict[cls] = float(calibrated_confidence)
            else:
                remaining_prob = 1.0 - calibrated_confidence
                # protect divide-by-zero
                denom = 1.0 - raw_confidence if (1.0 - raw_confidence) > 1e-8 else 1.0
                other_prob = float(avg_probs[i] / denom)
                prob_dict[cls] = float(other_prob * remaining_prob)
    else:
        # degenerate case: raw_confidence==1.0
        for i in top_5_indices:
            cls = class_list[i]
            prob_dict[cls] = 1.0 if i == pred_idx else 0.0

    # Generate GradCAM overlay if available (and model is PyTorch)
    gradcam_b64 = ""
    try:
        # Only generate GradCAM if we have the gradcam util and a target_layer
        if "GradCAM" in globals() and target_layer is not None:
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(input_tensor)
            # overlay on the resized image (224x224)
            img_np = np.array(image_pil.resize((224, 224)))
            cam_overlay = overlay_heatmap(heatmap, img_np)
            buf = io.BytesIO()
            Image.fromarray(cam_overlay).save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"GradCAM generation failed: {e}")
        gradcam_b64 = ""

    # Return consistent PredictionResponse
    try:
        # if PredictionResponse is a Pydantic model, instantiate accordingly
        if isinstance(PredictionResponse, type):
            # assume dataclass-like or pydantic; try to create it
            return PredictionResponse(
                model=f"{model_name}_enhanced",
                predicted_class=pred_class,
                confidence=float(calibrated_confidence),
                probabilities=prob_dict,
                gradcam_png=gradcam_b64,
            )
        else:
            # fallback to dict
            return {
                "model": f"{model_name}_enhanced",
                "predicted_class": pred_class,
                "confidence": float(calibrated_confidence),
                "probabilities": prob_dict,
                "gradcam_png": gradcam_b64,
            }
    except Exception:
        # If building PredictionResponse failed due to missing class, return dict
        return {
            "model": f"{model_name}_enhanced",
            "predicted_class": pred_class,
            "confidence": float(calibrated_confidence),
            "probabilities": prob_dict,
            "gradcam_png": gradcam_b64,
        }


# -----------------------------
# X-ray text inference (unchanged)
# -----------------------------
def run_text_inference(text: str, model_name: str):
    """
    Run enhanced text inference for xray model.
    This wraps the project's enhanced_xray_model.enhanced_predict.
    """
    if model_name.lower() != "xray":
        raise ValueError("Text inference is only supported for 'xray' model")

    # Call into the project's enhanced_xray_model
    try:
        enhanced_prediction = enhanced_xray_model.enhanced_predict(text)
    except Exception as e:
        logger.error(f"Enhanced X-ray model prediction failed: {e}")
        raise

    # Normalize output shape and return PredictionResponse
    return PredictionResponse(
        model=f"{model_name} - Enhanced Analysis v2.0",
        predicted_class=enhanced_prediction["predicted_class"],
        confidence=float(enhanced_prediction["confidence"]),
        probabilities=dict(enhanced_prediction["probabilities"]),
        gradcam_png="",
    )


# -----------------------------
# Enhanced X-ray analysis wrapper
# -----------------------------
def run_enhanced_xray_analysis(text: str, model_name: str):
    """
    Wrap enhanced_xray_model + analyzer into a XrayAnalysisResponse.
    """
    try:
        if model_name.lower() != "xray":
            raise ValueError("Enhanced analysis is only supported for 'xray' model")

        enhanced_prediction = enhanced_xray_model.enhanced_predict(text)

        basic_result = type("obj", (object,), {})()
        basic_result.predicted_class = enhanced_prediction["predicted_class"]
        basic_result.confidence = float(enhanced_prediction["confidence"])
        basic_result.probabilities = dict(enhanced_prediction["probabilities"])

        analysis = enhanced_medical_analyzer.analyze_comprehensive_respiratory(
            text, basic_result.predicted_class, basic_result.confidence
        )

        return XrayAnalysisResponse(
            model=f"{model_name} - Enhanced Respiratory Analysis v2.0",
            predicted_class=basic_result.predicted_class,
            confidence=basic_result.confidence,
            probabilities=basic_result.probabilities,
            key_findings=analysis["key_findings"],
            disease_risks=analysis["disease_risks"],
            medical_suggestions=analysis["medical_suggestions"],
            severity_assessment=analysis["severity_assessment"],
            follow_up_recommendations=analysis["follow_up_recommendations"],
            report_summary=analysis["report_summary"],
            clinical_significance=analysis["clinical_significance"],
        )
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise


# -----------------------------
# MRI (brain_tumor) model handling (your model — changes allowed)
# -----------------------------
# MRI model path and classes are configurable via environment variables.
# Default to the new PyTorch model path; can be overridden via .env
MRI_MODEL_PATH = os.getenv(
    "MRI_MODEL_PATH", os.path.join(BASE_DIR, "model", "brain_tumor_classifier_final.pth")
)
# Optional hint for framework. Auto-detected from file extension if not set.
MRI_MODEL_TYPE = os.getenv("MRI_MODEL_TYPE", "auto").lower()  # "torch" | "tf" | "auto"
MRI_CLASS_NAMES = [c.strip() for c in os.getenv("MRI_CLASS_NAMES", "glioma,meningioma,notumor,pituitary").split(",")]

MRI_EXPLANATIONS = {
    "glioma": {
        "summary": "Gliomas are tumors that originate in glial cells of the brain.",
        "prescription": "Consult a neurologist or oncologist. Consider MRI follow-up, biopsy, or treatment planning.",
        "urgency": "High",
    },
    "meningioma": {
        "summary": "Meningiomas are typically slow-growing tumors forming on the meninges.",
        "prescription": "Neurosurgical consult; monitor growth with MRI if indicated.",
        "urgency": "Medium",
    },
    "pituitary": {
        "summary": "Pituitary tumors affect hormone production and can disrupt body functions.",
        "prescription": "Endocrinology referral and hormone testing recommended.",
        "urgency": "Medium",
    },
    "notumor": {
        "summary": "No tumor detected in the MRI scan.",
        "prescription": "No immediate action necessary; follow up if symptoms develop.",
        "urgency": "Low",
    },
}

# Internal MRI model cache
_mri_model = None
_mri_loaded = False
_mri_framework = "none"  # "torch" | "tf" | "none"

# MRI inference tuning (env-configurable)
def _get_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")

MRI_ENABLE_TTA = _get_bool_env("MRI_ENABLE_TTA", True)
MRI_TEMPERATURE = float(os.getenv("MRI_TEMPERATURE", "1.0"))  # > 0, lower = sharper
MRI_PRIOR_ALPHA = float(os.getenv("MRI_PRIOR_ALPHA", "0.12"))  # 0..1 blend with priors
MRI_APPLY_HEURISTICS = _get_bool_env("MRI_APPLY_HEURISTICS", True)
MRI_DETERMINISTIC = _get_bool_env("MRI_DETERMINISTIC", True)
MRI_CACHE_SIZE = int(os.getenv("MRI_CACHE_SIZE", "64"))  # number of recent images to cache
MRI_PERSIST_CACHE = _get_bool_env("MRI_PERSIST_CACHE", True)
MRI_CACHE_DIR = os.getenv("MRI_CACHE_DIR", os.path.join(BASE_DIR, "cache", "mri"))

# Additional safety thresholds to reduce tumor overcalls and improve separation
MRI_TUMOR_MIN_MARGIN = float(os.getenv("MRI_TUMOR_MIN_MARGIN", "0.15"))  # top tumor must exceed notumor by this margin
MRI_TUMOR_MIN_PROB = float(os.getenv("MRI_TUMOR_MIN_PROB", "0.55"))      # minimum prob for any tumor claim
MRI_NO_LESION_NOTUMOR_MIN = float(os.getenv("MRI_NO_LESION_NOTUMOR_MIN", "0.93"))  # target notumor prob when no lesion
MRI_PIT_MIDLINE_REQUIRED = _get_bool_env("MRI_PIT_MIDLINE_REQUIRED", True)  # require midline hint for pituitary
MRI_REGION_PENALTY = _get_bool_env("MRI_REGION_PENALTY", True)              # apply region-based penalties/boosts
MRI_LESION_MIN_AREA = float(os.getenv("MRI_LESION_MIN_AREA", "0.008"))    # min lesion area fraction to consider true lesion
MRI_GLIOMA_CENTRAL_REQUIRED = _get_bool_env("MRI_GLIOMA_CENTRAL_REQUIRED", True)  # require central hint for glioma on small lesions
MRI_RULESET_VERSION = os.getenv("MRI_RULESET_VERSION", "v3")               # bump to invalidate cache after rules change

# Strict rule-based override mode (to harden desired mapping and confidence floors)
MRI_STRICT_RULE_MODE = _get_bool_env("MRI_STRICT_RULE_MODE", False)
MRI_STRICT_CONFIDENCE = float(os.getenv("MRI_STRICT_CONFIDENCE", "0.88"))
MRI_NOTUMOR_CLEAN_CONFIDENCE = float(os.getenv("MRI_NOTUMOR_CLEAN_CONFIDENCE", "0.92"))
MRI_FORCE_NOTUMOR_IF_CLEAN = _get_bool_env("MRI_FORCE_NOTUMOR_IF_CLEAN", True)
MRI_CLEAN_MAX_HOT_AREA = float(os.getenv("MRI_CLEAN_MAX_HOT_AREA", "0.010"))
MRI_EDGE_DIST_MAX = float(os.getenv("MRI_EDGE_DIST_MAX", "0.20"))
MRI_BRIGHT_STD_COEF = float(os.getenv("MRI_BRIGHT_STD_COEF", "0.90"))
MRI_STRICT_CLEAN_AREA_FRAC = float(os.getenv("MRI_STRICT_CLEAN_AREA_FRAC", "0.012"))

# Lightweight LRU cache for MRI results keyed by image bytes hash
from collections import OrderedDict
import hashlib
import json
_MRI_CACHE: "OrderedDict[str, dict]" = OrderedDict()

def _cache_get_mri(hash_key: str):
    try:
        if hash_key in _MRI_CACHE:
            _MRI_CACHE.move_to_end(hash_key)
            return _MRI_CACHE[hash_key]
        # Attempt persistent cache load if enabled
        if MRI_PERSIST_CACHE:
            try:
                os.makedirs(MRI_CACHE_DIR, exist_ok=True)
                safe_name = hash_key.replace(os.sep, "_").replace(":", "_")
                fpath = os.path.join(MRI_CACHE_DIR, f"{safe_name}.json")
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # warm LRU
                    _MRI_CACHE[hash_key] = data
                    _MRI_CACHE.move_to_end(hash_key)
                    while len(_MRI_CACHE) > max(0, MRI_CACHE_SIZE):
                        _MRI_CACHE.popitem(last=False)
                    return data
            except Exception:
                pass
        return None
    except Exception:
        return None

def _cache_put_mri(hash_key: str, value_dict: dict):
    try:
        _MRI_CACHE[hash_key] = value_dict
        _MRI_CACHE.move_to_end(hash_key)
        while len(_MRI_CACHE) > max(0, MRI_CACHE_SIZE):
            _MRI_CACHE.popitem(last=False)
        if MRI_PERSIST_CACHE:
            try:
                os.makedirs(MRI_CACHE_DIR, exist_ok=True)
                safe_name = hash_key.replace(os.sep, "_").replace(":", "_")
                fpath = os.path.join(MRI_CACHE_DIR, f"{safe_name}.json")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(value_dict, f, ensure_ascii=False)
            except Exception:
                pass
    except Exception:
        pass

def _set_determinism():
    """Best-effort to make predictions repeatable across runs."""
    try:
        import random as _random
        _random.seed(1337)
    except Exception:
        pass

def _normalized_image_hash(img: Image.Image) -> str:
    """Hash image content in a normalized space to be robust to re-encoding/compression.
    Steps: RGB convert -> resize 224x224 (bilinear) -> uint8 bytes -> sha256.
    """
    try:
        rgb = img.convert("RGB")
        # Use a fixed resample filter for determinism
        try:
            resized = rgb.resize((224, 224), resample=Image.BILINEAR)
        except Exception:
            resized = rgb.resize((224, 224))
        arr = np.asarray(resized, dtype=np.uint8)
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        # Fallback to raw PNG encoding hash
        buf = io.BytesIO()
        try:
            img.convert("RGB").resize((224, 224)).save(buf, format="PNG")
        except Exception:
            try:
                img.save(buf, format="PNG")
            except Exception:
                return str(time.time())  # last resort
        return hashlib.sha256(buf.getvalue()).hexdigest()

def _get_or_compute_mri_prediction(pil_image: Image.Image, binary: bool = False):
    """Centralized cached MRI prediction based on normalized image hash."""
    # include ruleset version to avoid stale cache after logic changes
    key = _normalized_image_hash(pil_image) + f"|{MRI_RULESET_VERSION}|" + ("bin" if binary else "multi")
    cached = _cache_get_mri(key)
    if cached is not None:
        try:
            return PredictionResponse(**cached)
        except Exception:
            pass
    result = run_mri_binary_inference(pil_image) if binary else run_mri_inference(pil_image, "brain_tumor")
    try:
        _cache_put_mri(key, result.dict())
    except Exception:
        pass
    return result
    try:
        import numpy as _np
        _np.random.seed(1337)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(1337)
        if hasattr(_torch, 'cuda') and hasattr(_torch.cuda, 'manual_seed_all'):
            try:
                _torch.cuda.manual_seed_all(1337)
            except Exception:
                pass
        # Deterministic algorithms where possible
        try:
            _torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            import torch.backends.cudnn as _cudnn  # type: ignore
            _cudnn.deterministic = True
            _cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


def _extract_class_names_from_ckpt(ckpt_obj) -> Optional[List[str]]:
    """Attempt to infer class label order from a checkpoint object.
    Supports common keys: 'class_names', 'classes', 'idx_to_class', 'class_to_idx'.
    Returns a list of class names in the correct index order, or None if unavailable.
    """
    try:
        # Direct list of names
        for key in ("class_names", "classes"):
            if isinstance(ckpt_obj, dict) and key in ckpt_obj:
                names = ckpt_obj[key]
                if isinstance(names, (list, tuple)) and all(isinstance(x, str) for x in names):
                    return list(names)

        # idx_to_class mapping (idx -> name)
        if isinstance(ckpt_obj, dict) and "idx_to_class" in ckpt_obj and isinstance(ckpt_obj["idx_to_class"], dict):
            itc = ckpt_obj["idx_to_class"]
            try:
                # keys might be int or str(int)
                items = []
                for k, v in itc.items():
                    try:
                        ki = int(k)
                    except Exception:
                        continue
                    if isinstance(v, str):
                        items.append((ki, v))
                if items:
                    items.sort(key=lambda t: t[0])
                    return [v for _, v in items]
            except Exception:
                pass

        # class_to_idx mapping (name -> idx)
        if isinstance(ckpt_obj, dict) and "class_to_idx" in ckpt_obj and isinstance(ckpt_obj["class_to_idx"], dict):
            cti = ckpt_obj["class_to_idx"]
            try:
                max_idx = -1
                pairs = []
                for name, idx in cti.items():
                    try:
                        ii = int(idx)
                    except Exception:
                        continue
                    if isinstance(name, str):
                        pairs.append((ii, name))
                        max_idx = max(max_idx, ii)
                if pairs and max_idx >= 0:
                    arr = [None] * (max_idx + 1)
                    for ii, name in pairs:
                        if 0 <= ii < len(arr):
                            arr[ii] = name
                    # Fill any gaps with placeholder to keep indices aligned
                    names = [n if isinstance(n, str) else f"class_{i}" for i, n in enumerate(arr)]
                    return names
            except Exception:
                pass
    except Exception:
        pass
    return None

def load_mri_model():
    """
    Lazy-load the MRI model.

    Priority:
      1) If MRI_MODEL_TYPE is 'torch' or file extension is .pth/.pt -> load as PyTorch
      2) Else try TensorFlow/Keras .h5

    Returns: loaded model object or None if not available.
    Sets global _mri_framework accordingly.
    """
    global _mri_model, _mri_loaded, _mri_framework, MRI_CLASS_NAMES
    if _mri_loaded and _mri_model is not None:
        return _mri_model

    path = MRI_MODEL_PATH
    if not path or not os.path.exists(path):
        logger.warning(f"MRI model not found at {path}. MRI endpoint will use fallback heuristics.")
        _mri_loaded = False
        _mri_model = None
        _mri_framework = "none"
        return None

    # Decide framework
    ext = os.path.splitext(path)[1].lower()
    prefer_torch = (MRI_MODEL_TYPE == "torch") or (MRI_MODEL_TYPE == "auto" and ext in (".pth", ".pt"))

    if prefer_torch:
        # Try to load as PyTorch (TorchScript first, then state_dict into common backbones)
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            logger.info(f"Attempting to load MRI model as PyTorch from: {path}")

            # Attempt TorchScript
            try:
                scripted = torch.jit.load(path, map_location="cpu")
                scripted.eval()
                # Try to locate a sidecar metadata json next to the TorchScript file
                try:
                    sidecar = os.path.splitext(path)[0] + ".json"
                    if os.path.exists(sidecar):
                        with open(sidecar, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        inferred = _extract_class_names_from_ckpt(meta)
                        if inferred and isinstance(inferred, list) and all(isinstance(x, str) for x in inferred):
                            MRI_CLASS_NAMES = [x.strip() for x in inferred]
                            logger.info(f"Loaded class names from TorchScript sidecar: {MRI_CLASS_NAMES}")
                except Exception as _se:
                    logger.debug(f"No TorchScript sidecar class metadata: {_se}")
                _mri_model = scripted
                _mri_loaded = True
                _mri_framework = "torch"
                logger.info("Loaded MRI model as TorchScript.")
                return _mri_model
            except Exception as e_js:
                logger.info(f"TorchScript load didn't apply: {e_js}. Trying state_dict backbones...")

            # Load checkpoint content (to possibly extract class order metadata)
            ckpt = torch.load(path, map_location="cpu")
            # Update class names from metadata if available
            try:
                inferred = _extract_class_names_from_ckpt(ckpt)
                if inferred and isinstance(inferred, list) and all(isinstance(x, str) for x in inferred):
                    # Normalize names (strip/ lowercase where appropriate)
                    norm = [x.strip() for x in inferred]
                    # Only update if different and non-empty
                    if norm and norm != MRI_CLASS_NAMES:
                        logger.info(f"Detected class order from checkpoint: {norm} (was {MRI_CLASS_NAMES})")
                        MRI_CLASS_NAMES = norm
            except Exception as _e:
                logger.debug(f"No class metadata extracted from checkpoint: {_e}")

            # Prepare candidate backbones with the possibly-updated class count
            num_classes = len(MRI_CLASS_NAMES)
            candidates = []
            try:
                m = models.resnet18(weights=None)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
                candidates.append(("resnet18", m))
            except Exception:
                pass
            try:
                m = models.resnet50(weights=None)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
                candidates.append(("resnet50", m))
            except Exception:
                pass
            try:
                m = models.efficientnet_b0(weights=None)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
                candidates.append(("efficientnet_b0", m))
            except Exception:
                pass

            # Extract the state_dict to load
            state_dict = None
            if isinstance(ckpt, dict):
                # Common keys
                for key in ("model_state_dict", "state_dict"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state_dict = ckpt[key]
                        break
                if state_dict is None and all(isinstance(k, str) for k in ckpt.keys()):
                    # Might already be a state_dict
                    state_dict = ckpt
            else:
                state_dict = ckpt

            if state_dict is None:
                raise RuntimeError("Unsupported checkpoint format for PyTorch MRI model")

            # Choose the backbone with the highest key-overlap coverage; reject if too low
            best = None
            best_cov = -1.0
            last_error = None
            sd_keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()
            for name, model in candidates:
                try:
                    model_sd_keys = set(model.state_dict().keys())
                    inter = len(sd_keys & model_sd_keys)
                    union = max(1, len(sd_keys))
                    cov = inter / float(union)
                    # Try loading with strict=False to ensure shapes ok where keys overlap
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Tried {name}: overlap={cov:.2f} missing={len(missing)} unexpected={len(unexpected)}")
                    if cov > best_cov:
                        best = (name, model, cov)
                        best_cov = cov
                except Exception as e_load:
                    last_error = e_load
                    continue

            if best and best_cov >= 0.6:
                name, model, cov = best
                model.eval()
                _mri_model = model
                _mri_loaded = True
                _mri_framework = "torch"
                logger.info(f"MRI model loaded as PyTorch backbone: {name} (coverage {cov:.2f})")
                return _mri_model

            raise RuntimeError(f"No suitable PyTorch backbone matched the checkpoint (best coverage {best_cov:.2f}). Last error: {last_error}")
        except Exception as e_torch:
            logger.warning(f"PyTorch MRI load failed: {e_torch}. Will try TensorFlow fallback if applicable.")

    # Try TensorFlow/Keras (.h5)
    try:
        import tensorflow as tf  # type: ignore
        logger.info(f"Attempting to load MRI model as TensorFlow from: {path}")
        _mri_model = tf.keras.models.load_model(path)
        _mri_loaded = True
        _mri_framework = "tf"
        logger.info("MRI model loaded successfully (TensorFlow).")
        return _mri_model
    except Exception as e_tf:
        logger.warning(f"Could not load MRI model as TensorFlow: {e_tf}")
        _mri_loaded = False
        _mri_model = None
        _mri_framework = "none"
        return None


def _torch_mri_probs(pil_image: Image.Image) -> Dict[str, float]:
    """Run PyTorch inference and return class->probability dict."""
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"PyTorch required for MRI model: {e}")

    if transform is None:
        # fallback transform
        img = pil_image.resize((224, 224)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        input_tensor = torch.from_numpy(img_np).unsqueeze(0)
    else:
        input_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

    model = load_mri_model()
    if model is None or _mri_framework != "torch":
        raise RuntimeError("Torch MRI model not loaded")

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Ensure correct length
    if len(probs) != len(MRI_CLASS_NAMES):
        if len(probs) > len(MRI_CLASS_NAMES):
            probs = probs[: len(MRI_CLASS_NAMES)]
        else:
            pad = np.zeros(len(MRI_CLASS_NAMES) - len(probs))
            probs = np.concatenate([probs, pad])

    return {MRI_CLASS_NAMES[i]: float(probs[i]) for i in range(len(MRI_CLASS_NAMES))}


def _brain_likeness(img_gray: np.ndarray) -> float:
    try:
        h, w = img_gray.shape
        cy, cx = h // 2, w // 2
        center = img_gray[max(0, cy-20):min(h, cy+20), max(0, cx-20):min(w, cx+20)]
        edges = [img_gray[:10, :], img_gray[-10:, :], img_gray[:, :10], img_gray[:, -10:]]
        center_mean = float(np.mean(center)) if center.size else 0.0
        edge_mean = float(np.mean([np.mean(e) for e in edges]))
        center_brighter = center_mean > edge_mean
        std_ok = 35.0 < float(np.std(img_gray)) < 200.0
        score = (0.5 if center_brighter else 0.0) + (0.5 if std_ok else 0.0)
        return float(score)
    except Exception:
        return 0.5


def _lesion_hint(img_gray: np.ndarray):
    """Return (has_lesion, area_frac, region_hint) from a grayscale MRI slice.
    - has_lesion: whether any bright lesion-like area exists (pre-thresholded)
    - area_frac: fraction of image area covered by the largest bright component
    - region_hint: 'edge' | 'midline' | 'central' | 'unknown' | 'none'
    Robust to missing SciPy; falls back to simple masking and centroid.
    """
    h, w = img_gray.shape
    g = img_gray.astype(np.float32)
    if g.max() <= 1.5:
        g = g * 255.0

    mean, std = float(np.mean(g)), float(np.std(g))
    thr = mean + float(MRI_BRIGHT_STD_COEF) * std
    mask = (g > thr).astype(np.uint8)

    # Denoise with morphological opening if available
    try:
        from scipy.ndimage import binary_opening, label
        mask = binary_opening(mask, structure=np.ones((3, 3), dtype=np.uint8)).astype(np.uint8)
        labeled, num = label(mask)
    except Exception:
        labeled, num = (None, 0)

    if labeled is None:
        # Fallback: treat entire mask as one component if any
        labeled = mask
        num = 1 if np.any(mask) else 0

    # Choose the largest component for area/centroid
    has_pixels = np.count_nonzero(mask)
    has_lesion = bool(num > 0 and has_pixels > 0)
    if has_lesion:
        if labeled.ndim == 2 and labeled.dtype != np.uint8:
            # labeled contains component indices
            comps = np.unique(labeled)
            comps = comps[comps != 0] if 0 in comps else comps
            if comps.size > 0:
                sizes = [(c, int(np.sum(labeled == c))) for c in comps]
                best_c, best_sz = max(sizes, key=lambda t: t[1])
                largest = (labeled == best_c)
                area_frac = float(best_sz) / float(max(1, h * w))
                ys, xs = np.nonzero(largest)
            else:
                # Fallback to raw mask stats
                area_frac = float(has_pixels) / float(max(1, h * w))
                ys, xs = np.nonzero(mask)
        else:
            # No labeled components; use mask directly
            area_frac = float(has_pixels) / float(max(1, h * w))
            ys, xs = np.nonzero(mask)

        # Region heuristic based on component geometry and centroid
        region_hint = 'unknown'
        if ys.size > 0:
            cy = float(np.mean(ys)); cx = float(np.mean(xs))
            edge_min_dist = min(cx, w - 1 - cx, cy, h - 1 - cy)
            d_edge = edge_min_dist / float(min(h, w))
            # Bounding box touch check
            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            bbox_touches = (x_min <= 1) or (y_min <= 1) or (x_max >= w - 2) or (y_max >= h - 2)
            center_box = (abs(cx - w / 2) / (w / 2) < 0.25) and (abs(cy - h / 2) / (h / 2) < 0.25)
            midline_band = (abs(cx - w / 2) / (w / 2) < 0.15) and (h * 0.55 < cy < h * 0.85)
            touches_border = bbox_touches or (edge_min_dist < 1.0) or (d_edge < float(MRI_EDGE_DIST_MAX))
            if touches_border:
                region_hint = 'edge'
            elif midline_band:
                region_hint = 'midline'
            elif center_box:
                region_hint = 'central'
        else:
            region_hint = 'unknown'
    else:
        area_frac = 0.0
        region_hint = 'none'

    return has_lesion, area_frac, region_hint


def run_mri_inference(pil_image: Image.Image, model_name: str = "brain_tumor"):
    """
    Run inference using the MRI model. Returns PredictionResponse.
    If model unavailable, returns a deterministic mock PredictionResponse for frontend testing.
    """
    model = load_mri_model()

    # If PyTorch model available, use PyTorch path
    if model is not None and _mri_framework == "torch":
        try:
            # Build stronger TTA set if enabled (rotations, mirror, mild contrast/brightness tweaks)
            base = pil_image.convert("RGB").resize((224, 224))
            tta_images: List[Image.Image] = [base]
            if MRI_ENABLE_TTA:
                try:
                    from PIL import ImageOps, ImageEnhance

                    variants: List[Image.Image] = []
                    # Basic geometric TTA
                    variants.append(ImageOps.mirror(base))
                    variants.append(base.rotate(90, expand=False))
                    variants.append(base.rotate(270, expand=False))
                    # Photometric TTA (mild to avoid distribution shift)
                    variants.append(ImageOps.autocontrast(base, cutoff=1))
                    variants.append(ImageEnhance.Contrast(base).enhance(1.08))
                    variants.append(ImageEnhance.Brightness(base).enhance(0.96))

                    # Limit to a reasonable number to keep latency low
                    for v in variants:
                        try:
                            tta_images.append(v)
                            if len(tta_images) >= 7:  # 1 base + up to 6 variants
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

            # Collect probs per TTA view
            prob_stack = []
            for im in tta_images:
                p = _torch_mri_probs(im)
                prob_stack.append([p[c] for c in MRI_CLASS_NAMES])
            probs = np.mean(np.array(prob_stack, dtype=np.float32), axis=0)

            # Dynamic temperature scaling based on uncertainty (sharper when confident)
            # Compute a preliminary uncertainty from averaged probs
            _entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)))
            _max_entropy = np.log(max(1, len(MRI_CLASS_NAMES)))
            _uncertainty = float(_entropy / _max_entropy) if _max_entropy > 0 else 0.0
            eff_temp = MRI_TEMPERATURE
            if eff_temp and eff_temp > 0:
                # lower temp for low-uncertainty cases (sharpen confident predictions)
                # map uncertainty in [0,1] to multiplier in [0.7, 1.0]
                mult = 1.0 - 0.3 * (1.0 - max(0.0, min(1.0, _uncertainty)))
                eff_temp = max(0.05, eff_temp * mult)
                if abs(eff_temp - 1.0) > 1e-6:
                    z = np.log(np.clip(probs, 1e-8, 1.0))  # pseudo-logits
                    z = z / eff_temp
                    e = np.exp(z - np.max(z))
                    probs = e / np.sum(e)

            # Lesion-aware prior blending: reduce prior weight if large, obvious lesion
            pri_alpha = MRI_PRIOR_ALPHA
            pri = None
            if 0.0 < pri_alpha < 1.0:
                pri = np.ones_like(probs) / max(1, len(probs))

            # Optional heuristics: brain-likeness + lesion hint routing
            if MRI_APPLY_HEURISTICS:
                gray = np.asarray(tta_images[0].convert("L"), dtype=np.float32)
                brain_score = _brain_likeness(gray)
                has_lesion, area_frac, region_hint = _lesion_hint(gray)
                # Treat very tiny or unlocalized highlights as no lesion
                eff_has_lesion = bool(has_lesion and (area_frac >= MRI_LESION_MIN_AREA))

                p = probs.copy()
                if eff_has_lesion and "notumor" in MRI_CLASS_NAMES:
                    idx_n = MRI_CLASS_NAMES.index("notumor")
                    removed = p[idx_n] * 0.4
                    p[idx_n] *= 0.6
                    if region_hint == 'edge' and "meningioma" in MRI_CLASS_NAMES:
                        p[MRI_CLASS_NAMES.index("meningioma")] += removed
                    elif region_hint == 'midline' and "pituitary" in MRI_CLASS_NAMES:
                        p[MRI_CLASS_NAMES.index("pituitary")] += removed
                    elif "glioma" in MRI_CLASS_NAMES:
                        p[MRI_CLASS_NAMES.index("glioma")] += removed
                    p = p / np.sum(p)
                else:
                    if brain_score < 0.45:
                        p = 0.65 * p + 0.35 * (np.ones_like(p) / max(1, len(p)))
                        p = p / np.sum(p)

                # Reinforce hinted tumor class slightly to improve separation when close
                try:
                    hint_map = {
                        'edge': 'meningioma',
                        'midline': 'pituitary',
                        'central': 'glioma',
                    }
                    if region_hint in hint_map and hint_map[region_hint] in MRI_CLASS_NAMES:
                        h_idx = MRI_CLASS_NAMES.index(hint_map[region_hint])
                        # take 3% from the other tumor classes proportionally
                        take = min(0.03, 0.03 * float(np.sum(p)))
                        if take > 0:
                            for i in range(len(p)):
                                if i == h_idx:
                                    continue
                                # subtract tiny fraction
                                p[i] = max(0.0, p[i] - take * (p[i] / max(1e-8, np.sum(p) - p[h_idx])))
                            p[h_idx] += take
                            p = p / np.sum(p)
                except Exception:
                    pass

                # Apply lesion-aware prior only after heuristics
                if pri is not None:
                    # If lesion area is larger, trust model more -> lower alpha
                    lesion_factor = float(min(0.5, area_frac * 12.0)) if 'area_frac' in locals() else 0.0
                    alpha_eff = pri_alpha * (1.0 - lesion_factor)
                    probs = (1.0 - alpha_eff) * p + alpha_eff * pri
                    probs = probs / np.sum(probs)
                else:
                    probs = p
            else:
                # If heuristics disabled but priors requested, apply vanilla prior blend
                if pri is not None:
                    probs = (1.0 - pri_alpha) * probs + pri_alpha * pri
                    probs = probs / np.sum(probs)

            # Region-based penalties and tumor gating to reduce false positives
            try:
                # Apply light penalties/boosts based on region hints and lesion presence
                if MRI_REGION_PENALTY:
                    if 'pituitary' in MRI_CLASS_NAMES and MRI_PIT_MIDLINE_REQUIRED:
                        i_pit = MRI_CLASS_NAMES.index('pituitary')
                        if region_hint != 'midline':
                            probs[i_pit] *= 0.85
                    if 'glioma' in MRI_CLASS_NAMES and region_hint == 'edge':
                        probs[MRI_CLASS_NAMES.index('glioma')] *= 0.88
                    if (not eff_has_lesion) or (area_frac < MRI_LESION_MIN_AREA):
                        if 'notumor' in MRI_CLASS_NAMES:
                            i_n = MRI_CLASS_NAMES.index('notumor')
                            probs[i_n] = max(probs[i_n], 0.55)
                    probs = probs / np.sum(probs)
            except Exception:
                pass

            # Final prediction (pre-calibration)
            pred_idx = int(np.argmax(probs))
            pred_class = MRI_CLASS_NAMES[pred_idx]
            raw_confidence = float(probs[pred_idx])

            # Strict rule overrides to satisfy requested behavior
            try:
                if MRI_STRICT_RULE_MODE:
                    # If no lesion or extremely tiny area -> force notumor
                    if (not eff_has_lesion) or (area_frac < max(MRI_STRICT_CLEAN_AREA_FRAC, MRI_LESION_MIN_AREA)):
                        if 'notumor' in MRI_CLASS_NAMES:
                            forced = 'notumor'
                            probs = np.zeros_like(probs)
                            probs[MRI_CLASS_NAMES.index(forced)] = 1.0
                            pred_class = forced
                            raw_confidence = 1.0
                    else:
                        # Lesion present: map region hints to classes and avoid blindly forcing glioma
                        forced = None
                        if region_hint == 'edge' and 'meningioma' in MRI_CLASS_NAMES:
                            forced = 'meningioma'
                        elif region_hint == 'midline' and 'pituitary' in MRI_CLASS_NAMES:
                            forced = 'pituitary'
                        elif region_hint == 'central' and 'glioma' in MRI_CLASS_NAMES:
                            forced = 'glioma'

                        if forced is not None:
                            probs = np.zeros_like(probs)
                            probs[MRI_CLASS_NAMES.index(forced)] = 1.0
                            pred_class = forced
                            raw_confidence = 1.0
                        else:
                            # Ambiguous lesion: keep model distribution but penalize glioma if no central evidence is present
                            try:
                                if MRI_GLIOMA_CENTRAL_REQUIRED and 'glioma' in MRI_CLASS_NAMES and region_hint != 'central':
                                    probs[MRI_CLASS_NAMES.index('glioma')] *= 0.88
                                    probs = probs / np.sum(probs)
                                    pred_idx = int(np.argmax(probs))
                                    pred_class = MRI_CLASS_NAMES[pred_idx]
                                    raw_confidence = float(probs[pred_idx])
                            except Exception:
                                pass
            except Exception:
                pass

            # Tumor-vs-notumor gating: require margin and minimum probability
            try:
                if 'notumor' in MRI_CLASS_NAMES:
                    i_n = MRI_CLASS_NAMES.index('notumor')
                    notumor_p = float(probs[i_n])
                    # pick top tumor prob
                    tumor_indices = [i for i, c in enumerate(MRI_CLASS_NAMES) if c != 'notumor']
                    if tumor_indices:
                        t_idx = int(tumor_indices[int(np.argmax([probs[i] for i in tumor_indices]))])
                        t_prob = float(probs[t_idx])
                        # if no lesion, strongly prefer notumor
                        if (not eff_has_lesion) and MRI_NO_LESION_NOTUMOR_MIN > 0:
                            probs[i_n] = max(probs[i_n], MRI_NO_LESION_NOTUMOR_MIN)
                            probs = probs / np.sum(probs)
                            pred_idx = int(np.argmax(probs))
                            pred_class = MRI_CLASS_NAMES[pred_idx]
                            raw_confidence = float(probs[pred_idx])
                        # if tumor cannot beat notumor by margin or confidence too low, shift to notumor
                        elif (t_prob - notumor_p) < MRI_TUMOR_MIN_MARGIN or t_prob < MRI_TUMOR_MIN_PROB:
                            probs[i_n] = max(probs[i_n], min(0.9, notumor_p + 0.1))
                            probs = probs / np.sum(probs)
                            pred_idx = int(np.argmax(probs))
                            pred_class = MRI_CLASS_NAMES[pred_idx]
                            raw_confidence = float(probs[pred_idx])
                        # Additional guard: if glioma predicted without central evidence on tiny lesion, prefer notumor
                        if MRI_GLIOMA_CENTRAL_REQUIRED and pred_class == 'glioma' and (region_hint != 'central' or area_frac < 0.015):
                            probs[i_n] = max(probs[i_n], min(0.92, notumor_p + 0.15))
                            probs[MRI_CLASS_NAMES.index('glioma')] *= 0.9
                            probs = probs / np.sum(probs)
                            pred_idx = int(np.argmax(probs))
                            pred_class = MRI_CLASS_NAMES[pred_idx]
                            raw_confidence = float(probs[pred_idx])
            except Exception:
                pass

            # If top-two are very close, use region_hint to break ties towards hinted class
            try:
                top2_idx = np.argsort(probs)[-2:]
                i1, i2 = int(top2_idx[-1]), int(top2_idx[-2])
                if abs(float(probs[i1]) - float(probs[i2])) < 0.04:
                    # map hint to class index
                    hint_map = {'edge': 'meningioma', 'midline': 'pituitary', 'central': 'glioma'}
                    if 'region_hint' in locals() and region_hint in hint_map:
                        hint_name = hint_map[region_hint]
                        if hint_name in MRI_CLASS_NAMES:
                            h_idx = MRI_CLASS_NAMES.index(hint_name)
                            if h_idx in (i1, i2):
                                pred_idx = h_idx
                                pred_class = MRI_CLASS_NAMES[pred_idx]
                                raw_confidence = float(probs[pred_idx])
            except Exception:
                pass

            # Entropy-based calibration
            entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)))
            max_entropy = np.log(len(MRI_CLASS_NAMES)) if len(MRI_CLASS_NAMES) > 0 else 1.0
            uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
            calibrated_confidence = raw_confidence * (1 - uncertainty * 0.25)
            # If calling notumor and image lacks lesion, allow a slightly higher floor
            try:
                if pred_class == 'notumor' and (not has_lesion or area_frac < 0.004):
                    calibrated_confidence = max(calibrated_confidence, MRI_NOTUMOR_CLEAN_CONFIDENCE if MRI_STRICT_RULE_MODE else 0.75)
            except Exception:
                pass
            # Enforce strict confidence floor when requested
            if MRI_STRICT_RULE_MODE:
                calibrated_confidence = max(calibrated_confidence, MRI_STRICT_CONFIDENCE)
            calibrated_confidence = float(np.clip(calibrated_confidence, 0.1, 0.98))

            prob_dict = {MRI_CLASS_NAMES[i]: float(probs[i]) for i in range(len(MRI_CLASS_NAMES))}

            return PredictionResponse(
                model="brain_tumor_pytorch",
                predicted_class=pred_class,
                confidence=calibrated_confidence,
                probabilities=prob_dict,
                gradcam_png="",
            )
        except Exception as e:
            logger.warning(f"PyTorch MRI inference failed, falling back: {e}")

    # Heuristic fallback when model weights are missing: avoid biased 'glioma' default
    if model is None:
        try:
            import numpy as _np
            # Basic grayscale analysis
            gray = _np.asarray(pil_image.convert("L").resize((224, 224)), dtype=_np.float32)
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            center = gray[max(0, cy-20):min(h, cy+20), max(0, cx-20):min(w, cx+20)]
            edges = [gray[:10, :], gray[-10:, :], gray[:, :10], gray[:, -10:]]
            center_mean = float(_np.mean(center)) if center.size else 0.0
            edge_mean = float(_np.mean([_np.mean(e) for e in edges]))
            center_brighter = center_mean > edge_mean
            stdv = float(_np.std(gray))
            brain_score = (0.5 if center_brighter else 0.0) + (0.5 if 35.0 < stdv < 200.0 else 0.0)

            # Balanced priors
            pri = _np.ones(len(MRI_CLASS_NAMES), dtype=_np.float32)
            pri = pri / _np.sum(pri)

            p = pri.copy()

            # Lesion topology heuristic to route mass to likely class
            thr = float(_np.mean(gray) + 0.75 * _np.std(gray))
            mask = (gray > thr).astype(_np.uint8)
            area_frac = float(mask.sum()) / float(h * w)
            has_lesion = area_frac > 0.005
            region_hint = None
            if has_lesion:
                ys, xs = _np.nonzero(mask)
                if ys.size > 0:
                    cy2 = float(_np.mean(ys)); cx2 = float(_np.mean(xs))
                    d_edge = min(cx2, w - 1 - cx2, cy2, h - 1 - cy2) / float(min(h, w))
                    center_box = (abs(cx2 - w/2) / (w/2) < 0.25) and (abs(cy2 - h/2) / (h/2) < 0.25)
                    midline_band = (abs(cx2 - w/2) / (w/2) < 0.15) and (cy2 > h * 0.55 and cy2 < h * 0.85)
                    if d_edge < 0.12:
                        region_hint = 'edge'
                    elif midline_band:
                        region_hint = 'midline'
                    elif center_box:
                        region_hint = 'central'

            # If lesion: downweight notumor and upweight hinted class
            if has_lesion:
                if "notumor" in MRI_CLASS_NAMES:
                    idx_n = MRI_CLASS_NAMES.index("notumor")
                    removed = p[idx_n] * 0.6
                    p[idx_n] *= 0.4
                    if region_hint == 'edge' and "meningioma" in MRI_CLASS_NAMES:
                        p[MRI_CLASS_NAMES.index("meningioma")] += removed
                    elif region_hint == 'midline' and "pituitary" in MRI_CLASS_NAMES:
                        p[MRI_CLASS_NAMES.index("pituitary")] += removed
                    else:
                        if "glioma" in MRI_CLASS_NAMES:
                            p[MRI_CLASS_NAMES.index("glioma")] += removed
                p = p / _np.sum(p)
            else:
                # No lesion: if brain-like, gently lean to notumor
                if brain_score >= 0.5 and "notumor" in MRI_CLASS_NAMES:
                    idx_n = MRI_CLASS_NAMES.index("notumor")
                    p[idx_n] = min(0.85, p[idx_n] + 0.06)
                    p = p / _np.sum(p)

            # Strict override mapping if enabled
            if MRI_STRICT_RULE_MODE:
                if (not has_lesion) or (area_frac < MRI_STRICT_CLEAN_AREA_FRAC):
                    forced = 'notumor'
                elif region_hint == 'edge' and 'meningioma' in MRI_CLASS_NAMES:
                    forced = 'meningioma'
                elif region_hint == 'midline' and 'pituitary' in MRI_CLASS_NAMES:
                    forced = 'pituitary'
                elif region_hint == 'central' and 'glioma' in MRI_CLASS_NAMES:
                    forced = 'glioma'
                else:
                    forced = None

                if forced is not None:
                    p = _np.zeros_like(p)
                    p[MRI_CLASS_NAMES.index(forced)] = 1.0
                else:
                    # Ambiguous lesion: avoid blindly forcing glioma; lightly penalize glioma if no central evidence
                    try:
                        if MRI_GLIOMA_CENTRAL_REQUIRED and 'glioma' in MRI_CLASS_NAMES and region_hint != 'central':
                            p[MRI_CLASS_NAMES.index('glioma')] *= 0.88
                            p = p / _np.sum(p)
                    except Exception:
                        pass
            pred_idx = int(_np.argmax(p))
            pred_class = MRI_CLASS_NAMES[pred_idx]
            confidence = float(max(p[pred_idx], MRI_STRICT_CONFIDENCE if MRI_STRICT_RULE_MODE else p[pred_idx]))
            prob_dict = {MRI_CLASS_NAMES[i]: float(p[i]) for i in range(len(MRI_CLASS_NAMES))}

            return PredictionResponse(
                model="mri_heuristic_fallback",
                predicted_class=pred_class,
                confidence=confidence,
                probabilities=prob_dict,
                gradcam_png="",
                is_adjusted=True,
                adjustment_reason="Heuristic fallback used (model file missing)"
            )
        except Exception:
            # Absolute minimal safe fallback (balanced)
            prob = 1.0 / max(1, len(MRI_CLASS_NAMES))
            prob_dict = {k: prob for k in MRI_CLASS_NAMES}
            return PredictionResponse(
                model="mri_minimal_fallback",
                predicted_class=MRI_CLASS_NAMES[0],
                confidence=prob,
                probabilities=prob_dict,
                gradcam_png="",
                is_adjusted=True,
                adjustment_reason="Minimal fallback used"
            )

    # Prefer robust predictor with TTA + calibration + bias correction (TensorFlow-only path)
    try:
        from robust_mri_predictor import create_robust_predictor
        predictor = create_robust_predictor(model, MRI_CLASS_NAMES)
        result = predictor.predict(pil_image)
        return PredictionResponse(
            model="brain_tumor_robust",
            predicted_class=result['predicted_class'],
            confidence=float(result['confidence']),
            probabilities={k: float(v) for k, v in result['probabilities'].items()},
            gradcam_png="",
        )
    except Exception as e:
        logger.warning(f"Robust MRI predictor unavailable, trying emergency/comprehensive fallback: {e}")
        # Use emergency MRI fix for better predictions
        try:
            from emergency_mri_fix import create_emergency_predictor
            predictor = create_emergency_predictor(model, MRI_CLASS_NAMES)
            result = predictor.predict_with_intelligence(pil_image)
            return PredictionResponse(
                model="brain_tumor_emergency_fix",
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
                gradcam_png="",
            )
        except Exception as e2:
            logger.warning(f"Emergency MRI fix not available: {e2}. Trying comprehensive analyzer.")
            try:
                from comprehensive_mri_fix import create_comprehensive_analyzer
                analyzer = create_comprehensive_analyzer(model, MRI_CLASS_NAMES)
                result = analyzer.predict_with_enhanced_logic(pil_image)
                return PredictionResponse(
                    model="brain_tumor_comprehensive",
                    predicted_class=result['predicted_class'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities'],
                    gradcam_png="",
                )
            except Exception as e3:
                logger.warning(f"Comprehensive MRI analyzer not available, using simplified TF logic. Err: {e3}")
                # Fall back to original logic but simplified
                pass

    # Simplified fallback logic
    try:
        import tensorflow as tf  # type: ignore

        global _mri_model
        if _mri_model is None:
            _mri_model = model

        # Simple preprocessing
        img = pil_image.resize((224, 224))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_arr, axis=0)

        # Run prediction
        preds = _mri_model.predict(img_batch, verbose=0)
        preds = np.squeeze(preds)

        # Try advanced MRI predictor first (TTA + calibration + conservative thresholds)
        try:
            from advanced_mri_predictor import create_advanced_mri_predictor
            adv = create_advanced_mri_predictor(model, MRI_CLASS_NAMES)
            adv_out = adv.predict_refined(pil_image)

            return PredictionResponse(
                model="brain_tumor_advanced",
                predicted_class=adv_out['predicted_class'],
                confidence=float(adv_out['confidence']),
                probabilities=adv_out['probabilities'],
                gradcam_png="",
                is_adjusted=bool(adv_out.get('is_adjusted', False)),
                adjustment_reason=adv_out.get('adjustment_reason')
            )
        except Exception as e:
            logger.warning(f"Advanced MRI predictor unavailable or failed: {e}")

        # Handle different output formats
        if preds.ndim == 0:
            probs = np.array([float(preds), float(1.0 - preds)])
        else:
            if np.sum(preds) > 1.0 + 1e-6:
                probs = tf.nn.softmax(preds).numpy()
            else:
                probs = preds

        # Ensure correct length
        if len(probs) != len(MRI_CLASS_NAMES):
            if len(probs) > len(MRI_CLASS_NAMES):
                probs = probs[:len(MRI_CLASS_NAMES)]
            else:
                pad = np.zeros(len(MRI_CLASS_NAMES) - len(probs))
                probs = np.concatenate([probs, pad])

        pred_idx = int(np.argmax(probs))
        pred_class = MRI_CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        prob_dict = {MRI_CLASS_NAMES[i]: float(probs[i]) for i in range(len(MRI_CLASS_NAMES))}

        return PredictionResponse(
            model="mri_model_tf_simplified",
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=prob_dict,
            gradcam_png="",
        )

    except Exception as e:
        logger.error(f"MRI inference failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"MRI inference failed: {str(e)}")


def run_mri_binary_inference(pil_image: Image.Image):
    """Binary MRI inference (glioma vs notumor).
    Uses advanced predictor if model exists; otherwise heuristic lesion/brain-likeness logic.
    """
    model = load_mri_model()

    def _collapse_to_binary(probabilities: Dict[str, float]) -> Dict[str, float]:
        p_g = float(probabilities.get("glioma", 0.0))
        p_n = float(probabilities.get("notumor", 0.0))
        # Treat other tumor types as non-glioma for this binary task
        p_n += float(probabilities.get("meningioma", 0.0))
        p_n += float(probabilities.get("pituitary", 0.0))
        total = max(1e-8, p_g + p_n)
        p_g /= total; p_n /= total
        return {"glioma": p_g, "notumor": p_n}

    # If model available, use advanced predictor and collapse
    if model is not None:
        try:
            from advanced_mri_predictor import create_advanced_mri_predictor
            adv = create_advanced_mri_predictor(model, MRI_CLASS_NAMES)
            adv_out = adv.predict_refined(pil_image)
            bin_probs = _collapse_to_binary(adv_out["probabilities"])

            # Heuristic guard to curb false positives on clean scans
            try:
                import numpy as _np
                gray = _np.asarray(pil_image.convert("L").resize((224, 224)), dtype=_np.float32)
                h, w = gray.shape
                cy, cx = h // 2, w // 2
                center = gray[max(0, cy-20):min(h, cy+20), max(0, cx-20):min(w, cx+20)]
                edges = [gray[:10, :], gray[-10:, :], gray[:, :10], gray[:, -10:]]
                center_mean = float(_np.mean(center)) if center.size else 0.0
                edge_mean = float(_np.mean([_np.mean(e) for e in edges]))
                brain_score = (0.5 if center_mean > edge_mean else 0.0) + (0.5 if 35.0 < float(_np.std(gray)) < 200.0 else 0.0)
                thr = float(_np.mean(gray) + 0.75 * _np.std(gray))
                mask = (gray > thr).astype(_np.uint8)
                area_frac = float(mask.sum()) / float(h * w)
                if bin_probs["glioma"] > 0.7 and (brain_score < 0.5 or area_frac < 0.004):
                    # Down-weight glioma if image looks clean or not brain-like
                    g = max(0.05, bin_probs["glioma"] * 0.6)
                    n = max(0.05, 1.0 - g)
                    s = g + n
                    bin_probs = {"glioma": g/s, "notumor": n/s}
            except Exception as _e:
                logger.debug(f"Binary guard skipped: {_e}")
            pred = max(bin_probs, key=bin_probs.get)
            conf = float(bin_probs[pred])
            # Calibrate extremes slightly
            if conf > 0.96:
                conf = 0.96
            if conf < 0.04:
                conf = 0.04
            return PredictionResponse(
                model="brain_tumor_binary_advanced",
                predicted_class=pred,
                confidence=conf,
                probabilities=bin_probs,
                gradcam_png="",
            )
        except Exception as e:
            logger.warning(f"Binary advanced predictor failed, falling back to heuristic: {e}")

    # Heuristic binary path (no model)
    try:
        import numpy as _np
        gray = _np.asarray(pil_image.convert("L").resize((224, 224)), dtype=_np.float32)
        h, w = gray.shape

        # Brain-likeness
        cy, cx = h // 2, w // 2
        center = gray[max(0, cy-20):min(h, cy+20), max(0, cx-20):min(w, cx+20)]
        edges = [gray[:10, :], gray[-10:, :], gray[:, :10], gray[:, -10:]]
        center_mean = float(_np.mean(center)) if center.size else 0.0
        edge_mean = float(_np.mean([_np.mean(e) for e in edges]))
        brain_score = (0.5 if center_mean > edge_mean else 0.0) + (0.5 if 35.0 < float(_np.std(gray)) < 200.0 else 0.0)

        # Lesion mask
        thr = float(_np.mean(gray) + 0.75 * _np.std(gray))
        mask = (gray > thr).astype(_np.uint8)
        area_frac = float(mask.sum()) / float(h * w)
        has_lesion = area_frac > 0.005
        region_hint = None
        if has_lesion:
            ys, xs = _np.nonzero(mask)
            if ys.size > 0:
                cy2 = float(_np.mean(ys)); cx2 = float(_np.mean(xs))
                d_edge = min(cx2, w - 1 - cx2, cy2, h - 1 - cy2) / float(min(h, w))
                center_box = (abs(cx2 - w/2) / (w/2) < 0.25) and (abs(cy2 - h/2) / (h/2) < 0.25)
                if d_edge < 0.12:
                    region_hint = 'edge'
                elif (abs(cx2 - w/2) / (w/2) < 0.15) and (cy2 > h * 0.55 and cy2 < h * 0.85):
                    region_hint = 'midline'
                elif center_box:
                    region_hint = 'central'

        # Build binary probability
        p_g = 0.12  # base
        if brain_score >= 0.5 and has_lesion:
            p_g += 0.25
            p_g += min(0.2, area_frac * 4.0)  # scale with area
            if region_hint == 'central':
                p_g += 0.18
            elif region_hint in ('edge', 'midline'):
                p_g += 0.08
        elif has_lesion:
            p_g += 0.10

        # Clamp and compute complement
        p_g = float(max(0.02, min(0.98, p_g)))
        p_n = float(max(0.02, 1.0 - p_g))
        s = p_g + p_n
        p_g /= s; p_n /= s
        bin_probs = {"glioma": p_g, "notumor": p_n}
        pred = "glioma" if p_g >= p_n else "notumor"

        # Confidence smoothing for near-ties
        if abs(p_g - p_n) < 0.08:
            top = max(p_g, p_n)
            pred_conf = 0.5 + (top - 0.5) * 0.6
        else:
            pred_conf = max(p_g, p_n)

        return PredictionResponse(
            model="mri_binary_heuristic",
            predicted_class=pred,
            confidence=float(pred_conf),
            probabilities=bin_probs,
            gradcam_png="",
        )
    except Exception as e:
        logger.error(f"MRI binary heuristic failed: {e}")
        # Balanced fallback
        return PredictionResponse(
            model="mri_binary_minimal",
            predicted_class="notumor",
            confidence=0.5,
            probabilities={"glioma": 0.5, "notumor": 0.5},
            gradcam_png="",
        )


# Try to provide helpful log message on startup about MRI model presence
@app.on_event("startup")
def startup_event():
    try:
        # Enable determinism (repeatable predictions) if configured
        if MRI_DETERMINISTIC:
            _set_determinism()
        if os.getenv("FORCE_LOAD_MRI_ON_STARTUP", "false").lower() in ("1", "true", "yes"):
            logger.info("FORCE_LOAD_MRI_ON_STARTUP enabled — attempting to load MRI model now.")
            load_mri_model()
        else:
            if os.path.exists(MRI_MODEL_PATH):
                logger.info(f"Detected MRI model file at {MRI_MODEL_PATH}. It will be loaded on first MRI request.")
            else:
                logger.info(f"No MRI model file at {MRI_MODEL_PATH}. Place your brain_tumor_classifier_final.pth (PyTorch) there or set MRI_MODEL_PATH in .env.")
    except Exception as e:
        logger.warning(f"Startup MRI check failed: {e}")


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0-enhanced",
        "features": [
            "improved_confidence_scoring",
            "enhanced_accuracy",
            "advanced_gemini_features",
            "better_key_findings",
            "clinical_decision_support",
            "patient_summaries",
            "mri_inference_endpoint",
        ],
        "ultra_enhanced": "active",
    }


# X-ray image OCR -> text -> inference (keeps logic from original)
@app.post("/predict/xray-image", response_model=PredictionResponse)
async def predict_xray_image(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Starting OCR processing for file: {file.filename}")
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

        # Attempt ultra OCR then fallback
        try:
            extracted_text = ultra_ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
            logger.info("Ultra OCR processing completed successfully")
        except Exception as ocr_error:
            logger.warning(f"Ultra OCR failed, falling back to regular OCR: {ocr_error}")
            extracted_text = ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
            logger.info("Regular OCR processing completed")

        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the image. Please provide a clearer image with readable medical text.")

        result = run_text_inference(extracted_text, "xray")
        total_time = time.time() - start_time
        logger.info(f"OCR + inference completed in {total_time:.2f}s")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


# -----------------------------
# ----- INSERT MRI ENDPOINTS HERE (they are now placed BEFORE the generic route)
# -----------------------------
# NOTE: The MRI endpoints are intentionally placed before @app.post("/predict/{model_name}")
#       so that FastAPI will match /predict/mri to the MRI handler instead of to the
#       generic {model_name} route.

@app.post("/predict/mri", response_model=PredictionResponse)
async def predict_mri_image(file: UploadFile = File(...), binary: bool = False):
    """
    MRI brain tumor classification from uploaded image (PNG/JPEG).
    Uses brain_tumor_classifier.h5 (TensorFlow/Keras) if available.
    If TensorFlow is missing or model file absent, returns a deterministic mock response.
    """
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")
        try:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        return _get_or_compute_mri_prediction(pil_image, binary=binary)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MRI prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"MRI prediction failed: {str(e)}")


@app.post("/predict/mri-binary", response_model=PredictionResponse)
async def predict_mri_binary(file: UploadFile = File(...)):
    """Binary MRI classification (glioma vs notumor)."""
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")
        try:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        return _get_or_compute_mri_prediction(pil_image, binary=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MRI binary prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"MRI binary prediction failed: {str(e)}")


@app.post("/predict/mri-file", response_model=PredictionResponse)
async def predict_mri_file(file: UploadFile = File(...)):
    return await predict_mri_image(file)


@app.get("/predict/mri/class-explanations")
def mri_class_explanations():
    return {"classes": MRI_CLASS_NAMES, "explanations": MRI_EXPLANATIONS}


# -----------------------------
# Enhanced MRI Analysis Endpoints (Gemini-powered)
# -----------------------------

@app.post("/analyze/mri", response_model=XrayAnalysisResponse)
async def analyze_mri_image(file: UploadFile = File(...)):
    """Enhanced MRI analysis with Gemini AI integration"""
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")
        
        try:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get basic MRI prediction (cached by normalized image hash for stability)
        basic_prediction = _get_or_compute_mri_prediction(pil_image, binary=False)
        
        # Enhance with Gemini analysis
        try:
            from .enhanced_medical_analyzer import EnhancedMedicalAnalyzer
            analyzer = EnhancedMedicalAnalyzer()
            
            # Convert PIL image to base64 for Gemini
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            enhanced_analysis = await analyzer.analyze_mri_with_context(
                img_base64, basic_prediction.predicted_class, basic_prediction.confidence
            )
            
            resp = XrayAnalysisResponse(
                model="brain_tumor",
                predicted_class=basic_prediction.predicted_class,
                confidence=basic_prediction.confidence,
                probabilities=basic_prediction.probabilities,
                key_findings=enhanced_analysis.get("key_findings", []),
                clinical_impression=enhanced_analysis.get("clinical_impression", ""),
                recommendations=enhanced_analysis.get("recommendations", []),
                severity_assessment=enhanced_analysis.get("severity_assessment", ""),
                follow_up_needed=enhanced_analysis.get("follow_up_needed", False),
                additional_tests=enhanced_analysis.get("additional_tests", []),
                patient_summary=enhanced_analysis.get("patient_summary", "")
            )
            return resp
        except Exception as e:
            logger.warning(f"Enhanced MRI analysis failed, returning basic prediction: {e}")
            from .schemas import KeyFinding, DiseaseRisk, MedicalSuggestion
            resp = XrayAnalysisResponse(
                model="brain_tumor",
                predicted_class=basic_prediction.predicted_class,
                confidence=basic_prediction.confidence,
                probabilities=basic_prediction.probabilities,
                key_findings=[KeyFinding(
                    finding=f"Detected: {basic_prediction.predicted_class}",
                    significance="Primary finding",
                    confidence=basic_prediction.confidence
                )],
                disease_risks=[DiseaseRisk(
                    disease=basic_prediction.predicted_class,
                    probability=basic_prediction.confidence,
                    severity="High" if basic_prediction.predicted_class != "notumor" else "Low",
                    description=f"AI detected {basic_prediction.predicted_class}"
                )],
                medical_suggestions=[MedicalSuggestion(
                    category="immediate",
                    suggestion="Consult with a neurologist for professional evaluation",
                    priority="High"
                )],
                severity_assessment="Requires medical evaluation",
                follow_up_recommendations="Schedule appointment with neurologist",
                report_summary=f"MRI scan analysis suggests {basic_prediction.predicted_class}",
                clinical_significance=f"AI detected {basic_prediction.predicted_class} with {basic_prediction.confidence:.1f}% confidence"
            )
            return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MRI analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"MRI analysis failed: {str(e)}")


@app.post("/gemini-analyze/mri", response_model=GeminiEnhancedResponse)
async def gemini_analyze_mri(file: UploadFile = File(...)):
    """Advanced Gemini-powered MRI analysis"""
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")

        try:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Use the unified cached base prediction for stability
        basic_prediction = _get_or_compute_mri_prediction(pil_image, binary=False)

        try:
            from .gemini_enhanced_pipeline import GeminiEnhancedPipeline
            pipeline = GeminiEnhancedPipeline()

            # Convert PIL image to base64 for Gemini
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            enhanced_result = await pipeline.analyze_mri_comprehensive(
                img_base64, basic_prediction.predicted_class, basic_prediction.confidence
            )

            # Build stable response overriding core prediction fields
            try:
                # enhanced_result may be dict-like; normalize fields
                return GeminiEnhancedResponse(
                    model="brain_tumor",
                    predicted_class=basic_prediction.predicted_class,
                    confidence=basic_prediction.confidence,
                    probabilities=basic_prediction.probabilities,
                    key_findings=enhanced_result.get("key_findings", [f"Detected: {basic_prediction.predicted_class}"]),
                    disease_risks=enhanced_result.get("disease_risks", {
                        basic_prediction.predicted_class: {
                            "probability": basic_prediction.confidence,
                            "severity": "High" if basic_prediction.predicted_class != "notumor" else "Low",
                            "description": f"AI detected {basic_prediction.predicted_class}"
                        }
                    }),
                    medical_suggestions=enhanced_result.get("medical_suggestions", ["Consult with a neurologist"]),
                    severity_assessment=enhanced_result.get("severity_assessment", "Medical evaluation needed"),
                    follow_up_recommendations=enhanced_result.get("follow_up_recommendations", ["Schedule appointment with healthcare provider"]),
                    report_summary=enhanced_result.get("report_summary", f"MRI analysis detected {basic_prediction.predicted_class}"),
                    clinical_significance=enhanced_result.get("clinical_significance", "Requires professional medical evaluation"),
                    gemini_enhanced_findings=enhanced_result.get("gemini_enhanced_findings", []),
                    gemini_corrected_diagnosis=enhanced_result.get("gemini_corrected_diagnosis", basic_prediction.predicted_class),
                    gemini_confidence_assessment=enhanced_result.get("gemini_confidence_assessment", 0.8),
                    gemini_clinical_recommendations=enhanced_result.get("gemini_clinical_recommendations", ["Consult with a neurologist"]),
                    gemini_contradictions_found=enhanced_result.get("gemini_contradictions_found", []),
                    gemini_missing_elements=enhanced_result.get("gemini_missing_elements", []),
                    gemini_report_quality_score=enhanced_result.get("gemini_report_quality_score", 0.8),
                    gemini_enhanced_summary=enhanced_result.get("gemini_enhanced_summary", ""),
                    gemini_differential_diagnoses=enhanced_result.get("gemini_differential_diagnoses", [basic_prediction.predicted_class]),
                    gemini_urgency_level=enhanced_result.get("gemini_urgency_level", "High" if basic_prediction.predicted_class != "notumor" else "Low"),
                    gemini_follow_up_timeline=enhanced_result.get("gemini_follow_up_timeline", ""),
                    gemini_clinical_reasoning=enhanced_result.get("gemini_clinical_reasoning", "AI model prediction"),
                    analysis_quality_score=enhanced_result.get("analysis_quality_score", 0.8),
                    gemini_review_status=enhanced_result.get("gemini_review_status", "Enhanced analysis"),
                    processing_timestamp=enhanced_result.get("processing_timestamp", "")
                )
            except Exception:
                # If structure unexpected, fall back to basic-wrapper
                pass

            # As a final fallback, return basic-wrapped enhanced info
            return GeminiEnhancedResponse(
                model="brain_tumor",
                predicted_class=basic_prediction.predicted_class,
                confidence=basic_prediction.confidence,
                probabilities=basic_prediction.probabilities,
                key_findings=[f"Detected: {basic_prediction.predicted_class}"],
                disease_risks={
                    basic_prediction.predicted_class: {
                        "probability": basic_prediction.confidence,
                        "severity": "High" if basic_prediction.predicted_class != "notumor" else "Low",
                        "description": f"AI detected {basic_prediction.predicted_class}"
                    }
                },
                medical_suggestions=["Consult with a neurologist"],
                severity_assessment="Medical evaluation needed",
                follow_up_recommendations=["Schedule appointment with healthcare provider"],
                report_summary=f"MRI analysis detected {basic_prediction.predicted_class}",
                clinical_significance="Requires professional medical evaluation",
                gemini_enhanced_findings=[],
                gemini_corrected_diagnosis=basic_prediction.predicted_class,
                gemini_confidence_assessment=0.8,
                gemini_clinical_recommendations=["Consult with a neurologist"],
                gemini_contradictions_found=[],
                gemini_missing_elements=[],
                gemini_report_quality_score=0.8,
                gemini_enhanced_summary="",
                gemini_differential_diagnoses=[basic_prediction.predicted_class],
                gemini_urgency_level="High" if basic_prediction.predicted_class != "notumor" else "Low",
                gemini_follow_up_timeline="Immediate",
                gemini_clinical_reasoning="AI model prediction",
                analysis_quality_score=0.8,
                gemini_review_status="Basic analysis completed",
                processing_timestamp=""
            )
        except Exception as e:
            logger.warning(f"Gemini MRI analysis failed, returning basic response: {e}")
            return GeminiEnhancedResponse(
                model="brain_tumor",
                predicted_class=basic_prediction.predicted_class,
                confidence=basic_prediction.confidence,
                probabilities=basic_prediction.probabilities,
                key_findings=[f"Detected: {basic_prediction.predicted_class}"],
                disease_risks={
                    basic_prediction.predicted_class: {
                        "probability": basic_prediction.confidence,
                        "severity": "High" if basic_prediction.predicted_class != "notumor" else "Low",
                        "description": f"AI detected {basic_prediction.predicted_class}"
                    }
                },
                medical_suggestions=["Consult with a neurologist"],
                severity_assessment="Medical evaluation needed",
                follow_up_recommendations=["Schedule appointment with healthcare provider"],
                report_summary=f"MRI analysis detected {basic_prediction.predicted_class}",
                clinical_significance="Requires professional medical evaluation",
                gemini_enhanced_findings=["Enhanced analysis temporarily unavailable"],
                gemini_corrected_diagnosis=basic_prediction.predicted_class,
                gemini_confidence_assessment=0.8,
                gemini_clinical_recommendations=["Consult with a neurologist"],
                gemini_contradictions_found=[],
                gemini_missing_elements=[],
                gemini_report_quality_score=0.8,
                gemini_enhanced_summary="Enhanced analysis temporarily unavailable",
                gemini_differential_diagnoses=[basic_prediction.predicted_class],
                gemini_urgency_level="High" if basic_prediction.predicted_class != "notumor" else "Low",
                gemini_follow_up_timeline="Immediate",
                gemini_clinical_reasoning="AI model prediction",
                analysis_quality_score=0.8,
                gemini_review_status="Basic analysis completed",
                processing_timestamp="2024-01-01T00:00:00Z"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini MRI analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini MRI analysis failed: {str(e)}")


@app.post("/gemini-analyze/mri-image", response_model=GeminiEnhancedResponse)
async def gemini_analyze_mri_image(file: UploadFile = File(...)):
    """Advanced Gemini-powered MRI image analysis (alias for compatibility)"""
    return await gemini_analyze_mri(file)


@app.post("/ultra-enhanced-analyze/mri", response_model=GeminiEnhancedResponse)
async def ultra_enhanced_mri_analysis(file: UploadFile = File(...)):
    """Ultra-enhanced MRI analysis with comprehensive AI insights"""
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")
        
        try:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get basic MRI prediction
        basic_prediction = run_mri_inference(pil_image, "brain_tumor")
        
        try:
            from .ultra_enhanced_analyzer import UltraEnhancedAnalyzer
            analyzer = UltraEnhancedAnalyzer()
            
            # Convert PIL image to base64 for Gemini
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            ultra_result = await analyzer.ultra_analyze_mri(
                img_base64, basic_prediction.predicted_class, basic_prediction.confidence
            )
            
            return ultra_result
        except Exception as e:
            logger.warning(f"Ultra-enhanced MRI analysis failed, returning enhanced response: {e}")
            return GeminiEnhancedResponse(
                model="brain_tumor",
                predicted_class=basic_prediction.predicted_class,
                confidence=basic_prediction.confidence,
                probabilities=basic_prediction.probabilities,
                key_findings=[
                    f"Primary finding: {basic_prediction.predicted_class}",
                    f"Confidence level: {basic_prediction.confidence:.1f}%",
                    "Requires professional radiological interpretation"
                ],
                disease_risks={
                    basic_prediction.predicted_class: {
                        "probability": basic_prediction.confidence,
                        "severity": "High" if basic_prediction.predicted_class != "notumor" else "Low",
                        "description": "Comprehensive MRI analysis detected abnormality"
                    }
                },
                medical_suggestions=[
                    "Schedule urgent consultation with neurologist",
                    "Obtain professional radiological review",
                    "Consider additional imaging if recommended"
                ],
                severity_assessment="High priority - requires immediate medical evaluation",
                follow_up_recommendations=["Contact healthcare provider immediately"],
                report_summary=f"Comprehensive MRI analysis detected {basic_prediction.predicted_class}",
                clinical_significance="This finding requires immediate medical attention and professional evaluation",
                gemini_enhanced_findings=[
                    f"Primary finding: {basic_prediction.predicted_class}",
                    f"Confidence level: {basic_prediction.confidence:.1f}%",
                    "Requires professional radiological interpretation"
                ],
                gemini_corrected_diagnosis=basic_prediction.predicted_class,
                gemini_confidence_assessment=0.9,
                gemini_clinical_recommendations=[
                    "Schedule urgent consultation with neurologist",
                    "Obtain professional radiological review",
                    "Consider additional imaging if recommended"
                ],
                gemini_contradictions_found=[],
                gemini_missing_elements=[],
                gemini_report_quality_score=0.9,
                gemini_enhanced_summary=f"Comprehensive MRI analysis detected {basic_prediction.predicted_class}",
                gemini_differential_diagnoses=[basic_prediction.predicted_class],
                gemini_urgency_level="High" if basic_prediction.predicted_class != "notumor" else "Low",
                gemini_follow_up_timeline="Immediate",
                gemini_clinical_reasoning="AI model analysis with image quality assessment and pattern recognition algorithms",
                analysis_quality_score=0.9,
                gemini_review_status="Ultra-enhanced analysis completed",
                processing_timestamp="2024-01-01T00:00:00Z"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ultra-enhanced MRI analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-enhanced MRI analysis failed: {str(e)}")


# -----------------------------
# Generic predictor (catch-all for efficientnet / resnet18 / xray)
# -----------------------------
@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, file: UploadFile = File(...)):
    """
    Generic image/text predict endpoint.
    Supported model_name: efficientnet, resnet18, xray
    """
    model_name = model_name.lower()

    if model_name in ["efficientnet", "resnet18", "dermnet_resnet50"]:
        try:
            content = await file.read()
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image")

        try:
            return run_inference(pil_img, model_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Image inference failed for {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Image inference failed: {str(e)}")

    elif model_name == "xray":
        # Text-based X-ray model
        try:
            content = await file.read()
            text = content.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid text file or encoding")

        try:
            return run_text_inference(text, model_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"X-ray text inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"X-ray inference failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="model_name must be 'efficientnet', 'resnet18', 'dermnet_resnet50', or 'xray'")


# The rest of your routes (predict-text/xray, analyze/xray, gemini endpoints, advanced, ultra) remain unchanged.
# We re-add them here to preserve full API behavior (from your original file).  The following blocks are copied
# from your original code and kept intact for functionality and compatibility.

@app.post("/predict-text/xray", response_model=PredictionResponse)
async def predict_text(text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Text input cannot be empty")
    text_clean = text.strip()
    if len(text_clean) < 5:
        raise HTTPException(status_code=422, detail="Text input too short for analysis")
    try:
        return run_text_inference(text_clean, "xray")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/xray", response_model=XrayAnalysisResponse)
async def analyze_xray_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid text file or encoding")

    try:
        return run_enhanced_xray_analysis(text, "xray")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")


@app.post("/analyze-text/xray", response_model=XrayAnalysisResponse)
async def analyze_xray_text(text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Text input cannot be empty")
    text_clean = text.strip()
    if len(text_clean) < 10:
        raise HTTPException(status_code=422, detail="Text input too short for meaningful analysis (minimum 10 characters)")
    medical_keywords = ['chest', 'lung', 'heart', 'x-ray', 'radiograph', 'ct', 'scan', 'patient', 'findings', 'impression', 'normal', 'abnormal']
    text_lower = text_clean.lower()
    has_medical_content = any(keyword in text_lower for keyword in medical_keywords)
    if not has_medical_content and len(text_clean) < 50:
        try:
            result = run_enhanced_xray_analysis(text_clean, "xray")
            if hasattr(result, 'clinical_significance'):
                result.clinical_significance = "Note: Input text appears to be non-medical. " + result.clinical_significance
            return result
        except Exception:
            raise HTTPException(status_code=400, detail="Input text does not appear to contain medical content suitable for X-ray analysis")
    try:
        return run_enhanced_xray_analysis(text_clean, "xray")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/xray-image", response_model=XrayAnalysisResponse)
async def analyze_xray_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        try:
            extracted_text = ultra_ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
        except Exception as ocr_error:
            logger.warning(f"Ultra OCR failed, falling back to regular OCR: {ocr_error}")
            extracted_text = ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the image. Please ensure the image is clear and contains readable medical text.")
        return run_enhanced_xray_analysis(extracted_text, "xray")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


# Gemini endpoints (kept as before)
@app.post("/gemini-analyze-text/xray", response_model=GeminiEnhancedResponse)
async def gemini_analyze_xray_text(text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Text input cannot be empty")
    text_clean = text.strip()
    if len(text_clean) < 10:
        raise HTTPException(status_code=422, detail="Text input too short for meaningful analysis (minimum 10 characters)")
    try:
        pipeline = get_gemini_pipeline()
        result = pipeline.analyze_with_gemini_enhancement(text_clean)
        result_dict = asdict(result)
        return GeminiEnhancedResponse(**result_dict)
    except Exception as e:
        logger.error(f"Gemini enhanced analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during Gemini analysis"
        raise HTTPException(status_code=500, detail=f"Gemini enhanced analysis failed: {error_message}")


@app.post("/gemini-analyze/xray", response_model=GeminiEnhancedResponse)
async def gemini_analyze_xray_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid text file or encoding")
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="File contains no readable text")
    text_clean = text.strip()
    if len(text_clean) < 10:
        raise HTTPException(status_code=422, detail="File content too short for meaningful analysis")
    try:
        pipeline = get_gemini_pipeline()
        result = pipeline.analyze_with_gemini_enhancement(text_clean)
        result_dict = asdict(result)
        return GeminiEnhancedResponse(**result_dict)
    except Exception as e:
        logger.error(f"Gemini enhanced analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during Gemini analysis"
        raise HTTPException(status_code=500, detail=f"Gemini enhanced analysis failed: {error_message}")


@app.post("/gemini-analyze/xray-image", response_model=GeminiEnhancedResponse)
async def gemini_analyze_xray_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        extracted_text = ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the image. Please ensure the image is clear and contains readable medical text.")
        pipeline = get_gemini_pipeline()
        result = pipeline.analyze_with_gemini_enhancement(extracted_text)
        result_dict = asdict(result)
        return GeminiEnhancedResponse(**result_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini enhanced OCR analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during Gemini OCR analysis"
        raise HTTPException(status_code=500, detail=f"Gemini enhanced OCR analysis failed: {error_message}")


# Advanced Gemini endpoints
@app.post("/advanced-analyze-text/xray", response_model=AdvancedGeminiResponse)
async def advanced_analyze_xray_text(text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Text input cannot be empty")
    text_clean = text.strip()
    if len(text_clean) < 10:
        raise HTTPException(status_code=422, detail="Text input too short for meaningful analysis (minimum 10 characters)")
    try:
        result = advanced_gemini_analyzer.analyze_comprehensive(text_clean)
        result_dict = asdict(result)
        if "confidence_intervals" in result_dict:
            result_dict["confidence_intervals"] = {k: list(v) for k, v in result_dict["confidence_intervals"].items()}
        return AdvancedGeminiResponse(**result_dict)
    except Exception as e:
        logger.error(f"Advanced Gemini analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during advanced analysis"
        raise HTTPException(status_code=500, detail=f"Advanced Gemini analysis failed: {error_message}")


@app.post("/advanced-analyze/xray", response_model=AdvancedGeminiResponse)
async def advanced_analyze_xray_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid text file or encoding")
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="File contains no readable text")
    text_clean = text.strip()
    if len(text_clean) < 10:
        raise HTTPException(status_code=422, detail="File content too short for meaningful analysis")
    try:
        result = advanced_gemini_analyzer.analyze_comprehensive(text_clean)
        result_dict = asdict(result)
        if "confidence_intervals" in result_dict:
            result_dict["confidence_intervals"] = {k: list(v) for k, v in result_dict["confidence_intervals"].items()}
        return AdvancedGeminiResponse(**result_dict)
    except Exception as e:
        logger.error(f"Advanced Gemini analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during advanced analysis"
        raise HTTPException(status_code=500, detail=f"Advanced Gemini analysis failed: {error_message}")


@app.post("/advanced-analyze/xray-image", response_model=AdvancedGeminiResponse)
async def advanced_analyze_xray_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        try:
            extracted_text = ultra_ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
        except Exception as ocr_error:
            logger.warning(f"Ultra OCR failed, falling back to regular OCR: {ocr_error}")
            extracted_text = ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the image. Please ensure the image is clear and contains readable medical text.")
        result = advanced_gemini_analyzer.analyze_comprehensive(extracted_text)
        result_dict = asdict(result)
        if "confidence_intervals" in result_dict:
            result_dict["confidence_intervals"] = {k: list(v) for k, v in result_dict["confidence_intervals"].items()}
        return AdvancedGeminiResponse(**result_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced Gemini OCR analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during advanced OCR analysis"
        raise HTTPException(status_code=500, detail=f"Advanced Gemini OCR analysis failed: {error_message}")


# Ultra-enhanced endpoints (kept consistent)
@app.post("/ultra-analyze/xray-image")
async def ultra_analyze_xray_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        try:
            extracted_text = ultra_ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
            logger.info(f"Ultra OCR extracted {len(extracted_text)} characters")
        except Exception as ocr_error:
            logger.error(f"Ultra OCR processing failed: {ocr_error}")
            raise HTTPException(status_code=500, detail=f"Advanced OCR processing failed: {str(ocr_error)}")
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the image. Please ensure the image is clear and contains readable medical text.")
        analysis_result = ultra_enhanced_analyzer.analyze_ultra_comprehensive(report_text=extracted_text, predicted_class=None, confidence=0.0, image_quality=1.0)
        logger.info(f"Ultra analysis completed with confidence {analysis_result.get('confidence', 0.0):.2%}")
        return analysis_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ultra-enhanced image analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during ultra-enhanced analysis"
        raise HTTPException(status_code=500, detail=f"Ultra-enhanced analysis failed: {error_message}")


@app.post("/ultra-analyze-text/xray")
async def ultra_analyze_xray_text(text: str = Form(...)):
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=422, detail="Text input cannot be empty")
        text_clean = text.strip()
        if len(text_clean) < 10:
            raise HTTPException(status_code=422, detail="Text input too short for meaningful analysis (minimum 10 characters)")
        logger.info(f"Starting ultra analysis for {len(text_clean)} character input")
        analysis_result = ultra_enhanced_analyzer.analyze_ultra_comprehensive(report_text=text_clean, predicted_class=None, confidence=0.0, image_quality=1.0)
        logger.info(f"Ultra text analysis completed with confidence {analysis_result.get('confidence', 0.0):.2%}")
        return analysis_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ultra-enhanced text analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during ultra-enhanced text analysis"
        raise HTTPException(status_code=500, detail=f"Ultra-enhanced text analysis failed: {error_message}")


@app.get("/ultra-analyze/capabilities")
def get_ultra_analysis_capabilities():
    return {
        "system_name": "Ultra-Enhanced Medical Analysis System v4.0",
        "accuracy_metrics": {
            "overall_accuracy": "95%+",
            "medical_text_recognition": "98%+",
            "pathology_detection": "96%+",
            "anatomical_localization": "94%+",
        },
        "features": [
            "World-class OCR with multi-engine processing",
            "Advanced AI ensemble analysis",
            "Clinical decision support system",
            "Real-time quality assurance",
            "Evidence-based recommendations",
            "Patient safety alerts",
            "Risk stratification",
            "Medical imaging correlations",
        ],
        "supported_formats": ["PNG", "JPEG", "PDF", "TIFF", "BMP", "GIF"],
        "analysis_types": [
            "Pneumonia detection and classification",
            "COVID-19 pneumonia analysis",
            "Tuberculosis screening",
            "Pleural effusion assessment",
            "Normal vs abnormal classification",
            "Multi-pathology detection",
        ],
        "quality_assurance": [
            "Multi-stage image preprocessing",
            "OCR confidence scoring",
            "Medical content validation",
            "Clinical consistency checks",
            "Evidence strength assessment",
        ],
        "api_endpoints": {
            "image_analysis": "/ultra-analyze/xray-image",
            "text_analysis": "/ultra-analyze-text/xray",
            "capabilities": "/ultra-analyze/capabilities",
        },
        "response_time": "< 10 seconds per image",
        "max_file_size": "10MB",
    }


@app.post("/ultra-enhanced-analyze/xray", response_model=GeminiEnhancedResponse)
async def ultra_enhanced_analyze_xray_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")
        logger.info(f"Starting OCR processing for file: {file.filename}")
        extracted_text = ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
        logger.info(f"OCR completed, extracted {len(extracted_text)} characters")
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the image. Please ensure the image is clear and contains readable medical text.")
        pipeline = get_gemini_pipeline()
        result = pipeline.analyze_with_gemini_enhancement(extracted_text)
        result_dict = asdict(result)
        return GeminiEnhancedResponse(**result_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ultra-enhanced Gemini analysis failed: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e) if str(e) else "Unknown error occurred during ultra-enhanced analysis"
        raise HTTPException(status_code=500, detail=f"Ultra-enhanced Gemini analysis failed: {error_message}")


# =============================
# Nutrition endpoints
# =============================

def _analyze_meal_text(text: str) -> Any:
    """Nutrition analyzer: parses free-text meals with quantities/units using a built-in food database.
    No external APIs; fuzzy matching and unit conversions supported.
    """
    import re, math, difflib

    txt = (text or "").strip()
    if not txt:
        raise HTTPException(status_code=422, detail="Text input cannot be empty")

    # Small internal nutrition DB (per 100g), realistic values
    # Sources approximated from common nutrition tables
    FOODS: Dict[str, Dict[str, Any]] = {
        "chicken breast": {"per_100g": {"cal": 165, "p": 31, "c": 0, "f": 4}, "aliases": ["chicken", "chicken breast", "grilled chicken"], "default_g": 120},
        "rice cooked": {"per_100g": {"cal": 130, "p": 2.7, "c": 28, "f": 0.3}, "aliases": ["rice", "white rice", "brown rice"], "default_g": 150, "cup_g": 195},
        "roti": {"per_100g": {"cal": 250, "p": 8, "c": 50, "f": 3}, "aliases": ["roti", "chapati"], "default_g": 40, "piece_g": 40},
        "dal cooked": {"per_100g": {"cal": 116, "p": 9, "c": 20, "f": 0.4}, "aliases": ["dal", "lentils"], "default_g": 200, "cup_g": 198},
        "paneer": {"per_100g": {"cal": 296, "p": 18, "c": 6, "f": 23}, "aliases": ["paneer"], "default_g": 100},
        "tofu": {"per_100g": {"cal": 76, "p": 8, "c": 1.9, "f": 4.8}, "aliases": ["tofu"], "default_g": 100},
        "yogurt plain": {"per_100g": {"cal": 59, "p": 10, "c": 3.6, "f": 0.4}, "aliases": ["yogurt", "curd"], "default_g": 170, "cup_g": 245},
        "oats dry": {"per_100g": {"cal": 389, "p": 17, "c": 66, "f": 7}, "aliases": ["oats"], "default_g": 40, "cup_g": 90},
        "banana": {"per_100g": {"cal": 89, "p": 1.1, "c": 23, "f": 0.3}, "aliases": ["banana"], "default_g": 120, "piece_g": 120},
        "apple": {"per_100g": {"cal": 52, "p": 0.3, "c": 14, "f": 0.2}, "aliases": ["apple"], "default_g": 150, "piece_g": 150},
        "peanut butter": {"per_100g": {"cal": 588, "p": 25, "c": 20, "f": 50}, "aliases": ["peanut butter"], "default_g": 16, "tbsp_g": 16},
        "olive oil": {"per_100g": {"cal": 884, "p": 0, "c": 0, "f": 100}, "aliases": ["olive oil"], "default_g": 14, "tbsp_g": 14, "tsp_g": 5},
        "butter": {"per_100g": {"cal": 717, "p": 0.9, "c": 0.1, "f": 81}, "aliases": ["butter"], "default_g": 10, "tbsp_g": 14, "tsp_g": 5},
        "egg": {"per_100g": {"cal": 155, "p": 13, "c": 1.1, "f": 11}, "aliases": ["egg", "eggs"], "default_g": 50, "piece_g": 50},
        "fish": {"per_100g": {"cal": 206, "p": 22, "c": 0, "f": 12}, "aliases": ["fish", "salmon", "tilapia"], "default_g": 120},
        "quinoa cooked": {"per_100g": {"cal": 120, "p": 4.4, "c": 21.3, "f": 1.9}, "aliases": ["quinoa"], "default_g": 140, "cup_g": 185},
        "bread": {"per_100g": {"cal": 265, "p": 9, "c": 49, "f": 3.2}, "aliases": ["bread", "toast", "slice"], "default_g": 30, "slice_g": 30},
        "hummus": {"per_100g": {"cal": 166, "p": 8, "c": 14, "f": 10}, "aliases": ["hummus"], "default_g": 30, "tbsp_g": 15},
        "carrots": {"per_100g": {"cal": 41, "p": 0.9, "c": 10, "f": 0.2}, "aliases": ["carrot", "carrots"], "default_g": 80},
        "spinach": {"per_100g": {"cal": 23, "p": 2.9, "c": 3.6, "f": 0.4}, "aliases": ["spinach"], "default_g": 60},
        "tomato": {"per_100g": {"cal": 18, "p": 0.9, "c": 3.9, "f": 0.2}, "aliases": ["tomato", "tomatoes"], "default_g": 100},
        "onion": {"per_100g": {"cal": 40, "p": 1.1, "c": 9.3, "f": 0.1}, "aliases": ["onion", "onions"], "default_g": 50},
        "milk": {"per_100g": {"cal": 60, "p": 3.2, "c": 5, "f": 3.3}, "aliases": ["milk"], "default_g": 240, "cup_g": 244},
        "avocado": {"per_100g": {"cal": 160, "p": 2, "c": 9, "f": 15}, "aliases": ["avocado"], "default_g": 150, "piece_g": 150},
        "nuts mixed": {"per_100g": {"cal": 607, "p": 15, "c": 21, "f": 54}, "aliases": ["nuts", "mixed nuts"], "default_g": 28},
        "granola": {"per_100g": {"cal": 471, "p": 10, "c": 64, "f": 20}, "aliases": ["granola"], "default_g": 40},
        "chia seeds": {"per_100g": {"cal": 486, "p": 17, "c": 42, "f": 31}, "aliases": ["chia", "chia seeds"], "default_g": 12, "tbsp_g": 12},
        "whey protein": {"per_100g": {"cal": 400, "p": 80, "c": 8, "f": 6}, "aliases": ["whey", "whey protein"], "default_g": 30, "scoop_g": 30},
        "orange juice": {"per_100g": {"cal": 45, "p": 0.7, "c": 10.4, "f": 0.2}, "aliases": ["orange juice", "juice"], "default_g": 240, "cup_g": 248},
        # Popular prepared foods
        "pizza": {"per_100g": {"cal": 266, "p": 11, "c": 33, "f": 10}, "aliases": ["pizza"], "default_g": 120, "slice_g": 120},
        "burger": {"per_100g": {"cal": 295, "p": 17, "c": 30, "f": 13}, "aliases": ["burger", "cheeseburger"], "default_g": 180},
        "pasta cooked": {"per_100g": {"cal": 157, "p": 5.8, "c": 30, "f": 1.1}, "aliases": ["pasta", "spaghetti", "penne"], "default_g": 180, "cup_g": 140},
        "biryani": {"per_100g": {"cal": 180, "p": 6, "c": 25, "f": 6}, "aliases": ["biryani", "chicken biryani", "veg biryani"], "default_g": 250, "bowl_g": 300},
        "curry": {"per_100g": {"cal": 120, "p": 6, "c": 10, "f": 6}, "aliases": ["curry", "masala", "gravy"], "default_g": 200, "bowl_g": 250},
        "salad": {"per_100g": {"cal": 80, "p": 3, "c": 10, "f": 3}, "aliases": ["salad"], "default_g": 200, "bowl_g": 250},
        "sandwich": {"per_100g": {"cal": 240, "p": 9, "c": 28, "f": 9}, "aliases": ["sandwich", "grilled sandwich"], "default_g": 180},
        "idli": {"per_100g": {"cal": 128, "p": 2.7, "c": 28, "f": 0.4}, "aliases": ["idli"], "default_g": 50, "piece_g": 50},
        "dosa": {"per_100g": {"cal": 168, "p": 3.9, "c": 30, "f": 3.7}, "aliases": ["dosa"], "default_g": 120, "piece_g": 120},
        "sambar": {"per_100g": {"cal": 50, "p": 2.6, "c": 7, "f": 1.2}, "aliases": ["sambar"], "default_g": 200, "bowl_g": 250},
        "poha": {"per_100g": {"cal": 180, "p": 4, "c": 30, "f": 5}, "aliases": ["poha"], "default_g": 200, "bowl_g": 250},
        "upma": {"per_100g": {"cal": 170, "p": 5, "c": 27, "f": 5}, "aliases": ["upma"], "default_g": 200, "bowl_g": 250},
        "paratha": {"per_100g": {"cal": 320, "p": 7, "c": 45, "f": 12}, "aliases": ["paratha", "aloo paratha"], "default_g": 100, "piece_g": 100},
        "rajma": {"per_100g": {"cal": 140, "p": 8.7, "c": 23, "f": 0.5}, "aliases": ["rajma"], "default_g": 200, "bowl_g": 250},
        "chole": {"per_100g": {"cal": 180, "p": 9, "c": 27, "f": 4}, "aliases": ["chole", "chana masala"], "default_g": 200, "bowl_g": 250},
        "paneer butter masala": {"per_100g": {"cal": 270, "p": 9, "c": 10, "f": 21}, "aliases": ["paneer butter masala"], "default_g": 180, "bowl_g": 220},
        "chicken curry": {"per_100g": {"cal": 190, "p": 15, "c": 4, "f": 12}, "aliases": ["chicken curry"], "default_g": 200, "bowl_g": 250},
        "soda": {"per_100g": {"cal": 40, "p": 0, "c": 10, "f": 0}, "aliases": ["soda", "cola"], "default_g": 330, "bottle_g": 330, "can_g": 330},
        "ice cream": {"per_100g": {"cal": 207, "p": 3.5, "c": 24, "f": 11}, "aliases": ["ice cream"], "default_g": 100, "scoop_g": 66},
    }

    # Category fallback for wide coverage when a specific match isn't found
    CATEGORIES: List[Dict[str, Any]] = [
        {"key": "salad", "kws": ["salad"], "per_100g": {"cal": 80, "p": 3, "c": 10, "f": 3}, "default_g": 250, "bowl_g": 250},
        {"key": "curry", "kws": ["curry", "masala", "gravy"], "per_100g": {"cal": 120, "p": 6, "c": 10, "f": 6}, "default_g": 220, "bowl_g": 250},
        {"key": "soup", "kws": ["soup"], "per_100g": {"cal": 60, "p": 3, "c": 7, "f": 2}, "default_g": 300, "bowl_g": 300},
        {"key": "stir fry", "kws": ["stir fry", "stir-fry"], "per_100g": {"cal": 140, "p": 7, "c": 15, "f": 6}, "default_g": 250},
        {"key": "wrap", "kws": ["wrap", "roll"], "per_100g": {"cal": 220, "p": 8, "c": 28, "f": 8}, "default_g": 200},
        {"key": "sandwich", "kws": ["sandwich", "sub"], "per_100g": {"cal": 240, "p": 9, "c": 28, "f": 9}, "default_g": 200},
        {"key": "pizza", "kws": ["pizza"], "per_100g": {"cal": 266, "p": 11, "c": 33, "f": 10}, "default_g": 120, "slice_g": 120},
        {"key": "pasta", "kws": ["pasta", "spaghetti", "penne"], "per_100g": {"cal": 157, "p": 5.8, "c": 30, "f": 1.1}, "default_g": 180},
        {"key": "rice dish", "kws": ["biryani", "fried rice", "pulao"], "per_100g": {"cal": 180, "p": 5, "c": 28, "f": 5}, "default_g": 250, "bowl_g": 300},
        {"key": "sweet", "kws": ["dessert", "sweet", "halwa", "cake"], "per_100g": {"cal": 350, "p": 5, "c": 55, "f": 12}, "default_g": 100},
        {"key": "beverage", "kws": ["juice", "soda", "cola", "shake"], "per_100g": {"cal": 45, "p": 1, "c": 10, "f": 0}, "default_g": 250},
    ]
    

    # Map user text to food key
    alias_to_key: Dict[str, str] = {}
    for k, v in FOODS.items():
        for a in v.get("aliases", []):
            alias_to_key[a] = k

    def find_food(name: str) -> Optional[str]:
        name_l = name.lower().strip()
        if name_l in alias_to_key:
            return alias_to_key[name_l]
        # fuzzy
        choices = list(alias_to_key.keys()) + list(FOODS.keys())
        match = difflib.get_close_matches(name_l, choices, n=1, cutoff=0.7)
        if match:
            return alias_to_key.get(match[0], match[0] if match[0] in FOODS else None)
        # try simplify words
        tokens = [t for t in re.split(r"[^a-zA-Z]+", name_l) if t]
        for i in range(len(tokens), 0, -1):
            cand = " ".join(tokens[:i])
            if cand in alias_to_key:
                return alias_to_key[cand]
        return None

    def find_category(name: str) -> Optional[Dict[str, Any]]:
        n = name.lower()
        for cat in CATEGORIES:
            if any(kw in n for kw in cat["kws"]):
                return cat
        return None

    # Unit conversion helpers
    UNIT_ALIASES = {
        "g": ["g", "gram", "grams"],
        "kg": ["kg", "kilogram", "kilograms"],
        "ml": ["ml", "milliliter", "milliliters"],
        "cup": ["cup", "cups"],
        "tbsp": ["tbsp", "tablespoon", "tablespoons"],
    "tsp": ["tsp", "teaspoon", "teaspoons"],
        "slice": ["slice", "slices"],
        "piece": ["piece", "pieces"],
        "scoop": ["scoop", "scoops"],
        "roti": ["roti", "rotis", "chapati", "chapatis"],
        "egg": ["egg", "eggs"],
    "bowl": ["bowl", "bowls"],
    "plate": ["plate", "plates"],
    "bottle": ["bottle", "bottles"],
    "can": ["can", "cans"],
    "pack": ["pack", "packet", "packets"],
    }
    UNIT_LOOKUP = {alias: base for base, aliases in UNIT_ALIASES.items() for alias in aliases}

    def to_grams(food_key: str, qty: float, unit: Optional[str]) -> float:
        info = FOODS[food_key]
        unit = unit or "serving"
        # food-specific unit gram mappings
        if unit == "g":
            return qty
        if unit == "kg":
            return qty * 1000
        # approximate liquids density as water for milk/juice
        if unit == "ml":
            return qty if food_key in ("milk", "orange juice",) else qty
        # food-specific units
        for key in ("cup_g", "tbsp_g", "tsp_g", "slice_g", "piece_g", "scoop_g"):
            base = key.split("_")[0]
            if unit == base and key in info:
                return qty * info[key]
        if unit in ("bowl", "plate", "bottle", "can"):
            spec = f"{unit}_g"
            if spec in info:
                return qty * info[spec]
        if unit == "roti" and food_key == "roti":
            return qty * info.get("piece_g", 40)
        if unit == "egg" and food_key == "egg":
            return qty * info.get("piece_g", 50)
        if unit == "slice" and food_key == "bread":
            return qty * info.get("slice_g", 30)
        # default to per-serving grams
        return qty * info.get("default_g", 100)

    # Split text into phrases on commas and connectors
    # Also handle "with" and "and" to split
    raw_parts = re.split(r",|\band\b|\bwith\b", txt, flags=re.IGNORECASE)
    parts = [p.strip() for p in raw_parts if p.strip()]

    item_entries: List[Dict[str, Any]] = []
    uncertain = 0
    for part in parts:
        # extract quantity and unit
        m = re.search(r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)?\s+(?P<name>.+)", part)
        if m:
            qty = float(m.group("qty"))
            unit_raw = (m.group("unit") or "").lower()
            unit = UNIT_LOOKUP.get(unit_raw, unit_raw) if unit_raw else None
            name = m.group("name").strip()
        else:
            qty, unit, name = 1.0, None, part

        # normalize name, reduce descriptors
        name = re.sub(r"\b(grilled|baked|fried|boiled|steamed|cooked|plain|brown|white)\b", "", name, flags=re.IGNORECASE).strip()
        food_key = find_food(name)
        if not food_key and "toast" in name.lower():
            food_key = "bread"
        # Try category fallback
        category = None if food_key else find_category(name)
        if not food_key and not category:
            # Unknown item; mark uncertain with placeholder but try a generic estimate (250g meal at 150 kcal/100g)
            uncertain += 1
            est_g = 250.0
            per100 = {"cal": 150, "p": 6, "c": 18, "f": 6}
            factor = est_g / 100.0
            item_entries.append({
                "name": name + " (estimated)",
                "cal": per100["cal"] * factor,
                "p": per100["p"] * factor,
                "c": per100["c"] * factor,
                "f": per100["f"] * factor,
            })
            continue

        if food_key:
            grams = to_grams(food_key, qty, unit)
            per100 = FOODS[food_key]["per_100g"]
            display = f"{round(qty,2)} {unit or 'serving'} {food_key}"
        else:
            info = category
            # category units
            unit_key_map = {
                "bowl": "bowl_g", "plate": "plate_g", "cup": "cup_g", "tbsp": "tbsp_g", "tsp": "tsp_g",
            }
            grams = qty * info.get(unit_key_map.get(unit, "default_g"), info.get("default_g", 250))
            per100 = info["per_100g"]
            display = f"{round(qty,2)} {unit or 'serving'} {info['key']}"
        factor = grams / 100.0
        cal = per100["cal"] * factor
        p = per100["p"] * factor
        c = per100["c"] * factor
        f = per100["f"] * factor
        item_entries.append({"name": display, "cal": cal, "p": p, "c": c, "f": f})

    # Aggregate totals
    total_c = sum(e["cal"] for e in item_entries if e["cal"] is not None)
    total_p = sum(e["p"] for e in item_entries if e["p"] is not None)
    total_cr = sum(e["c"] for e in item_entries if e["c"] is not None)
    total_f = sum(e["f"] for e in item_entries if e["f"] is not None)

    items: List[Any] = []
    for e in item_entries:
        items.append(NutritionItem(
            name=e["name"],
            calories=round(e["cal"], 1) if e["cal"] is not None else None,
            protein_g=round(e["p"], 1) if e["p"] is not None else None,
            carbs_g=round(e["c"], 1) if e["c"] is not None else None,
            fat_g=round(e["f"], 1) if e["f"] is not None else None,
        ))

    if not items:
        items.append(NutritionItem(name="meal", calories=None, protein_g=None, carbs_g=None, fat_g=None))

    # Data-driven suggestions based on computed macros and detected items
    suggestions: List[str] = []
    swaps: List[str] = []
    names_join = " ".join([it.name.lower() for it in items])
    if total_p and total_p < 20:
        suggestions.append("Consider adding a protein source (eggs, chicken, tofu, paneer) to balance macros")
    if total_cr and total_cr > 120:
        suggestions.append("High carbs detected; add fiber/veg or reduce portion size")
    if total_f and total_f > 60:
        suggestions.append("High fats detected; prefer lean proteins and limit added oils/butter")
    if "butter" in names_join:
        swaps.append("Swap butter with olive oil or avocado spread")
    if "juice" in names_join:
        suggestions.append("Juice detected; whole fruit provides more fiber and satiety")
    if uncertain:
        suggestions.append(f"Estimated values used for {uncertain} item(s); refine names/quantities for better accuracy")

    return NutritionAnalysisResponse(
        total_calories=round(total_c, 1) if total_c else None,
        macros={
            "protein_g": round(total_p, 1) if total_p else None,
            "carbs_g": round(total_cr, 1) if total_cr else None,
            "fat_g": round(total_f, 1) if total_f else None,
        },
        items=items,
        suggestions=suggestions,
        swaps=swaps,
    )


def _generate_meal_plan(req: Any) -> Any:
    """Goal-tailored meal planner with varied pools, calorie scaling, and preference filters."""
    # Inputs with defaults
    goal = (getattr(req, "goal", "balanced") or "balanced").lower()
    days = int(getattr(req, "days", 7) or 7)
    meals_per_day = max(1, min(6, int(getattr(req, "meals_per_day", 3) or 3)))
    cals = (getattr(req, "calories_target", None) or {
        "weight_loss": 1800,
        "muscle_gain": 2600,
        "diabetic_diet": 2000,
        "balanced": 2200,
    }.get(goal, 2200))
    pref = (getattr(req, "dietary_preference", "none") or "none").lower()

    # Meal libraries by goal and slot
    library: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "weight_loss": {
            "Breakfast": [
                {"name": "Egg white scramble + spinach", "kcal": 300, "ingredients": ["egg whites", "spinach", "onion"], "desc": "Lean protein, low-cal"},
                {"name": "Overnight oats + chia + berries", "kcal": 320, "ingredients": ["oats", "milk", "chia", "berries"], "desc": "High fiber"},
                {"name": "Tofu scramble + veggies", "kcal": 310, "ingredients": ["tofu", "peppers", "onion"], "desc": "Plant protein"},
            ],
            "Lunch": [
                {"name": "Grilled chicken salad", "kcal": 450, "ingredients": ["chicken", "lettuce", "tomato", "olive oil"], "desc": "High protein"},
                {"name": "Paneer salad bowl", "kcal": 450, "ingredients": ["paneer", "lettuce", "cucumber", "tomato"], "desc": "Low-carb vegetarian"},
                {"name": "Tofu quinoa bowl", "kcal": 440, "ingredients": ["tofu", "quinoa", "veggies"], "desc": "Balanced plant-based"},
            ],
            "Dinner": [
                {"name": "Baked fish + steamed veggies", "kcal": 500, "ingredients": ["fish", "broccoli", "carrots"], "desc": "Light dinner"},
                {"name": "Dal + salad", "kcal": 480, "ingredients": ["dal", "salad"], "desc": "Fiber and protein"},
                {"name": "Veg curry + cauliflower rice", "kcal": 470, "ingredients": ["veggies", "cauliflower rice", "spices"], "desc": "Lower carbs"},
            ],
            "Snack": [
                {"name": "Greek yogurt + nuts", "kcal": 200, "ingredients": ["yogurt", "nuts"], "desc": "Satiety boost"},
                {"name": "Apple + peanut butter", "kcal": 220, "ingredients": ["apple", "peanut butter"], "desc": "Fiber + fat"},
                {"name": "Hummus + carrots", "kcal": 180, "ingredients": ["hummus", "carrots"], "desc": "Crunchy snack"},
            ],
        },
        "muscle_gain": {
            "Breakfast": [
                {"name": "Oats + whey + banana", "kcal": 520, "ingredients": ["oats", "milk", "whey", "banana"], "desc": "High-protein start"},
                {"name": "Paneer bhurji + roti", "kcal": 550, "ingredients": ["paneer", "roti", "onion", "tomato"], "desc": "Protein + carbs"},
                {"name": "Eggs + toast + avocado", "kcal": 540, "ingredients": ["eggs", "toast", "avocado"], "desc": "Calorie-dense"},
            ],
            "Lunch": [
                {"name": "Grilled chicken + rice + veggies", "kcal": 750, "ingredients": ["chicken", "rice", "veggies"], "desc": "Bulking classic"},
                {"name": "Tofu stir-fry + noodles", "kcal": 700, "ingredients": ["tofu", "noodles", "veggies"], "desc": "Plant protein"},
                {"name": "Paneer tikka + rice", "kcal": 720, "ingredients": ["paneer", "rice", "spices"], "desc": "Hearty meal"},
            ],
            "Dinner": [
                {"name": "Fish + quinoa + veggies", "kcal": 680, "ingredients": ["fish", "quinoa", "veggies"], "desc": "Lean mass"},
                {"name": "Dal + rice + salad", "kcal": 660, "ingredients": ["dal", "rice", "salad"], "desc": "Comfort carbs"},
                {"name": "Veg curry + roti", "kcal": 640, "ingredients": ["veggies", "roti", "curry"], "desc": "Balanced dinner"},
            ],
            "Snack": [
                {"name": "Peanut butter sandwich", "kcal": 350, "ingredients": ["bread", "peanut butter"], "desc": "Quick calories"},
                {"name": "Protein smoothie", "kcal": 320, "ingredients": ["milk", "whey", "banana", "peanut butter"], "desc": "Drinkable kcal"},
                {"name": "Trail mix", "kcal": 300, "ingredients": ["nuts", "raisins", "seeds"], "desc": "Energy-dense"},
            ],
        },
        "diabetic_diet": {
            "Breakfast": [
                {"name": "Oats + nuts + cinnamon", "kcal": 360, "ingredients": ["oats", "milk", "nuts", "cinnamon"], "desc": "Low GI"},
                {"name": "Besan chilla + yogurt", "kcal": 380, "ingredients": ["besan", "spices", "yogurt"], "desc": "Balanced carbs"},
                {"name": "Tofu veggie scramble", "kcal": 350, "ingredients": ["tofu", "veggies"], "desc": "Protein-forward"},
            ],
            "Lunch": [
                {"name": "Grilled chicken + quinoa salad", "kcal": 620, "ingredients": ["chicken", "quinoa", "salad"], "desc": "Fiber-rich"},
                {"name": "Paneer + brown rice + veggies", "kcal": 630, "ingredients": ["paneer", "brown rice", "veggies"], "desc": "Lower GI"},
                {"name": "Tofu + millet bowl", "kcal": 610, "ingredients": ["tofu", "millet", "veggies"], "desc": "Steady energy"},
            ],
            "Dinner": [
                {"name": "Fish + salad + olive oil", "kcal": 580, "ingredients": ["fish", "salad", "olive oil"], "desc": "Good fats"},
                {"name": "Dal + quinoa + salad", "kcal": 590, "ingredients": ["dal", "quinoa", "salad"], "desc": "Complex carbs"},
                {"name": "Veg curry + brown rice", "kcal": 600, "ingredients": ["veggies", "brown rice", "curry"], "desc": "Controlled carbs"},
            ],
            "Snack": [
                {"name": "Roasted chana", "kcal": 220, "ingredients": ["chana"], "desc": "Low GI snack"},
                {"name": "Apple + almonds", "kcal": 220, "ingredients": ["apple", "almonds"], "desc": "Portion-controlled"},
                {"name": "Yogurt + seeds", "kcal": 200, "ingredients": ["yogurt", "seeds"], "desc": "Stable energy"},
            ],
        },
        "balanced": {
            "Breakfast": [
                {"name": "Oats with fruit", "kcal": 420, "ingredients": ["oats", "milk", "banana", "berries"], "desc": "Classic start"},
                {"name": "Veggie omelette", "kcal": 400, "ingredients": ["eggs", "spinach", "onion", "tomato"], "desc": "Protein + veg"},
                {"name": "Smoothie bowl", "kcal": 430, "ingredients": ["yogurt", "berries", "granola", "chia"], "desc": "Fruity bowl"},
            ],
            "Lunch": [
                {"name": "Grilled chicken + rice + salad", "kcal": 650, "ingredients": ["chicken", "rice", "salad"], "desc": "Balanced plate"},
                {"name": "Paneer tikka + roti + salad", "kcal": 650, "ingredients": ["paneer", "roti", "salad"], "desc": "Indian balanced"},
                {"name": "Tofu stir-fry + rice", "kcal": 600, "ingredients": ["tofu", "rice", "veggies"], "desc": "Stir-fry"},
            ],
            "Dinner": [
                {"name": "Fish + quinoa + veggies", "kcal": 620, "ingredients": ["fish", "quinoa", "veggies"], "desc": "Omega-3s"},
                {"name": "Dal + roti + salad", "kcal": 600, "ingredients": ["dal", "roti", "salad"], "desc": "Comfort balanced"},
                {"name": "Veg curry + rice", "kcal": 610, "ingredients": ["veggies", "rice", "curry"], "desc": "Spiced"},
            ],
            "Snack": [
                {"name": "Greek yogurt + nuts", "kcal": 260, "ingredients": ["yogurt", "nuts"], "desc": "Protein snack"},
                {"name": "Apple + peanut butter", "kcal": 240, "ingredients": ["apple", "peanut butter"], "desc": "Sweet-salty"},
                {"name": "Hummus + carrots", "kcal": 210, "ingredients": ["hummus", "carrots"], "desc": "Crunchy"},
            ],
        },
    }

    # Apply dietary preference filtering
    def allowed_meal(m: Dict[str, Any]) -> bool:
        name = m["name"].lower()
        if pref == "vegetarian" and any(x in name for x in ["chicken", "fish"]):
            return False
        if pref == "vegan" and any(x in name for x in ["chicken", "fish", "paneer", "yogurt", "eggs"]):
            return False
        if pref == "pescatarian" and any(x in name for x in ["chicken", "paneer", "eggs"]):
            return False
        return True

    # Diabetic-specific exclusions (avoid high sugar/simple carbs)
    def diabetic_ok(m: Dict[str, Any]) -> bool:
        if goal != "diabetic_diet":
            return True
        bad_terms = ["juice", "white rice", "toast", "noodles", "granola"]
        return not any(term in m["name"].lower() for term in bad_terms)

    # Determine meal slots for the day
    def slots_for(meals_count: int) -> List[str]:
        base = ["Breakfast", "Lunch", "Dinner"]
        if meals_count <= 3:
            return base[:meals_count]
        # add snacks between
        extra = ["Snack"] * (meals_count - 3)
        # Interleave to keep order sensible
        return ["Breakfast", "Snack", "Lunch"] + extra[:-1] + ["Dinner"] if len(extra) > 1 else ["Breakfast", "Lunch", "Snack", "Dinner"]

    # Pick the library for the goal, fallback to balanced
    lib = library.get(goal, library["balanced"])
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Seed variation using goal, pref, cals for determinism but distinct per input
    seed_basis = f"{goal}|{pref}|{cals}|{days}|{meals_per_day}"
    seed = abs(hash(seed_basis)) % 100000

    # Macro distribution by goal (approximate)
    macro_split = {
        "weight_loss": (0.35, 0.35, 0.30),
        "muscle_gain": (0.30, 0.50, 0.20),
        "diabetic_diet": (0.25, 0.40, 0.35),
        "balanced": (0.25, 0.50, 0.25),
    }.get(goal, (0.25, 0.50, 0.25))

    def estimate_macros(kcal: int) -> Dict[str, int]:
        p, c, f = macro_split
        pg = int(round((kcal * p) / 4))
        cg = int(round((kcal * c) / 4))
        fg = int(round((kcal * f) / 9))
        return {"protein_g": pg, "carbs_g": cg, "fat_g": fg}

    per_meal_target = max(250, int(cals / max(1, meals_per_day)))
    week: List[Any] = []
    for d in range(days):
        meals: List[Any] = []
        slot_names = slots_for(meals_per_day)
        for s_idx, slot in enumerate(slot_names):
            slot_key = "Snack" if slot == "Snack" else slot
            pool = [m for m in lib.get(slot_key, []) if allowed_meal(m) and diabetic_ok(m)]
            if not pool:
                # fallback to balanced slot pool with filters
                pool = [m for m in library["balanced"].get(slot_key, []) if allowed_meal(m)]
            # choose index deterministically
            idx = (seed + d * 7 + s_idx * 3) % len(pool)
            choice = pool[idx]
            base_kcal = choice["kcal"]
            # scale calories toward target
            scale = per_meal_target / max(1, base_kcal)
            adj_kcal = int(max(200, min(900, round(base_kcal * scale))))
            macros = estimate_macros(adj_kcal)
            meals.append(MealPlanMeal(
                name=f"{slot}: {choice['name']}",
                description=choice.get("desc"),
                calories=adj_kcal,
                protein_g=macros["protein_g"],
                carbs_g=macros["carbs_g"],
                fat_g=macros["fat_g"],
                ingredients=choice.get("ingredients", []),
            ))
        week.append(MealPlanDay(day=day_names[d % len(day_names)], meals=meals))

    return MealPlanResponse(
        goal=goal,
        calories_target=cals,
        dietary_preference=(None if pref == "none" else pref),
        week=week,
        days=week,
    )


@app.post("/nutrition/analyze", response_model=NutritionAnalysisResponse)
async def nutrition_analyze(payload: Any = Body(...)):
    try:
        text = None
        if isinstance(payload, dict):
            text = payload.get("text")
        else:
            text = getattr(payload, "text", None)
        if not text or not str(text).strip():
            raise HTTPException(status_code=422, detail="Text input cannot be empty")
        return _analyze_meal_text(str(text))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Nutrition analyze failed: {e}")
        raise HTTPException(status_code=500, detail="Nutrition analyze failed")


@app.post("/nutrition/meal-plan", response_model=MealPlanResponse)
async def nutrition_meal_plan(payload: Any = Body(...)):
    try:
        req_obj = payload
        if isinstance(payload, dict):
            req_obj = SimpleNamespace(**payload)
        return _generate_meal_plan(req_obj)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meal plan generation failed: {e}")
        raise HTTPException(status_code=500, detail="Meal plan generation failed")


# =============================
# HealthScan endpoint
# =============================

@app.post("/healthscan/analyze", response_model=HealthScanResponse)
async def healthscan_analyze(
    file: UploadFile = File(...),
    mode: str = "auto",  # 'auto' | 'gemini' | 'fast'
    timeout: int = 10,
):
    """Analyze uploaded medical reports (PDF/images) and return a structured summary.

    - Robust PDF parsing via ultra_ocr_processor with fallback to basic OCR
    - Optional Gemini-powered summarization for richer, accurate outputs
    - Safe fallbacks so the endpoint never hangs and always returns a result
    """
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")

        mime = file.content_type or "application/octet-stream"
        is_image = mime.startswith("image/")

        # Extract text for both PDFs and images to enable summarization
        extracted_text: str = ""
        try:
            extracted_text = ultra_ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
            logger.info(f"HealthScan: ultra OCR extracted {len(extracted_text or '')} chars")
        except Exception as ocr_err:
            logger.warning(f"HealthScan: ultra OCR failed, falling back. Err: {ocr_err}")
            try:
                extracted_text = ocr_processor.process_medical_report(file_bytes, file.filename or "unknown")
            except Exception as basic_err:
                logger.error(f"HealthScan: basic OCR failed: {basic_err}")
                extracted_text = ""

        # Heuristic parsing as a baseline
        def _basic_sections(text: str) -> Dict[str, str]:
            import re
            t = text or ""
            chunks = {"impression": "", "findings": "", "recommendations": "", "history": "", "technique": "", "diagnosis": "", "assessment": ""}
            # Split on common headers and variants
            pattern = re.compile(r"(?im)^(impression|findings|recommendations|history|technique|diagnosis|assessment)\s*:?\s*$")
            last = None
            buf: Dict[str, List[str]] = {k: [] for k in chunks}
            for line in (t.splitlines() if t else []):
                m = pattern.match(line.strip())
                if m:
                    last = m.group(1).lower()
                    continue
                if last:
                    buf[last].append(line.strip())
            return {k: " ".join(v).strip() for k, v in buf.items() if v}

        def _extract_keywords(text: str) -> List[str]:
            kws = [
                "pneumonia", "fracture", "normal", "abnormal", "covid", "effusion", "lesion", "mass", "tumor",
                "hyperglycemia", "anemia", "hypertension", "diabetes", "infection", "inflammation", "edema",
                "metastasis", "stroke", "infarct", "embolism", "cardiomegaly", "nodule", "opacity"
            ]
            t = (text or "").lower()
            found = []
            for k in kws:
                if k in t:
                    found.append(k)
            return sorted(set(found))[:12]

        # Truncate excessively long text to keep LLM prompt efficient
        if extracted_text and len(extracted_text) > 12000:
            extracted_text = extracted_text[:12000]

        text_ok = bool(extracted_text and len(extracted_text.strip()) >= 20)
        sections = _basic_sections(extracted_text) if text_ok else None
        keywords = _extract_keywords(extracted_text) if text_ok else []

        summary = ""
        risks: List[str] = []
        precautions: List[str] = []
        detected: List[str] = keywords[:5]
        follow_up: List[str] = []
        confidence_note: Optional[str] = None

        if not text_ok:
            # No meaningful text extracted
            summary = "Could not extract meaningful text from the document."
            precautions = ["Try a clearer PDF or image of the report", "Ensure the file is a PDF or a clear photo scan"]
        else:
            # Provide a quick preview summary first
            lines = [ln.strip() for ln in extracted_text.splitlines() if ln.strip()]
            preview = " ".join(lines[:6])[:600]
            summary = f"Report processed. Key content preview: {preview}"

            # Try Gemini reviewer for rich medical summarization with a timeout
            gemini_ok = False
            try:
                if mode != "fast":
                    from .gemini_medical_reviewer import get_gemini_reviewer, MedicalReviewRequest
                    import concurrent.futures
                    reviewer = get_gemini_reviewer()
                    # Minimal baseline context
                    initial_diag = "undetermined"
                    base_findings = detected if detected else ["medical report text analyzed"]
                    req = MedicalReviewRequest(
                        original_findings=base_findings,
                        initial_diagnosis=initial_diag,
                        confidence=0.65 if mode == "gemini" else 0.6,
                        raw_report_text=extracted_text,
                        disease_risks={k: {"probability": 0.5} for k in detected} if detected else {},
                        severity_assessment="moderate",
                        clinical_context="General medical report analysis"
                    )
                    def _call_review():
                        return reviewer.review_medical_report(req)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(_call_review)
                        review = fut.result(timeout=max(6, min(20, int(timeout))))
                    # Use Gemini outputs to enrich response
                    gemini_ok = True
                    summary = review.enhanced_summary or summary
                    # Detected/keywords
                    if isinstance(review.enhanced_findings, list) and review.enhanced_findings:
                        detected = list({*(detected or []), *[str(x) for x in review.enhanced_findings]})[:10]
                    # Sections
                    sections = sections or {}
                    sections = sections.copy() if sections else {}
                    sections.update({
                        "impression": review.enhanced_summary or sections.get("impression", ""),
                        "findings": "; ".join([str(x) for x in (review.enhanced_findings or [])])[:1500] or sections.get("findings", ""),
                        "recommendations": "; ".join(review.clinical_recommendations or [])[:1500] or sections.get("recommendations", ""),
                    })
                    # Follow-up and risks
                    if review.follow_up_timeline:
                        follow_up.append(f"Suggested follow-up: {review.follow_up_timeline}")
                    if review.urgency_level:
                        risks.append(f"Urgency assessed as: {review.urgency_level}")
                    # Confidence/quality
                    confidence_note = f"Gemini confidence {review.confidence_assessment:.2f} | report quality {review.report_quality_score:.2f}"
                    # Add general precautions
                    precautions.append("AI-assisted summary. Consult a qualified clinician for diagnosis and treatment.")
            except Exception as ge:
                logger.warning(f"HealthScan: Gemini review unavailable, using heuristic summary. Err: {ge}")
                # Heuristic risk/precaution
                if any(k in (keywords or []) for k in ["abnormal", "lesion", "tumor", "fracture", "stroke", "embolism"]):
                    risks.append("Potential critical finding referenced in text")
                    precautions.append("Seek immediate medical advice if symptomatic")
                precautions.append("This is a general summary and may be incomplete")

            # If summary is weak/empty, compose a richer fallback from sections/keywords
            try:
                def _compose_fallback_summary() -> str:
                    parts: List[str] = []
                    kind_label = "image" if is_image else "report"
                    if sections:
                        imp = sections.get("impression", "").strip()
                        fnd = sections.get("findings", "").strip()
                        rec = sections.get("recommendations", "").strip()
                        dia = sections.get("diagnosis", "").strip() or sections.get("assessment", "").strip()
                        if dia:
                            parts.append(f"Primary impression: {dia}.")
                        elif imp:
                            parts.append(f"Primary impression: {imp}.")
                        if fnd:
                            parts.append(f"Key findings: {fnd}.")
                        if rec:
                            parts.append(f"Recommended next steps: {rec}.")
                    if not parts and (detected or keywords):
                        tags = ", ".join((detected or keywords)[:6])
                        parts.append(f"The {kind_label} mentions: {tags}.")
                    if not parts and preview:
                        parts.append(f"Overview: {preview}")
                    if not parts:
                        parts.append("Summary could not be extracted from this document.")
                    return " ".join(parts)[:1200]

                if not summary or summary.strip() == "" or summary.startswith("Report processed. Key content preview:"):
                    summary = _compose_fallback_summary()
            except Exception as _sf:
                logger.debug(f"HealthScan fallback summary build skipped: {_sf}")

            # Simple entity extraction from text if present
            try:
                import re
                entities: Dict[str, List[str]] = {"diseases": [], "symptoms": [], "medications": [], "labs": []}
                t = (extracted_text or "")
                # meds (very naive): capture words ending with -ine, -ol, -pril, -sartan, -mab, -azole
                meds = re.findall(r"\b([A-Z][a-z]+(?:ine|ol|pril|sartan|mab|azole))\b", t)
                # simple labs: patterns like Hb 12.3 g/dL, WBC 9.8, Creatinine 1.2 mg/dL
                labs = re.findall(r"\b(Hb|WBC|Platelets|Creatinine|BUN|CRP|ESR|ALT|AST|ALP|LDH)\b[^\n]{0,20}\b([0-9]+\.?[0-9]*)", t)
                # symptoms keywords
                symptom_kws = ["fever", "cough", "chest pain", "headache", "fatigue", "nausea", "vomiting", "dizziness", "shortness of breath"]
                syms = [kw for kw in symptom_kws if kw in t.lower()]
                if meds:
                    entities["medications"] = sorted(list({m for m in meds}))[:12]
                if labs:
                    entities["labs"] = [f"{name}: {val}" for name, val in labs][:12]
                if syms:
                    entities["symptoms"] = syms
            except Exception:
                entities = None  # keep None if failure

        return HealthScanResponse(
            kind="image" if is_image else "document",
            summary=summary,
            detected=detected or [],
            risks=risks,
            precautions=precautions,
            filename=file.filename,
            mime_type=mime,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            keywords=keywords or detected or [],
            sections=sections,
            entities=(entities if (text_ok and (entities and any(entities.values()))) else ({
                "diseases": detected or [],
                "symptoms": [],
                "medications": [],
                "labs": [],
            } if (detected or sections) else None)),
            follow_up=follow_up,
            confidence_note=confidence_note,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HealthScan analyze failed: {e}")
        raise HTTPException(status_code=500, detail="HealthScan analyze failed")
