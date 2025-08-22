import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18, resnet50
from transformers import BertForSequenceClassification, BertTokenizer
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_EFFNET = os.path.join(BASE_DIR, "model", "model.pth")
MODEL_PATH_RESNET = os.path.join(BASE_DIR, "model", "resnet18_skin_cancer.pth")
MODEL_PATH_DERMNET_RESNET50 = os.path.join(BASE_DIR, "model", "dermnet_resnet50_full.pth")
MODEL_PATH_XRAY = os.path.join(BASE_DIR, "saved_xray_model")
# Default path for MRI model (now PyTorch by default; backend will still support TF fallback).
MODEL_PATH_BRAIN_TUMOR = os.path.join(BASE_DIR, "model", "brain_tumor_classifier_final.pth")

EFFICIENTNET_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
RESNET18_CLASSES = ['benign', 'malignant', 'suspicious', 'other']
# DermNet ResNet50 classes (23 classes as per the model checkpoint)
DERMNET_RESNET50_CLASSES = [
    'acne_and_rosacea', 'actinic_keratosis_basal_cell_carcinoma', 'atopic_dermatitis', 'bullous_disease',
    'cellulitis_impetigo', 'eczema', 'exanthems_and_drug_eruptions', 'hair_loss_photos_alopecia_and_other_hair_diseases',
    'herpes_hpv_and_other_stds', 'light_diseases_and_disorders_of_pigmentation', 'lupus_and_other_connective_tissue_diseases',
    'melanoma_skin_cancer_nevi_and_moles', 'nail_fungus_and_other_nail_disease', 'poison_ivy_photos_and_other_contact_dermatitis',
    'psoriasis_pictures_lichen_planus_and_related_diseases', 'scabies_lyme_disease_and_other_infestations_and_bites',
    'seborrheic_keratoses_and_other_benign_tumors', 'systemic_disease', 'tinea_ringworm_candidiasis_and_other_fungal_infections',
    'urticaria_hives', 'vascular_tumors', 'vasculitis_photos', 'warts_molluscum_and_other_viral_infections'
]
# X-ray model classes (binary classification)
XRAY_CLASSES = ['normal', 'abnormal']
# Brain tumor MRI classes
BRAIN_TUMOR_CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']

def load_efficientnet():
    try:
        if not os.path.exists(MODEL_PATH_EFFNET):
            logger.error(f"EfficientNet model not found at: {MODEL_PATH_EFFNET}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH_EFFNET}")
        
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(EFFICIENTNET_CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH_EFFNET, map_location='cpu'))
        model.eval()
        logger.info("EfficientNet model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load EfficientNet model: {e}")
        raise

def load_resnet18():
    try:
        if not os.path.exists(MODEL_PATH_RESNET):
            logger.error(f"ResNet18 model not found at: {MODEL_PATH_RESNET}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH_RESNET}")
        
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(RESNET18_CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH_RESNET, map_location='cpu'))
        model.eval()
        logger.info("ResNet18 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load ResNet18 model: {e}")
        raise

def load_dermnet_resnet50():
    try:
        if not os.path.exists(MODEL_PATH_DERMNET_RESNET50):
            logger.error(f"DermNet ResNet50 model not found at: {MODEL_PATH_DERMNET_RESNET50}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH_DERMNET_RESNET50}")
        
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(DERMNET_RESNET50_CLASSES))
        
        # Load the checkpoint
        checkpoint = torch.load(MODEL_PATH_DERMNET_RESNET50, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Load from model_state_dict
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded DermNet ResNet50 model with {checkpoint.get('num_classes', 'unknown')} classes")
            elif 'state_dict' in checkpoint:
                # Load from state_dict
                model.load_state_dict(checkpoint['state_dict'])
                logger.info("Loaded DermNet ResNet50 model from state_dict")
            else:
                # Try to load directly as state_dict
                model.load_state_dict(checkpoint)
                logger.info("Loaded DermNet ResNet50 model directly")
        else:
            # Direct state_dict
            model.load_state_dict(checkpoint)
            logger.info("Loaded DermNet ResNet50 model directly")
        
        model.eval()
        logger.info("DermNet ResNet50 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load DermNet ResNet50 model: {e}")
        raise

def load_xray_model():
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH_XRAY)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH_XRAY)
        model.eval()
        logger.info("X-ray model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load X-ray model: {e}")
        raise

def load_brain_tumor_model():
    """Deprecated direct MRI loader. Use app.main.load_mri_model instead."""
    raise RuntimeError("Use app.main.load_mri_model for MRI loading")
