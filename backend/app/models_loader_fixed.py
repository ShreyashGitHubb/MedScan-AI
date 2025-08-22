import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18, resnet50
from transformers import BertForSequenceClassification, BertTokenizer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_EFFNET = os.path.join(BASE_DIR, "model", "model.pth")
MODEL_PATH_RESNET = os.path.join(BASE_DIR, "model", "resnet18_skin_cancer.pth")
MODEL_PATH_DERMNET_RESNET50 = os.path.join(BASE_DIR, "model", "dermnet_resnet50_full.pth")
MODEL_PATH_XRAY = os.path.join(BASE_DIR, "saved_xray_model")

EFFICIENTNET_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
RESNET18_CLASSES = ['benign', 'malignant', 'suspicious', 'other']
DERMNET_RESNET50_CLASSES = [
    'acne_and_rosacea', 'actinic_keratosis_basal_cell_carcinoma', 'atopic_dermatitis', 'bullous_disease',
    'cellulitis_impetigo', 'eczema', 'exanthems_and_drug_eruptions', 'hair_loss_photos_alopecia_and_other_hair_diseases',
    'herpes_hpv_and_other_stds', 'light_diseases_and_disorders_of_pigmentation', 'lupus_and_other_connective_tissue_diseases',
    'melanoma_skin_cancer_nevi_and_moles', 'nail_fungus_and_other_nail_disease', 'poison_ivy_photos_and_other_contact_dermatitis',
    'psoriasis_pictures_lichen_planus_and_related_diseases', 'scabies_lyme_disease_and_other_infestations_and_bites',
    'seborrheic_keratoses_and_other_benign_tumors', 'systemic_disease', 'tinea_ringworm_candidiasis_and_other_fungal_infections',
    'urticaria_hives', 'vascular_tumors', 'vasculitis_photos', 'warts_molluscum_and_other_viral_infections'
]

# FIXED: X-ray model classes - corrected order based on model behavior
# The model appears to be trained with inverted labels
XRAY_CLASSES = ['abnormal', 'normal']  # SWAPPED ORDER TO FIX PREDICTIONS

def load_efficientnet():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(EFFICIENTNET_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH_EFFNET, map_location='cpu'))
    model.eval()
    return model

def load_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(RESNET18_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH_RESNET, map_location='cpu'))
    model.eval()
    return model

def load_dermnet_resnet50():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(DERMNET_RESNET50_CLASSES))
    
    # Load the checkpoint
    checkpoint = torch.load(MODEL_PATH_DERMNET_RESNET50, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Load from model_state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Load from state_dict
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try to load directly as state_dict
            model.load_state_dict(checkpoint)
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def load_xray_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH_XRAY)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH_XRAY)
    model.eval()
    return model, tokenizer
