from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple

class PredictionResponse(BaseModel):
    model: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam_png: str  # base64-encoded PNG image

class DiseaseRisk(BaseModel):
    disease: str
    probability: float
    severity: str  # "Low", "Moderate", "High"
    description: str

class MedicalSuggestion(BaseModel):
    category: str  # "immediate", "follow_up", "lifestyle", "monitoring"
    suggestion: str
    priority: str  # "High", "Medium", "Low"

class KeyFinding(BaseModel):
    finding: str
    significance: str
    location: Optional[str] = None
    confidence: Optional[float] = None

class XrayAnalysisResponse(BaseModel):
    model: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    
    # Enhanced analysis
    key_findings: List[KeyFinding]
    disease_risks: List[DiseaseRisk]
    medical_suggestions: List[MedicalSuggestion]
    severity_assessment: str
    follow_up_recommendations: str
    
    # Report analysis
    report_summary: str
    clinical_significance: str

class GeminiEnhancedResponse(BaseModel):
    """
    Comprehensive X-ray analysis response with Gemini AI enhancement
    """
    # Original analysis results
    model: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    
    # Enhanced medical analysis
    key_findings: List[str]
    disease_risks: Dict[str, Any]
    medical_suggestions: List[str]
    severity_assessment: str
    follow_up_recommendations: List[str]
    report_summary: str
    clinical_significance: str
    
    # Gemini AI enhancements
    gemini_enhanced_findings: List[str]
    gemini_corrected_diagnosis: str
    gemini_confidence_assessment: float
    gemini_clinical_recommendations: List[str]
    gemini_contradictions_found: List[str]
    gemini_missing_elements: List[str]
    gemini_report_quality_score: float
    gemini_enhanced_summary: str
    gemini_differential_diagnoses: List[str]
    gemini_urgency_level: str
    gemini_follow_up_timeline: str
    gemini_clinical_reasoning: str
    
    # Quality metrics
    analysis_quality_score: float
    gemini_review_status: str
    processing_timestamp: str
    
    # Advanced Gemini features (optional)
    patient_summary: Optional[Dict[str, Any]] = None
    clinical_decision_support: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    enhanced_confidence_metrics: Optional[Dict[str, Any]] = None

class AdvancedGeminiResponse(BaseModel):
    """
    Ultra-advanced X-ray analysis response with ensemble AI, clinical decision support, 
    and comprehensive medical analysis
    """
    # Core prediction
    model: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    
    # Enhanced medical analysis
    key_findings: List[str]
    disease_risks: Dict[str, Any]
    medical_suggestions: List[str]
    severity_assessment: str
    follow_up_recommendations: List[str]
    report_summary: str
    clinical_significance: str
    
    # Advanced Gemini enhancements
    gemini_enhanced_findings: List[str]
    gemini_corrected_diagnosis: str
    gemini_confidence_assessment: float
    gemini_clinical_recommendations: List[str]
    gemini_contradictions_found: List[str]
    gemini_missing_elements: List[str]
    gemini_report_quality_score: float
    gemini_enhanced_summary: str
    gemini_differential_diagnoses: List[str]
    gemini_urgency_level: str
    gemini_follow_up_timeline: str
    gemini_clinical_reasoning: str
    
    # Advanced features
    ensemble_confidence_score: float
    clinical_decision_support: List[str]
    risk_stratification: Dict[str, Any]
    quality_assurance_metrics: Dict[str, float]
    medical_imaging_correlations: List[str]
    patient_safety_alerts: List[str]
    radiologist_review_priority: str
    evidence_based_recommendations: List[str]
    diagnostic_accuracy_indicators: Dict[str, float]
    
    # Metadata
    analysis_quality_score: float
    processing_timestamp: str
    confidence_intervals: Dict[str, List[float]]  # Simplified from Tuple for JSON serialization
    analysis_version: str


# =============================
# Nutrition Planner Schemas
# =============================

class NutritionAnalysisRequest(BaseModel):
    text: str

class NutritionItem(BaseModel):
    name: str
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None

class NutritionAnalysisResponse(BaseModel):
    total_calories: Optional[float] = None
    macros: Dict[str, Optional[float]]  # { protein_g, carbs_g, fat_g }
    items: List[NutritionItem] = []
    suggestions: List[str] = []  # healthy tips
    swaps: List[str] = []        # healthy alternatives

class MealPlanRequest(BaseModel):
    goal: str  # "weight_loss" | "muscle_gain" | "diabetic_diet" | other
    calories_target: Optional[int] = None
    dietary_preference: Optional[str] = None  # e.g., vegetarian, vegan, none
    days: Optional[int] = None
    meals_per_day: Optional[int] = None

class MealPlanMeal(BaseModel):
    name: str
    description: Optional[str] = None
    calories: Optional[int] = None
    protein_g: Optional[int] = None
    carbs_g: Optional[int] = None
    fat_g: Optional[int] = None
    ingredients: Optional[List[str]] = None

class MealPlanDay(BaseModel):
    day: str
    meals: List[MealPlanMeal]

class MealPlanResponse(BaseModel):
    goal: str
    calories_target: Optional[int] = None
    dietary_preference: Optional[str] = None
    week: List[MealPlanDay]
    # Back-compat: some clients used `days` instead of `week`
    days: Optional[List[MealPlanDay]] = None


# =============================
# Nutrition Coach Schemas
# =============================

class CoachRequest(BaseModel):
    goal: Optional[str] = None
    dietary_preference: Optional[str] = None
    calories_target: Optional[int] = None
    context: Optional[str] = None  # free text context or recent meals

class CoachResponse(BaseModel):
    tips: List[str] = []
    habits: List[str] = []
    quick_meals: List[str] = []
    hydration: List[str] = []


# =============================
# HealthScan Schemas
# =============================

class HealthScanResponse(BaseModel):
    """Unified response for HealthScan image/document analysis"""
    kind: str  # "image" | "document"
    summary: str
    detected: List[str] = []
    risks: List[str] = []
    precautions: List[str] = []
    disclaimer: str = "This is not a medical diagnosis. Please consult a doctor for confirmation."
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    timestamp: str
    # Optional richer content
    keywords: List[str] = []
    sections: Optional[Dict[str, str]] = None  # e.g., {impression, findings, recommendations}
    entities: Optional[Dict[str, List[str]]] = None  # {diseases, symptoms, medications, labs}
    follow_up: List[str] = []
    confidence_note: Optional[str] = None
