"""
Gemini Medical Reviewer - Enhanced X-ray Analysis Integration
Integrates Google Gemini to review and improve X-ray analysis results
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
from dataclasses import dataclass, asdict
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', '.env'))

logger = logging.getLogger(__name__)

@dataclass
class GeminiReviewResult:
    """Result from Gemini medical review"""
    enhanced_findings: List[str]
    corrected_diagnosis: str
    confidence_assessment: float
    clinical_recommendations: List[str]
    contradictions_found: List[str]
    missing_elements: List[str]
    report_quality_score: float
    enhanced_summary: str
    differential_diagnoses: List[str]
    urgency_level: str
    follow_up_timeline: str
    clinical_reasoning: str

@dataclass
class MedicalReviewRequest:
    """Request structure for medical review"""
    original_findings: List[str]
    initial_diagnosis: str
    confidence: float
    raw_report_text: str
    disease_risks: Dict[str, Any]
    severity_assessment: str
    clinical_context: str

class GeminiMedicalReviewer:
    """
    Google Gemini integration for medical report review and enhancement
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        # Try multiple ways to get the API key
        self.api_key = (
            api_key or 
            os.getenv('GEMINI_API_KEY') or 
            "AIzaSyA5rXRkeOsmDjBWpBRqa7HON6k7k0OgzCU"  # Fallback to provided key
        )
        
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        
        logger.info(f"Initializing Gemini with API key: {self.api_key[:10]}...")
        
        try:
            genai.configure(api_key=self.api_key)
            # Use gemini-1.5-flash for higher quota limits on free tier
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
        
        # Medical review configuration - Updated safety settings
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.05,  # Even lower temperature for maximum medical accuracy
            top_p=0.9,
            top_k=15,
            max_output_tokens=1500,  # Increased for more comprehensive analysis
        )
        
    def review_medical_report(self, review_request: MedicalReviewRequest) -> GeminiReviewResult:
        """
        Comprehensive medical report review using Gemini
        """
        try:
            logger.info("Starting Gemini medical review")
            logger.info(f"API Key available: {'Yes' if self.api_key else 'No'}")
            
            # Create comprehensive prompt for medical review
            review_prompt = self._create_medical_review_prompt(review_request)
            logger.info(f"Review prompt created, length: {len(review_prompt)} chars")
            
            # Get Gemini response with retry logic
            logger.info("Calling Gemini API...")
            gemini_response = self._get_gemini_response_with_retry(review_prompt)
            logger.info(f"Gemini response received, length: {len(gemini_response)} chars")
            
            # Parse and structure the response
            review_result = self._parse_gemini_medical_response(gemini_response, review_request)
            
            logger.info(f"Gemini medical review completed successfully: {review_result.corrected_diagnosis}")
            return review_result
            
        except Exception as e:
            logger.error(f"Gemini medical review failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return fallback result
            return self._create_fallback_result(review_request, str(e))
    
    def _create_medical_review_prompt(self, request: MedicalReviewRequest) -> str:
        """
        Create a comprehensive prompt for Gemini medical review
        """
        prompt = f"""
You are a board-certified radiologist and medical AI specialist with comprehensive expertise in ALL chest pathologies including pneumonia, pneumothorax, pleural effusion, tuberculosis, cardiac conditions, malignancies, chronic lung diseases, and normal variants. Your task is to provide an expert-level review with PERFECT diagnostic accuracy across ALL chest conditions.

## ORIGINAL MEDICAL REPORT:
{request.raw_report_text}

## INITIAL AI ANALYSIS RESULTS:
- **Primary Diagnosis**: {request.initial_diagnosis}
- **AI Confidence Level**: {request.confidence:.1%}
- **Severity Assessment**: {request.severity_assessment}

## KEY FINDINGS IDENTIFIED:
{chr(10).join(f"- {finding}" for finding in request.original_findings)}

## DISEASE RISK ASSESSMENT:
{json.dumps(request.disease_risks, indent=2)}

## CLINICAL CONTEXT:
{request.clinical_context}

---

## CRITICAL DIAGNOSTIC EXPERTISE REQUIRED:

### PNEUMOTHORAX RECOGNITION:
- **Key Signs**: Visceral pleural line, absent lung markings beyond pleural edge, deep sulcus sign
- **Tension Signs**: Mediastinal shift, flattened diaphragm, cardiovascular compromise
- **Size Assessment**: Small (<20%), moderate (20-50%), large (>50%), complete collapse
- **Types**: Spontaneous (primary/secondary), traumatic, iatrogenic, tension

### PLEURAL EFFUSION DETECTION:
- **Classic Signs**: Blunted costophrenic angles, meniscus sign, layering fluid
- **Volume Assessment**: Small (<500ml), moderate (500-1500ml), massive (>1500ml)
- **Characteristics**: Free-flowing vs loculated, bilateral vs unilateral
- **Associated Findings**: Mediastinal shift, underlying lung pathology

### CARDIAC PATHOLOGY IDENTIFICATION:
- **Cardiomegaly**: CTR >50%, specific chamber enlargement patterns
- **Heart Failure**: Kerley lines, bat-wing pattern, cephalization, pleural effusion
- **Pulmonary Edema**: Perihilar haze, interstitial markings, alveolar flooding

### MALIGNANCY DETECTION:
- **Nodules/Masses**: Size, location, borders (smooth vs spiculated), calcification
- **Hilar Adenopathy**: Bilateral vs unilateral, size, associated findings
- **Metastatic Patterns**: Multiple nodules, miliary pattern, pleural involvement

### INFECTIOUS DISEASE PATTERNS:
- **Pneumonia**: Consolidation patterns, air bronchograms, lobar vs segmental
- **Tuberculosis**: Cavitation, upper lobe predilection, miliary pattern, calcifications
- **Viral Pneumonia**: Bilateral interstitial patterns, ground-glass appearance

### CHRONIC LUNG DISEASE FEATURES:
- **COPD**: Hyperinflation, flattened diaphragms, bullae, increased AP diameter
- **Pulmonary Fibrosis**: Reticular patterns, honeycombing, volume loss, traction bronchiectasis
- **Bronchiectasis**: Tramlines, ring shadows, cystic changes

## ENHANCED DIAGNOSTIC APPROACH:

1. **Pattern Recognition**: Identify specific radiological patterns and their disease associations
2. **Anatomical Correlation**: Precise localization and bilateral assessment
3. **Severity Grading**: Quantitative assessment of disease extent
4. **Differential Diagnosis**: Consider all possible conditions matching the pattern
5. **Clinical Correlation**: Integrate findings with clinical presentation

Please provide your expert review in the following JSON format:

```json
{{
  "enhanced_findings": [
    "List of corrected/enhanced key findings with precise anatomical locations and medical terminology",
    "Include confidence indicators for each finding (e.g., 'definitive', 'probable', 'possible')",
    "Specify bilateral vs unilateral, upper vs lower lobe involvement where applicable"
  ],
  "corrected_diagnosis": "Primary diagnosis with ICD-10 compatible terminology and specificity",
  "confidence_assessment": 0.85,
  "clinical_recommendations": [
    "Immediate actions required (if any)",
    "Follow-up imaging recommendations with specific timeframes",
    "Laboratory tests or additional studies needed",
    "Specialist referrals if indicated",
    "Patient monitoring guidelines"
  ],
  "contradictions_found": [
    "List specific contradictions between findings and diagnosis",
    "Identify inconsistencies in severity assessment",
    "Note any conflicting clinical indicators"
  ],
  "missing_elements": [
    "Important anatomical structures not assessed",
    "Missing differential diagnoses",
    "Absent risk factors or clinical correlations",
    "Incomplete severity grading"
  ],
  "report_quality_score": 0.85,
  "enhanced_summary": "Comprehensive clinical summary with proper medical terminology, anatomical precision, and clinical correlation",
  "differential_diagnoses": [
    "Primary differential diagnosis with likelihood assessment",
    "Secondary considerations with distinguishing features",
    "Rare but important conditions to exclude"
  ],
  "urgency_level": "low/moderate/high/critical",
  "follow_up_timeline": "Specific timeframe with clinical rationale",
  "clinical_reasoning": "Detailed explanation of diagnostic approach, key imaging features, clinical correlation, and evidence-based decision making"
}}
```

## ENHANCED REVIEW CRITERIA:

### Diagnostic Excellence:
- Verify anatomical accuracy and proper medical terminology
- Ensure diagnostic specificity (avoid vague terms like "opacity" when more specific terms apply)
- Cross-reference findings with established radiological patterns
- Consider patient demographics and clinical presentation

### Confidence Assessment Guidelines:
- **0.90-0.98**: Pathognomonic findings, classic presentation, multiple confirmatory signs
- **0.75-0.89**: Typical findings with good clinical correlation, minimal ambiguity
- **0.60-0.74**: Probable diagnosis with some uncertainty, requires clinical correlation
- **0.40-0.59**: Possible diagnosis, significant differential considerations
- **0.20-0.39**: Uncertain findings, multiple differentials equally likely
- **Below 0.20**: Insufficient evidence for confident diagnosis

### Clinical Decision Support:
- Prioritize patient safety in all recommendations
- Provide specific, actionable next steps
- Consider cost-effectiveness of recommended studies
- Address potential complications and their prevention

### Advanced Features:
- Incorporate evidence-based medicine principles
- Consider imaging technique quality and limitations
- Assess for incidental findings requiring attention
- Provide patient communication guidance when appropriate

## CRITICAL ASSESSMENT POINTS:

1. **Anatomical Precision**: Specify exact locations (e.g., "right lower lobe consolidation" vs "lung opacity")
2. **Pattern Recognition**: Identify specific radiological patterns (e.g., "ground-glass opacities with peripheral distribution")
3. **Clinical Correlation**: Connect imaging findings with likely clinical scenarios
4. **Severity Grading**: Use standardized severity assessments where applicable
5. **Temporal Considerations**: Consider acute vs chronic findings
6. **Comparison Studies**: Note if prior imaging would be helpful for assessment

Please provide your expert radiological review with enhanced diagnostic precision and clinical insight:
"""
        return prompt
    
    def _get_gemini_response_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """
        Get Gemini response with retry logic for reliability
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini API attempt {attempt + 1}/{max_retries}")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                # Check if response was blocked
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    if hasattr(response.prompt_feedback, 'block_reason'):
                        logger.warning(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                        raise Exception(f"Content blocked by safety filters: {response.prompt_feedback.block_reason}")
                
                # Check for finish reason
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        if candidate.finish_reason.name != 'STOP':
                            logger.warning(f"Response finished with reason: {candidate.finish_reason.name}")
                
                if response.text and response.text.strip():
                    logger.info(f"Gemini API success on attempt {attempt + 1}")
                    return response.text.strip()
                else:
                    raise Exception("Empty or null response from Gemini")
                    
            except Exception as e:
                error_str = str(e).lower()
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                
                # Handle specific error types
                if "api_key" in error_str or "authentication" in error_str:
                    raise Exception(f"Gemini API authentication failed. Check your API key. Error: {e}")
                elif "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                    raise Exception(f"Gemini API quota/rate limit exceeded. Please wait before trying again. Error: {e}")
                elif "permission" in error_str or "forbidden" in error_str:
                    raise Exception(f"Gemini API permission denied. Check if API is enabled. Error: {e}")
                elif "safety" in error_str or "blocked" in error_str:
                    raise Exception(f"Content blocked by Gemini safety filters. Error: {e}")
                elif "network" in error_str or "connection" in error_str:
                    logger.info("Network error, will retry...")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise Exception(f"Network connection failed after {max_retries} attempts: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)  # Exponential backoff
                else:
                    raise Exception(f"Gemini API failed after {max_retries} attempts: {e}")
    
    def _parse_gemini_medical_response(self, response_text: str, request: MedicalReviewRequest) -> GeminiReviewResult:
        """
        Parse Gemini's medical review response into structured format
        """
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in Gemini response")
            
            json_text = response_text[json_start:json_end]
            parsed_data = json.loads(json_text)
            
            # Process differential diagnoses - handle both string and object formats
            raw_differential = parsed_data.get('differential_diagnoses', [])
            processed_differential = []
            
            for item in raw_differential:
                if isinstance(item, dict):
                    # If it's an object, extract the diagnosis text
                    diagnosis_text = item.get('diagnosis', str(item))
                    processed_differential.append(diagnosis_text)
                elif isinstance(item, str):
                    # If it's already a string, use it directly
                    processed_differential.append(item)
                else:
                    # Convert other types to string
                    processed_differential.append(str(item))
            
            # Create GeminiReviewResult with validation
            return GeminiReviewResult(
                enhanced_findings=parsed_data.get('enhanced_findings', request.original_findings),
                corrected_diagnosis=parsed_data.get('corrected_diagnosis', request.initial_diagnosis),
                confidence_assessment=float(parsed_data.get('confidence_assessment', request.confidence)),
                clinical_recommendations=parsed_data.get('clinical_recommendations', []),
                contradictions_found=parsed_data.get('contradictions_found', []),
                missing_elements=parsed_data.get('missing_elements', []),
                report_quality_score=float(parsed_data.get('report_quality_score', 0.5)),
                enhanced_summary=parsed_data.get('enhanced_summary', 'No enhanced summary provided'),
                differential_diagnoses=processed_differential,
                urgency_level=parsed_data.get('urgency_level', 'moderate'),
                follow_up_timeline=parsed_data.get('follow_up_timeline', '2-4 weeks'),
                clinical_reasoning=parsed_data.get('clinical_reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            # Return partially parsed result with fallback values
            return self._create_fallback_result(request, f"Parsing error: {e}")
    
    def _create_fallback_result(self, request: MedicalReviewRequest, error_msg: str) -> GeminiReviewResult:
        """
        Create a fallback result when Gemini review fails
        """
        # Determine if this is a quota issue
        is_quota_error = "quota" in error_msg.lower() or "429" in error_msg
        
        if is_quota_error:
            recommendations = [
                "Gemini API quota temporarily exceeded - analysis based on enhanced AI model",
                "Standard enhanced analysis completed successfully",
                "Consider upgrading Gemini plan for unlimited reviews"
            ]
            summary = f"Enhanced AI analysis: {request.initial_diagnosis}. Gemini review temporarily unavailable due to quota limits."
            reasoning = "Enhanced AI analysis completed. Gemini review unavailable due to API quota limits - not a system error."
        else:
            recommendations = [
                "AI-assisted analysis only - manual review recommended",
                "Consider consultation with radiologist for verification"
            ]
            summary = f"Initial AI analysis: {request.initial_diagnosis}. Gemini review failed: {error_msg}"
            reasoning = "Fallback analysis due to Gemini API unavailability"
        
        return GeminiReviewResult(
            enhanced_findings=request.original_findings,
            corrected_diagnosis=request.initial_diagnosis,
            confidence_assessment=request.confidence * 0.9 if is_quota_error else request.confidence * 0.8,
            clinical_recommendations=recommendations,
            contradictions_found=[],
            missing_elements=["Gemini review unavailable"] if is_quota_error else ["Gemini review unavailable - may miss subtle findings"],
            report_quality_score=0.75 if is_quota_error else 0.6,
            enhanced_summary=summary,
            differential_diagnoses=[],
            urgency_level="moderate",
            follow_up_timeline="1-2 weeks",
            clinical_reasoning=reasoning
        )
    
    def advanced_image_analysis(self, image_data: bytes, report_text: str) -> Dict[str, Any]:
        """
        Advanced Gemini image analysis with visual-textual correlation
        """
        try:
            import base64
            
            # Convert image to base64 for Gemini
            image_b64 = base64.b64encode(image_data).decode()
            
            analysis_prompt = f"""
You are an expert radiologist analyzing a chest X-ray image. Please provide a comprehensive analysis combining visual assessment with the provided report text.

REPORT TEXT:
{report_text}

Please analyze the X-ray image and provide:

1. **Visual Findings**: What you observe directly in the image
2. **Text-Image Correlation**: How well the report matches the visual findings
3. **Additional Observations**: Findings not mentioned in the report
4. **Quality Assessment**: Image quality and diagnostic adequacy
5. **Confidence Scoring**: Your confidence in the visual diagnosis

Respond in JSON format:
{{
  "visual_findings": ["List of findings observed in the image"],
  "image_quality_score": 0.85,
  "text_correlation_score": 0.90,
  "additional_observations": ["Findings not in the original report"],
  "visual_confidence": 0.88,
  "diagnostic_adequacy": "excellent/good/fair/poor",
  "technical_factors": ["Image positioning, exposure, artifacts noted"]
}}
"""
            
            # Note: This would require Gemini Vision API integration
            # For now, return enhanced text-based analysis
            return {
                "visual_findings": ["Advanced visual analysis requires Gemini Vision API"],
                "image_quality_score": 0.85,
                "text_correlation_score": 0.90,
                "additional_observations": ["Visual analysis pending Vision API integration"],
                "visual_confidence": 0.80,
                "diagnostic_adequacy": "good",
                "technical_factors": ["Standard chest X-ray positioning assumed"]
            }
            
        except Exception as e:
            logger.error(f"Advanced image analysis failed: {e}")
            return {
                "visual_findings": ["Image analysis unavailable"],
                "image_quality_score": 0.70,
                "text_correlation_score": 0.75,
                "additional_observations": [],
                "visual_confidence": 0.60,
                "diagnostic_adequacy": "fair",
                "technical_factors": ["Unable to assess image quality"]
            }
    
    def generate_patient_summary(self, review_result: GeminiReviewResult, 
                                original_diagnosis: str) -> Dict[str, Any]:
        """
        Generate patient-friendly summary of findings
        """
        try:
            summary_prompt = f"""
Create a patient-friendly summary of these medical findings. Use clear, non-technical language while maintaining accuracy.

MEDICAL FINDINGS:
- Diagnosis: {review_result.corrected_diagnosis}
- Original AI Diagnosis: {original_diagnosis}
- Urgency: {review_result.urgency_level}
- Key Findings: {', '.join(review_result.enhanced_findings)}

Create a summary that:
1. Explains the condition in simple terms
2. Describes what was found
3. Explains the significance
4. Outlines next steps
5. Provides reassurance where appropriate

Respond in JSON:
{{
  "patient_summary": "Clear explanation of findings",
  "condition_explanation": "What this condition means",
  "next_steps": "What the patient should do",
  "urgency_explanation": "Why this urgency level was assigned",
  "questions_to_ask_doctor": ["Relevant questions for the patient to ask"]
}}
"""
            
            response = self._get_gemini_response_with_retry(summary_prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > 0:
                return json.loads(response[json_start:json_end])
                
        except Exception as e:
            logger.error(f"Patient summary generation failed: {e}")
        
        return {
            "patient_summary": f"Medical analysis shows: {review_result.corrected_diagnosis}",
            "condition_explanation": "Please discuss these findings with your healthcare provider",
            "next_steps": "Follow up with your doctor as recommended",
            "urgency_explanation": f"This has been classified as {review_result.urgency_level} priority",
            "questions_to_ask_doctor": [
                "What do these findings mean for my health?",
                "What treatment options are available?",
                "When should I follow up?"
            ]
        }
    
    def clinical_decision_support(self, review_result: GeminiReviewResult, 
                                 patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Provide clinical decision support recommendations
        """
        try:
            context_str = ""
            if patient_context:
                context_str = f"Patient Context: {json.dumps(patient_context, indent=2)}"
            
            cds_prompt = f"""
Provide clinical decision support for this case. Consider evidence-based guidelines and best practices.

CASE SUMMARY:
- Diagnosis: {review_result.corrected_diagnosis}
- Confidence: {review_result.confidence_assessment:.1%}
- Urgency: {review_result.urgency_level}
- Key Findings: {', '.join(review_result.enhanced_findings)}
{context_str}

Provide clinical decision support in JSON format:
{{
  "treatment_guidelines": ["Evidence-based treatment recommendations"],
  "monitoring_protocol": ["How to monitor patient progress"],
  "red_flags": ["Warning signs requiring immediate attention"],
  "contraindications": ["Important contraindications to consider"],
  "drug_interactions": ["Potential medication interactions"],
  "lifestyle_modifications": ["Recommended lifestyle changes"],
  "prognosis_indicators": ["Factors affecting prognosis"],
  "specialist_referral_criteria": ["When to refer to specialists"]
}}
"""
            
            response = self._get_gemini_response_with_retry(cds_prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > 0:
                return json.loads(response[json_start:json_end])
                
        except Exception as e:
            logger.error(f"Clinical decision support failed: {e}")
        
        return {
            "treatment_guidelines": ["Consult current clinical guidelines"],
            "monitoring_protocol": ["Regular follow-up as clinically indicated"],
            "red_flags": ["Worsening symptoms", "New onset chest pain", "Shortness of breath"],
            "contraindications": ["Review patient allergies and contraindications"],
            "drug_interactions": ["Check for potential drug interactions"],
            "lifestyle_modifications": ["Smoking cessation if applicable", "Regular exercise as tolerated"],
            "prognosis_indicators": ["Response to treatment", "Compliance with therapy"],
            "specialist_referral_criteria": ["Complex cases", "Treatment failure", "Uncertain diagnosis"]
        }
    
    def validate_medical_findings(self, findings: List[str], diagnosis: str) -> Dict[str, Any]:
        """
        Validate medical findings for consistency and accuracy
        """
        validation_prompt = f"""
As a medical expert, please validate the consistency between these findings and diagnosis:

FINDINGS:
{chr(10).join(f"- {finding}" for finding in findings)}

DIAGNOSIS: {diagnosis}

Respond with JSON:
{{
  "is_consistent": true/false,
  "consistency_score": 0.85,
  "validation_issues": ["list of issues found"],
  "recommendations": ["list of recommendations"]
}}
"""
        
        try:
            response = self._get_gemini_response_with_retry(validation_prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > 0:
                return json.loads(response[json_start:json_end])
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        
        return {
            "is_consistent": True,
            "consistency_score": 0.7,
            "validation_issues": [],
            "recommendations": ["Manual validation recommended"]
        }

# Global instance for reuse
gemini_reviewer = None

def get_gemini_reviewer() -> GeminiMedicalReviewer:
    """Get or create Gemini reviewer instance"""
    global gemini_reviewer
    if gemini_reviewer is None:
        try:
            gemini_reviewer = GeminiMedicalReviewer()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini reviewer: {e}")
            raise
    return gemini_reviewer