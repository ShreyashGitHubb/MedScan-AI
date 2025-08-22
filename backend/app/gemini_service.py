"""
Gemini AI Integration for Enhanced Medical Image Analysis
"""

import google.generativeai as genai
import os
import base64
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class GeminiMedicalAnalyzer:
    def __init__(self, api_key=None):
        """
        Initialize Gemini AI analyzer
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.available = True
            logger.info("Gemini AI initialized successfully")
        else:
            self.available = False
            logger.warning("Gemini API key not found. Enhanced analysis will be unavailable.")
    
    def analyze_mri_image(self, image_data, predicted_class, confidence):
        """
        Analyze MRI image using Gemini AI for enhanced insights
        """
        if not self.available:
            return self._get_fallback_analysis(predicted_class, confidence)
        
        try:
            # Convert image data to PIL Image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data
            
            # Create prompt for medical analysis
            prompt = f"""
            You are a medical AI assistant analyzing an MRI brain scan image. 
            
            Current AI model prediction: {predicted_class} (confidence: {confidence:.1%})
            
            Please analyze this image and provide:
            
            1. IMAGE VALIDATION:
            - Is this actually an MRI brain scan image?
            - If not, what type of image is this?
            - Rate the image quality for medical analysis (1-10)
            
            2. MEDICAL ANALYSIS (only if it's a real MRI):
            - Visual characteristics you observe
            - Consistency with the AI prediction
            - Any notable features or abnormalities
            
            3. RECOMMENDATIONS:
            - Should this prediction be trusted?
            - What additional steps should be taken?
            - Any red flags or concerns?
            
            Please be concise and focus on practical medical insights. If this is clearly not an MRI scan, 
            please state that clearly and explain why the AI prediction should not be trusted.
            
            Format your response in clear sections with bullet points.
            """
            
            response = self.model.generate_content([prompt, image])
            
            return {
                'gemini_analysis': response.text,
                'enhanced_confidence': self._calculate_enhanced_confidence(response.text, confidence),
                'image_validity': self._extract_image_validity(response.text),
                'recommendations': self._extract_recommendations(response.text)
            }
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._get_fallback_analysis(predicted_class, confidence)
    
    def _calculate_enhanced_confidence(self, analysis_text, original_confidence):
        """
        Adjust confidence based on Gemini's analysis
        """
        analysis_lower = analysis_text.lower()
        
        # Check for image validity indicators
        if any(phrase in analysis_lower for phrase in [
            'not an mri', 'not a brain scan', 'not medical', 'wallpaper', 
            'artwork', 'photograph', 'not suitable for medical analysis'
        ]):
            # Very low confidence for non-medical images
            return min(original_confidence * 0.2, 0.15)
        
        elif any(phrase in analysis_lower for phrase in [
            'poor quality', 'unclear', 'difficult to analyze', 'low resolution'
        ]):
            # Reduced confidence for poor quality images
            return min(original_confidence * 0.6, 0.4)
        
        elif any(phrase in analysis_lower for phrase in [
            'clear mri', 'good quality', 'typical brain scan', 'consistent with'
        ]):
            # Maintain or slightly increase confidence for good images
            return min(original_confidence * 1.1, 0.95)
        
        else:
            # Default: slightly reduce confidence to be conservative
            return original_confidence * 0.9
    
    def _extract_image_validity(self, analysis_text):
        """
        Extract image validity assessment from Gemini's response
        """
        analysis_lower = analysis_text.lower()
        
        if any(phrase in analysis_lower for phrase in [
            'not an mri', 'not a brain scan', 'not medical'
        ]):
            return 'invalid'
        elif any(phrase in analysis_lower for phrase in [
            'poor quality', 'unclear', 'difficult to analyze'
        ]):
            return 'poor_quality'
        elif any(phrase in analysis_lower for phrase in [
            'clear mri', 'good quality', 'typical brain scan'
        ]):
            return 'valid'
        else:
            return 'uncertain'
    
    def _extract_recommendations(self, analysis_text):
        """
        Extract key recommendations from Gemini's analysis
        """
        # Simple extraction - in a real implementation, you might use more sophisticated NLP
        lines = analysis_text.split('\n')
        recommendations = []
        
        in_recommendations = False
        for line in lines:
            if 'recommendation' in line.lower() or 'should' in line.lower():
                in_recommendations = True
            if in_recommendations and line.strip().startswith(('•', '-', '*')):
                recommendations.append(line.strip())
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _get_fallback_analysis(self, predicted_class, confidence):
        """
        Provide fallback analysis when Gemini is not available
        """
        return {
            'gemini_analysis': 'Enhanced AI analysis unavailable. Using standard model prediction only.',
            'enhanced_confidence': confidence,
            'image_validity': 'uncertain',
            'recommendations': [
                'Consult with medical professionals for proper diagnosis',
                'Ensure uploaded image is a proper MRI brain scan',
                'Consider additional medical imaging if symptoms persist'
            ]
        }
    
    def validate_medical_image(self, image_data):
        """
        Quick validation to check if image appears to be medical
        """
        if not self.available:
            return {'is_medical': 'uncertain', 'confidence': 0.5}
        
        try:
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data
            
            prompt = """
            Analyze this image and determine:
            1. Is this a medical image (MRI, CT, X-ray, etc.)?
            2. If medical, what type of medical image is it?
            3. Rate your confidence (0-100%) that this is a medical image.
            
            Respond in this exact format:
            MEDICAL: Yes/No
            TYPE: [type if medical, or "Not medical" if not]
            CONFIDENCE: [0-100]%
            """
            
            response = self.model.generate_content([prompt, image])
            
            # Parse response
            lines = response.text.strip().split('\n')
            result = {'is_medical': 'uncertain', 'confidence': 0.5, 'type': 'unknown'}
            
            for line in lines:
                if line.startswith('MEDICAL:'):
                    result['is_medical'] = 'yes' if 'yes' in line.lower() else 'no'
                elif line.startswith('TYPE:'):
                    result['type'] = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        conf_str = line.split(':', 1)[1].strip().replace('%', '')
                        result['confidence'] = float(conf_str) / 100.0
                    except:
                        pass
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini validation failed: {e}")
            return {'is_medical': 'uncertain', 'confidence': 0.5, 'type': 'unknown'}

# Global instance
gemini_analyzer = GeminiMedicalAnalyzer()