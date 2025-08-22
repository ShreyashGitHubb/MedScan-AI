"""
OCR processor for medical report images
Supports various image formats and PDF files
"""

import io
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import fitz  # PyMuPDF
from typing import Union, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        """Initialize OCR processor with lazy loading"""
        self.easyocr_reader = None
        self._easyocr_initialized = False
        self._setup_tesseract()
    
    def _setup_tesseract(self):
        """Setup Tesseract OCR with common installation paths"""
        try:
            import pytesseract
            
            # Common Tesseract installation paths on Windows
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.environ.get('USERNAME', '')),
                r"C:\tesseract\tesseract.exe"
            ]
            
            # Try to find Tesseract executable
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Tesseract found at: {path}")
                    break
            else:
                # Try default (might be in PATH)
                try:
                    pytesseract.get_tesseract_version()
                    logger.info("Tesseract found in PATH")
                except:
                    logger.warning("Tesseract not found. OCR will use EasyOCR only.")
                    
        except ImportError:
            logger.warning("pytesseract not installed")
        except Exception as e:
            logger.warning(f"Tesseract setup failed: {e}")
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy
        """
        try:
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Apply slight blur to reduce noise, then sharpen
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert to numpy array for OpenCV operations
            img_array = np.array(image)
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            return Image.fromarray(processed)
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}. Using original image.")
            return image
    
    def extract_text_tesseract(self, image: Image.Image) -> str:
        """
        Extract text using Tesseract OCR with medical report optimizations
        """
        try:
            # Check if Tesseract is available
            import pytesseract
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Enhanced Tesseract configuration for medical reports
            # PSM 6: Uniform block of text (good for medical reports)
            # OEM 3: Default, based on what is available (LSTM + Legacy)
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-()[]{}/<>?!@#$%^&*+=_|\\~` '
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            return text.strip()
            
        except pytesseract.TesseractNotFoundError:
            logger.warning("Tesseract not found. Please install Tesseract OCR for better results.")
            return ""
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""
    
    def _initialize_easyocr(self):
        """Lazy initialization of EasyOCR"""
        if not self._easyocr_initialized:
            try:
                logger.info("Initializing EasyOCR (this may take a few minutes on first run)...")
                # Initialize with GPU support if available, fallback to CPU
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for better compatibility
                self._easyocr_initialized = True
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
                self._easyocr_initialized = True  # Mark as tried to avoid repeated attempts

    def extract_text_easyocr(self, image: Image.Image) -> str:
        """
        Extract text using EasyOCR
        """
        try:
            # Initialize EasyOCR if not already done
            self._initialize_easyocr()
            
            if self.easyocr_reader is None:
                logger.warning("EasyOCR not available, skipping")
                return ""
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Extract text with EasyOCR
            results = self.easyocr_reader.readtext(img_array, detail=0, paragraph=True)
            
            # Join all detected text
            text = ' '.join(results)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF file
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            extracted_text = ""
            
            # Extract text from each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()
                
                if text.strip():
                    # If text is extractable, use it
                    extracted_text += text + "\n"
                else:
                    # If no text, try OCR on the page image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Try both OCR methods
                    ocr_text = self.extract_text_hybrid(image)
                    extracted_text += ocr_text + "\n"
            
            pdf_document.close()
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""
    
    def extract_text_hybrid(self, image: Image.Image) -> str:
        """
        Hybrid approach: try both OCR methods and return the better result
        Falls back to single method if one fails
        """
        try:
            # Get results from both OCR engines
            tesseract_text = self.extract_text_tesseract(image)
            easyocr_text = self.extract_text_easyocr(image)
            
            # If both methods produced text, choose the better one
            if tesseract_text and easyocr_text:
                # Choose the better result based on length and medical keywords
                medical_keywords = [
                    'patient', 'diagnosis', 'findings', 'impression', 'chest', 'lung', 'heart',
                    'radiograph', 'xray', 'x-ray', 'ct', 'mri', 'scan', 'examination',
                    'normal', 'abnormal', 'opacity', 'consolidation', 'pneumonia',
                    'fracture', 'bone', 'soft tissue', 'abdomen', 'pelvis'
                ]
                
                def score_text(text: str) -> int:
                    """Score text based on length and medical keyword presence"""
                    if not text:
                        return 0
                    
                    score = len(text.split())  # Base score on word count
                    
                    # Bonus for medical keywords
                    text_lower = text.lower()
                    keyword_bonus = sum(5 for keyword in medical_keywords if keyword in text_lower)
                    
                    return score + keyword_bonus
                
                tesseract_score = score_text(tesseract_text)
                easyocr_score = score_text(easyocr_text)
                
                # Return the better result
                if tesseract_score >= easyocr_score:
                    logger.info(f"Using Tesseract result (score: {tesseract_score} vs {easyocr_score})")
                    return tesseract_text
                else:
                    logger.info(f"Using EasyOCR result (score: {easyocr_score} vs {tesseract_score})")
                    return easyocr_text
            
            # Fallback to whichever method worked
            elif tesseract_text:
                logger.info("Using Tesseract result (EasyOCR failed or empty)")
                return tesseract_text
            elif easyocr_text:
                logger.info("Using EasyOCR result (Tesseract failed or empty)")
                return easyocr_text
            else:
                logger.warning("Both OCR methods failed or returned empty results")
                return ""
                
        except Exception as e:
            logger.error(f"Hybrid OCR failed: {e}")
            # Try fallback to Tesseract only
            try:
                return self.extract_text_tesseract(image)
            except:
                return ""
    
    def _post_process_medical_text(self, text: str) -> str:
        """
        Post-process OCR text to improve medical text recognition
        """
        if not text:
            return text
        
        # Common OCR corrections for medical terms
        medical_corrections = {
            # Common OCR errors in medical terms
            'flndings': 'findings',
            'Flndings': 'Findings',
            'FLNDINGS': 'FINDINGS',
            'lmpression': 'impression',
            'Lmpression': 'Impression',
            'LMPRESSION': 'IMPRESSION',
            'patjent': 'patient',
            'Patjent': 'Patient',
            'PATJENT': 'PATIENT',
            'radiograph': 'radiograph',
            'Radiograph': 'Radiograph',
            'RADIOGRAPH': 'RADIOGRAPH',
            'x-ray': 'x-ray',
            'X-ray': 'X-ray',
            'X-RAY': 'X-RAY',
            'xray': 'x-ray',
            'Xray': 'X-ray',
            'XRAY': 'X-RAY',
            'pneumonla': 'pneumonia',
            'Pneumonla': 'Pneumonia',
            'PNEUMONLA': 'PNEUMONIA',
            'consolidatlon': 'consolidation',
            'Consolidatlon': 'Consolidation',
            'CONSOLIDATLON': 'CONSOLIDATION',
            'effuslon': 'effusion',
            'Effuslon': 'Effusion',
            'EFFUSLON': 'EFFUSION',
            'pneumothorax': 'pneumothorax',
            'Pneumothorax': 'Pneumothorax',
            'PNEUMOTHORAX': 'PNEUMOTHORAX',
            'cardlomegaly': 'cardiomegaly',
            'Cardlomegaly': 'Cardiomegaly',
            'CARDLOMEGALY': 'CARDIOMEGALY',
            'blateral': 'bilateral',
            'Blateral': 'Bilateral',
            'BLATERAL': 'BILATERAL',
            'unlateral': 'unilateral',
            'Unlateral': 'Unilateral',
            'UNLATERAL': 'UNILATERAL',
            'normaI': 'normal',
            'NormaI': 'Normal',
            'NORMAI': 'NORMAL',
            'abnormaI': 'abnormal',
            'AbnormaI': 'Abnormal',
            'ABNORMAI': 'ABNORMAL',
        }
        
        # Apply corrections
        corrected_text = text
        for error, correction in medical_corrections.items():
            corrected_text = corrected_text.replace(error, correction)
        
        # Clean up spacing issues common in OCR
        import re
        
        # Fix spacing around colons (common in medical reports)
        corrected_text = re.sub(r'\s*:\s*', ': ', corrected_text)
        
        # Fix multiple spaces
        corrected_text = re.sub(r'\s+', ' ', corrected_text)
        
        # Fix common section headers that get mangled
        section_fixes = [
            (r'F\s*I\s*N\s*D\s*I\s*N\s*G\s*S\s*:', 'FINDINGS:'),
            (r'I\s*M\s*P\s*R\s*E\s*S\s*S\s*I\s*O\s*N\s*:', 'IMPRESSION:'),
            (r'H\s*I\s*S\s*T\s*O\s*R\s*Y\s*:', 'HISTORY:'),
            (r'T\s*E\s*C\s*H\s*N\s*I\s*Q\s*U\s*E\s*:', 'TECHNIQUE:'),
            (r'C\s*O\s*M\s*P\s*A\s*R\s*I\s*S\s*O\s*N\s*:', 'COMPARISON:'),
        ]
        
        for pattern, replacement in section_fixes:
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        # Clean up line breaks and spacing
        corrected_text = corrected_text.strip()
        
        return corrected_text
    
    def process_medical_report(self, file_bytes: bytes, filename: str) -> str:
        """
        Main method to process medical report files (images or PDF)
        """
        try:
            filename_lower = filename.lower()
            
            if filename_lower.endswith('.pdf'):
                # Process PDF file
                logger.info(f"Processing PDF file: {filename}")
                text = self.extract_text_from_pdf(file_bytes)
                
            else:
                # Process image file
                logger.info(f"Processing image file: {filename}")
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                
                # Use hybrid OCR approach
                text = self.extract_text_hybrid(image)
            
            # Post-process the extracted text for better medical term recognition
            text = self._post_process_medical_text(text)
            
            if not text or len(text.strip()) < 10:
                # Provide helpful error message based on available OCR engines
                if not self._easyocr_initialized and not self._tesseract_available():
                    raise ValueError("No OCR engines available. Please install Tesseract OCR or ensure EasyOCR can download its models.")
                else:
                    raise ValueError("Could not extract meaningful text from the file. Please ensure the image is clear and contains readable text.")
            
            logger.info(f"Successfully extracted {len(text)} characters from {filename}")
            return text
            
        except Exception as e:
            logger.error(f"Medical report processing failed: {e}")
            raise ValueError(f"Failed to process medical report: {str(e)}")
    
    def _tesseract_available(self) -> bool:
        """Check if Tesseract is available"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except:
            return False

# Global OCR processor instance
ocr_processor = OCRProcessor()