"""
Ultra-Enhanced OCR Processor for Medical Reports
World-class accuracy with advanced preprocessing, multi-engine OCR, and medical text optimization
"""

import io
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import easyocr
import fitz  # PyMuPDF
from typing import Union, List, Tuple, Dict, Any
import logging
from scipy import ndimage
from skimage import restoration, exposure, morphology
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance tuning via environment variables (with safe defaults)
ULTRA_OCR_MAX_PAGES = int(os.getenv("ULTRA_OCR_MAX_PAGES", "5"))  # default: 5 pages
ULTRA_OCR_RENDER_ZOOM = float(os.getenv("ULTRA_OCR_RENDER_ZOOM", "2.0"))  # default: 2x zoom
ULTRA_OCR_TIME_BUDGET_SEC = float(os.getenv("ULTRA_OCR_TIME_BUDGET_SEC", "12"))  # total budget per PDF
ULTRA_OCR_EARLY_EXIT_CHARS = int(os.getenv("ULTRA_OCR_EARLY_EXIT_CHARS", "2000"))  # stop when enough text

class UltraOCRProcessor:
    """
    Ultra-Enhanced OCR Processor for Medical Reports
    - Advanced image preprocessing with medical optimization
    - Multi-engine OCR with intelligent result selection
    - Medical text correction and validation
    - Quality assessment and confidence scoring
    """
    
    def __init__(self):
        """Initialize with advanced settings"""
        self.easyocr_reader = None
        self._easyocr_initialized = False
        self._setup_tesseract()
        
        # Advanced preprocessing parameters
        self.preprocessing_params = {
            'contrast_enhancement': 2.5,
            'sharpness_enhancement': 2.0,
            'brightness_adjustment': 1.2,
            'noise_reduction_strength': 3,
            'deskew_enabled': True,
            'adaptive_threshold': True,
            'morphological_cleaning': True
        }
        
        # Medical text patterns for validation
        self.medical_patterns = {
            'sections': [
                r'(?i)\b(findings?|impression|history|technique|comparison|recommendation)\s*:',
                r'(?i)\b(chest|lung|heart|thorax)\b',
                r'(?i)\b(x-?ray|radiograph|ct|scan|examination)\b'
            ],
            'anatomical': [
                r'(?i)\b(right|left|bilateral|upper|middle|lower|lobe|apex|base)\b',
                r'(?i)\b(mediastin|hilar|pleural|cardiac|pulmonary)\b'
            ],
            'pathological': [
                r'(?i)\b(opacity|consolidation|infiltrat|effusion|pneumonia)\b',
                r'(?i)\b(normal|abnormal|unremarkable|significant)\b'
            ]
        }

    def _setup_tesseract(self):
        """Enhanced Tesseract setup with medical configuration"""
        try:
            import pytesseract
            
            # Enhanced path detection
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\tesseract\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.environ.get('USERNAME', '')),
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.environ.get('USERNAME', ''))
            ]
            
            tesseract_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Tesseract found at: {path}")
                    tesseract_found = True
                    break
            
            if not tesseract_found:
                try:
                    # Test if Tesseract is in PATH
                    pytesseract.get_tesseract_version()
                    logger.info("Tesseract found in PATH")
                    tesseract_found = True
                except:
                    logger.warning("Tesseract not found. OCR will use EasyOCR only.")
            
            # Test Tesseract functionality
            if tesseract_found:
                try:
                    # Test with a simple image
                    test_image = Image.new('RGB', (100, 30), color='white')
                    pytesseract.image_to_string(test_image)
                    logger.info("Tesseract test successful")
                except Exception as e:
                    logger.warning(f"Tesseract test failed: {e}")
                    
        except ImportError:
            logger.warning("pytesseract not installed")
        except Exception as e:
            logger.warning(f"Tesseract setup failed: {e}")

    def process_medical_report(self, file_bytes: bytes, filename: str = "unknown") -> str:
        """
        Process medical report with ultra-enhanced OCR
        """
        try:
            logger.info(f"Processing medical report: {filename}")
            
            # Determine file type
            file_type = self._determine_file_type(file_bytes, filename)
            logger.info(f"Detected file type: {file_type}")
            
            if file_type == "pdf":
                extracted_text = self._process_pdf_ultra(file_bytes)
            else:
                # Process as image
                image = self._load_image_from_bytes(file_bytes)
                if image is None:
                    raise ValueError("Unable to load image from bytes")
                
                extracted_text = self._process_image_ultra(image)
            
            # Post-process and validate medical text
            processed_text = self._post_process_medical_text_ultra(extracted_text)
            
            # Quality assessment
            quality_score = self._assess_text_quality(processed_text)
            logger.info(f"Text extraction quality score: {quality_score:.2f}")
            
            if quality_score < 0.3:
                logger.warning("Low quality text extraction. Results may be unreliable.")
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Medical report processing failed: {e}")
            raise Exception(f"OCR processing failed: {str(e)}")

    def _determine_file_type(self, file_bytes: bytes, filename: str) -> str:
        """Determine file type from bytes and filename"""
        # Check PDF magic number
        if file_bytes.startswith(b'%PDF'):
            return "pdf"
        
        # Check image magic numbers
        if file_bytes.startswith(b'\x89PNG'):
            return "png"
        elif file_bytes.startswith(b'\xff\xd8\xff'):
            return "jpeg"
        elif file_bytes.startswith(b'GIF8'):
            return "gif"
        elif file_bytes.startswith(b'BM'):
            return "bmp"
        
        # Fallback to filename extension
        filename_lower = filename.lower()
        if filename_lower.endswith('.pdf'):
            return "pdf"
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif')):
            return "image"
        
        return "image"  # Default assumption

    def _load_image_from_bytes(self, file_bytes: bytes) -> Image.Image:
        """Load and validate image from bytes"""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions
            if image.size[0] < 100 or image.size[1] < 100:
                logger.warning(f"Image dimensions too small: {image.size}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    def _process_pdf_ultra(self, pdf_bytes: bytes) -> str:
        """Ultra-enhanced PDF processing"""
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted_text = ""
            start_time = time.time()
            
            max_pages = max(1, min(pdf_document.page_count, ULTRA_OCR_MAX_PAGES))
            for page_num in range(max_pages):
                page = pdf_document[page_num]
                
                # Try text extraction first
                text = page.get_text()
                
                if text.strip() and len(text.strip()) > 50:
                    # Good text extraction
                    extracted_text += self._clean_pdf_text(text) + "\n\n"
                else:
                    # Use OCR on page image
                    logger.info(f"Using OCR for PDF page {page_num + 1}")
                    
                    # Get high-resolution page image
                    zoom = max(1.5, min(3.0, ULTRA_OCR_RENDER_ZOOM))
                    mat = fitz.Matrix(zoom, zoom)  # Tunable zoom for performance
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    image = Image.open(io.BytesIO(img_data))
                    page_text = self._process_image_ultra(image)
                    extracted_text += page_text + "\n\n"
                
                # Early exit conditions: enough content or time exceeded
                if len(extracted_text) >= ULTRA_OCR_EARLY_EXIT_CHARS:
                    logger.info("Ultra OCR early exit: enough text extracted")
                    break
                if time.time() - start_time > ULTRA_OCR_TIME_BUDGET_SEC:
                    logger.info("Ultra OCR early exit: time budget exceeded")
                    break
            
            pdf_document.close()
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return ""

    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF extracted text"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # Fix common PDF extraction issues
        cleaned = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', cleaned)  # Remove hyphenation
        cleaned = re.sub(r'\n([a-z])', r' \1', cleaned)  # Join broken lines
        
        return cleaned.strip()

    def _process_image_ultra(self, image: Image.Image) -> str:
        """Ultra-enhanced image processing and OCR"""
        try:
            # Multi-stage preprocessing
            preprocessed_images = []
            # Original enhanced image first (fast path)
            enhanced = self._ultra_preprocess_image(image)
            preprocessed_images.append(("enhanced", enhanced))

            # Fast-path: try Tesseract on enhanced only; return early if strong
            try:
                fast_text = self._extract_text_tesseract_ultra(enhanced, "enhanced")
                if fast_text:
                    fast_conf = self._calculate_ocr_confidence(fast_text)
                    if fast_conf >= 0.65 and self._is_likely_medical_text(fast_text):
                        logger.info(f"Fast-path Tesseract accepted (conf {fast_conf:.2f})")
                        return fast_text
            except Exception as e:
                logger.debug(f"Fast-path Tesseract failed: {e}")

            # Add a couple of additional variants only if needed
            high_contrast = self._create_high_contrast_version(image)
            preprocessed_images.append(("high_contrast", high_contrast))
            binary = self._create_binary_version(image)
            preprocessed_images.append(("binary", binary))

            # Run OCR on versions and select best result
            ocr_results = []
            start = time.time()
            per_image_budget = 6.0  # seconds budget for per-image OCR

            for version_name, processed_image in preprocessed_images:
                try:
                    # Tesseract OCR
                    tesseract_result = self._extract_text_tesseract_ultra(processed_image, version_name)
                    if tesseract_result:
                        ocr_results.append({
                            'text': tesseract_result,
                            'method': f'tesseract_{version_name}',
                            'confidence': self._calculate_ocr_confidence(tesseract_result)
                        })
                        # If Tesseract is already confident, skip EasyOCR for speed
                        if ocr_results[-1]['confidence'] >= 0.6 and self._is_likely_medical_text(ocr_results[-1]['text']):
                            continue
                    
                    # EasyOCR as fallback or booster
                    # Respect per-image time budget
                    if time.time() - start < per_image_budget:
                        easyocr_result = self._extract_text_easyocr_ultra(processed_image, version_name)
                        if easyocr_result:
                            ocr_results.append({
                                'text': easyocr_result,
                                'method': f'easyocr_{version_name}',
                                'confidence': self._calculate_ocr_confidence(easyocr_result)
                            })
                        
                except Exception as e:
                    logger.warning(f"OCR failed for version {version_name}: {e}")
                    continue
            
            # Select best result
            if ocr_results:
                best_result = max(ocr_results, key=lambda x: x['confidence'])
                logger.info(f"Best OCR result from: {best_result['method']} (confidence: {best_result['confidence']:.2f})")
                return best_result['text']
            else:
                logger.warning("All OCR methods failed")
                return ""
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return ""

    def _ultra_preprocess_image(self, image: Image.Image) -> Image.Image:
        """Ultra-enhanced image preprocessing"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize if too small (improve OCR accuracy)
            min_size = 1000
            if max(image.size) < min_size:
                scale_factor = min_size / max(image.size)
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Advanced contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.preprocessing_params['contrast_enhancement'])
            
            # Brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.preprocessing_params['brightness_adjustment'])
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(self.preprocessing_params['sharpness_enhancement'])
            
            # Convert to numpy array for advanced processing
            img_array = np.array(image)
            
            # Noise reduction
            if self.preprocessing_params['noise_reduction_strength'] > 0:
                img_array = self._advanced_noise_reduction(img_array)
            
            # Deskewing
            if self.preprocessing_params['deskew_enabled']:
                img_array = self._deskew_image(img_array)
            
            # Adaptive thresholding
            if self.preprocessing_params['adaptive_threshold']:
                img_array = self._adaptive_threshold_enhanced(img_array)
            
            # Morphological operations
            if self.preprocessing_params['morphological_cleaning']:
                img_array = self._morphological_cleaning(img_array)
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.warning(f"Ultra preprocessing failed: {e}, using basic preprocessing")
            return self._basic_preprocess_image(image)

    def _create_high_contrast_version(self, image: Image.Image) -> Image.Image:
        """Create high contrast version for difficult text"""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Extreme contrast enhancement
            enhancer = ImageEnhance.Contrast(gray)
            contrast_img = enhancer.enhance(4.0)
            
            # Auto levels adjustment
            img_array = np.array(contrast_img)
            img_array = exposure.rescale_intensity(img_array, in_range='image', out_range=(0, 255))
            
            return Image.fromarray(img_array.astype(np.uint8))
            
        except Exception as e:
            logger.warning(f"High contrast processing failed: {e}")
            return image.convert('L')

    def _create_binary_version(self, image: Image.Image) -> Image.Image:
        """Create binary version with optimal threshold"""
        try:
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Otsu's threshold for optimal binarization
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(img_array)
            binary = img_array > threshold
            
            return Image.fromarray((binary * 255).astype(np.uint8))
            
        except Exception as e:
            logger.warning(f"Binary processing failed: {e}")
            return image.convert('1')  # Simple threshold

    def _create_denoised_version(self, image: Image.Image) -> Image.Image:
        """Create denoised version"""
        try:
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Non-local means denoising
            denoised = restoration.denoise_nl_means(img_array, h=0.1, fast_mode=True, patch_size=5, patch_distance=3)
            denoised = (denoised * 255).astype(np.uint8)
            
            return Image.fromarray(denoised)
            
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            # Fallback to simple median filter
            return image.filter(ImageFilter.MedianFilter(size=3))

    def _advanced_noise_reduction(self, img_array: np.ndarray) -> np.ndarray:
        """Advanced noise reduction techniques"""
        try:
            # Gaussian blur for noise reduction
            img_array = ndimage.gaussian_filter(img_array, sigma=0.5)
            
            # Median filter for salt-and-pepper noise
            img_array = ndimage.median_filter(img_array, size=2)
            
            return img_array
            
        except Exception:
            return img_array

    def _deskew_image(self, img_array: np.ndarray) -> np.ndarray:
        """Deskew image to correct rotation"""
        try:
            # Simple deskewing using projection profile
            from scipy import ndimage
            
            # Calculate horizontal projection
            horizontal_proj = np.sum(img_array < 128, axis=1)
            
            # Find dominant angle (simplified approach)
            angles = np.arange(-5, 6, 0.5)  # Test angles from -5 to +5 degrees
            variances = []
            
            for angle in angles:
                rotated = ndimage.rotate(img_array, angle, reshape=False, cval=255)
                proj = np.sum(rotated < 128, axis=1)
                variances.append(np.var(proj))
            
            # Find angle with maximum variance (best alignment)
            best_angle = angles[np.argmax(variances)]
            
            if abs(best_angle) > 0.5:  # Only apply if significant skew
                img_array = ndimage.rotate(img_array, best_angle, reshape=False, cval=255)
                logger.info(f"Deskewed image by {best_angle} degrees")
            
            return img_array
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return img_array

    def _adaptive_threshold_enhanced(self, img_array: np.ndarray) -> np.ndarray:
        """Enhanced adaptive thresholding"""
        try:
            # Multiple adaptive threshold methods
            thresh1 = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
            )
            
            thresh2 = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10
            )
            
            # Combine results
            combined = cv2.bitwise_and(thresh1, thresh2)
            
            return combined
            
        except Exception:
            # Fallback to simple threshold
            _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh

    def _morphological_cleaning(self, img_array: np.ndarray) -> np.ndarray:
        """Morphological operations for cleaning"""
        try:
            # Remove small noise
            kernel_small = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel_small)
            
            # Close gaps in characters
            kernel_close = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
            
            return cleaned
            
        except Exception:
            return img_array

    def _extract_text_tesseract_ultra(self, image: Image.Image, version_name: str) -> str:
        """Ultra-enhanced Tesseract OCR"""
        try:
            import pytesseract
            
            # Medical-optimized Tesseract configuration
            custom_configs = [
                # Configuration for medical reports
                '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-()[]{}/<>?!@#$%^&*+=_|\\~` ',
                
                # Alternative configuration for dense text
                '--oem 3 --psm 3 -c preserve_interword_spaces=1',
                
                # Configuration for single text block
                '--oem 3 --psm 8'
            ]
            
            best_result = ""
            best_confidence = 0
            
            for config in custom_configs:
                try:
                    # Extract text with confidence
                    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Filter high-confidence words
                    confident_words = []
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > 30:  # Confidence threshold
                            word = data['text'][i].strip()
                            if word:
                                confident_words.append(word)
                    
                    result_text = ' '.join(confident_words)
                    
                    # Calculate overall confidence
                    if confident_words:
                        avg_confidence = np.mean([int(c) for c in data['conf'] if int(c) > 30])
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_result = result_text
                            
                except Exception as e:
                    logger.warning(f"Tesseract config failed: {e}")
                    continue
            
            # Fallback to simple OCR if no good result
            if not best_result:
                best_result = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            
            return best_result.strip()
            
        except Exception as e:
            logger.error(f"Tesseract ultra OCR failed: {e}")
            return ""

    def _extract_text_easyocr_ultra(self, image: Image.Image, version_name: str) -> str:
        """Ultra-enhanced EasyOCR"""
        try:
            # Initialize EasyOCR if needed
            self._initialize_easyocr_ultra()
            
            if self.easyocr_reader is None:
                return ""
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Extract text with detailed results
            results = self.easyocr_reader.readtext(
                img_array, 
                detail=1,  # Get confidence scores
                paragraph=False,  # Process as individual text blocks
                width_ths=0.7,  # Text width threshold
                height_ths=0.7,  # Text height threshold
                decoder='greedy'  # Use greedy decoder for medical text
            )
            
            # Filter and combine results
            confident_texts = []
            for result in results:
                bbox, text, confidence = result
                
                # Filter by confidence and text quality
                if confidence > 0.3 and len(text.strip()) > 0:
                    # Additional medical text validation
                    if self._is_likely_medical_text(text):
                        confident_texts.append(text)
                    elif confidence > 0.6:  # High confidence non-medical text
                        confident_texts.append(text)
            
            # Join texts with appropriate spacing
            final_text = self._smart_text_joining(confident_texts, results)
            
            return final_text.strip()
            
        except Exception as e:
            logger.error(f"EasyOCR ultra failed: {e}")
            return ""

    def _initialize_easyocr_ultra(self):
        """Ultra initialization of EasyOCR with optimal settings"""
        if not self._easyocr_initialized:
            try:
                logger.info("Initializing EasyOCR with medical optimization...")
                
                # Initialize with optimal settings for medical reports
                self.easyocr_reader = easyocr.Reader(
                    ['en'], 
                    gpu=False,  # Use CPU for better compatibility
                    model_storage_directory=None,  # Use default
                    download_enabled=True
                )
                
                self._easyocr_initialized = True
                logger.info("EasyOCR initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
                self._easyocr_initialized = True

    def _is_likely_medical_text(self, text: str) -> bool:
        """Check if text is likely medical content"""
        text_lower = text.lower()
        
        # Check for medical keywords
        medical_keywords = [
            'chest', 'lung', 'heart', 'thorax', 'patient', 'diagnosis',
            'findings', 'impression', 'x-ray', 'radiograph', 'opacity',
            'consolidation', 'pneumonia', 'effusion', 'normal', 'abnormal'
        ]
        
        return any(keyword in text_lower for keyword in medical_keywords)

    def _smart_text_joining(self, texts: List[str], ocr_results: List) -> str:
        """Intelligently join OCR text results"""
        if not texts:
            return ""
        
        # Simple joining for now - could be enhanced with spatial analysis
        joined = ' '.join(texts)
        
        # Clean up common OCR artifacts
        joined = re.sub(r'\s+', ' ', joined)
        joined = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', joined)
        
        return joined

    def _calculate_ocr_confidence(self, text: str) -> float:
        """Calculate confidence score for OCR result"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        # Base confidence factors
        length_factor = min(1.0, len(text) / 100)  # Longer text = higher confidence
        
        # Medical content factor
        medical_score = 0.0
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                medical_score += matches * 0.1
        
        medical_factor = min(1.0, medical_score)
        
        # Text quality factor (proper spacing, punctuation)
        quality_score = 0.0
        if re.search(r'[.!?:]', text):  # Has punctuation
            quality_score += 0.2
        if re.search(r'\b[A-Z][a-z]+', text):  # Has proper capitalization
            quality_score += 0.2
        if not re.search(r'[^\w\s.!?:,()-]', text):  # No weird characters
            quality_score += 0.2
        
        # Combine factors
        final_confidence = (length_factor * 0.3 + medical_factor * 0.4 + quality_score * 0.3)
        
        return min(0.95, max(0.1, final_confidence))  # Cap between 0.1 and 0.95

    def _post_process_medical_text_ultra(self, text: str) -> str:
        """Ultra-enhanced post-processing of medical text"""
        if not text or len(text.strip()) < 3:
            return text
        
        # Stage 1: Basic cleaning
        processed = text.strip()
        
        # Stage 2: Advanced medical corrections
        processed = self._apply_advanced_medical_corrections(processed)
        
        # Stage 3: Structure improvement
        processed = self._improve_text_structure(processed)
        
        # Stage 4: Medical validation and enhancement
        processed = self._validate_and_enhance_medical_content(processed)
        
        return processed

    def _apply_advanced_medical_corrections(self, text: str) -> str:
        """Apply comprehensive medical text corrections"""
        corrections = {
            # Common OCR errors in medical terms (case-insensitive)
            r'\bflndings?\b': 'findings', r'\bFlndings?\b': 'Findings', r'\bFLNDINGS?\b': 'FINDINGS',
            r'\blmpression\b': 'impression', r'\bLmpression\b': 'Impression', r'\bLMPRESSION\b': 'IMPRESSION',
            r'\bpatjent\b': 'patient', r'\bPatjent\b': 'Patient', r'\bPATJENT\b': 'PATIENT',
            r'\bradiograph\b': 'radiograph', r'\bRadiograph\b': 'Radiograph', r'\bRADIOGRAPH\b': 'RADIOGRAPH',
            r'\bpneumonla\b': 'pneumonia', r'\bPneumonla\b': 'Pneumonia', r'\bPNEUMONLA\b': 'PNEUMONIA',
            r'\bconsolidatlon\b': 'consolidation', r'\bConsolidatlon\b': 'Consolidation',
            r'\beffuslon\b': 'effusion', r'\bEffuslon\b': 'Effusion', r'\bEFFUSLON\b': 'EFFUSION',
            r'\bcardlomegaly\b': 'cardiomegaly', r'\bCardlomegaly\b': 'Cardiomegaly',
            r'\bblateral\b': 'bilateral', r'\bBlateral\b': 'Bilateral', r'\bBLATERAL\b': 'BILATERAL',
            r'\bunllateral\b': 'unilateral', r'\bUnllateral\b': 'Unilateral',
            r'\bnormaI\b': 'normal', r'\bNormaI\b': 'Normal', r'\bNORMAI\b': 'NORMAL',
            r'\babnormaI\b': 'abnormal', r'\bAbnormaI\b': 'Abnormal', r'\bABNORMAI\b': 'ABNORMAL',
            r'\bopaclty\b': 'opacity', r'\bOpaclty\b': 'Opacity', r'\bOPACLTY\b': 'OPACITY',
            r'\binfiltrat\b': 'infiltrate', r'\bInfiltrat\b': 'Infiltrate',
            r'\bpleuraI\b': 'pleural', r'\bPleuraI\b': 'Pleural', r'\bPLEURAI\b': 'PLEURAL',
            
            # Anatomical corrections
            r'\brlght\b': 'right', r'\bRlght\b': 'Right', r'\bRIGHT\b': 'RIGHT',
            r'\bleft\b': 'left', r'\bLeft\b': 'Left',  # This might catch some false positives, so be careful
            r'\bupper\b': 'upper', r'\bUpper\b': 'Upper',
            r'\blower\b': 'lower', r'\bLower\b': 'Lower',
            r'\bmlddle\b': 'middle', r'\bMlddle\b': 'Middle', r'\bMIDDLE\b': 'MIDDLE',
            
            # Technical terms
            r'\bx-?ray\b': 'x-ray', r'\bX-?ray\b': 'X-ray', r'\bX-?RAY\b': 'X-RAY',
            r'\btechnlque\b': 'technique', r'\bTechnlque\b': 'Technique',
            r'\bcomparlson\b': 'comparison', r'\bComparlson\b': 'Comparison',
            r'\brecommendatlon\b': 'recommendation', r'\bRecommendatlon\b': 'Recommendation',
        }
        
        corrected = text
        for error_pattern, correction in corrections.items():
            corrected = re.sub(error_pattern, correction, corrected, flags=re.IGNORECASE)
        
        return corrected

    def _improve_text_structure(self, text: str) -> str:
        """Improve text structure and formatting"""
        # Fix spacing around punctuation
        improved = re.sub(r'\s*([:.!?])\s*', r'\1 ', text)
        
        # Fix multiple spaces
        improved = re.sub(r'\s+', ' ', improved)
        
        # Fix section headers that got mangled
        section_patterns = [
            (r'F\s*I\s*N\s*D\s*I\s*N\s*G\s*S\s*:?', 'FINDINGS:'),
            (r'I\s*M\s*P\s*R\s*E\s*S\s*S\s*I\s*O\s*N\s*:?', 'IMPRESSION:'),
            (r'H\s*I\s*S\s*T\s*O\s*R\s*Y\s*:?', 'HISTORY:'),
            (r'T\s*E\s*C\s*H\s*N\s*I\s*Q\s*U\s*E\s*:?', 'TECHNIQUE:'),
            (r'C\s*O\s*M\s*P\s*A\s*R\s*I\s*S\s*O\s*N\s*:?', 'COMPARISON:'),
            (r'R\s*E\s*C\s*O\s*M\s*M\s*E\s*N\s*D\s*A\s*T\s*I\s*O\s*N\s*:?', 'RECOMMENDATION:')
        ]
        
        for pattern, replacement in section_patterns:
            improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)
        
        # Ensure proper spacing after sections
        improved = re.sub(r'([A-Z]+:)([A-Za-z])', r'\1 \2', improved)
        
        # Fix broken words across lines
        improved = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', improved)
        
        return improved

    def _validate_and_enhance_medical_content(self, text: str) -> str:
        """Validate and enhance medical content"""
        if len(text.strip()) < 10:
            return text
        
        # Check if text contains medical content
        medical_content_score = 0
        text_lower = text.lower()
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                medical_content_score += matches
        
        if medical_content_score == 0:
            logger.info("Text appears to contain no medical content")
        else:
            logger.info(f"Medical content score: {medical_content_score}")
        
        # Enhance medical abbreviations
        enhanced = self._expand_medical_abbreviations(text)
        
        return enhanced

    def _expand_medical_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations for better analysis"""
        # Common medical abbreviations (be careful not to expand common words)
        abbreviations = {
            r'\bCXR\b': 'chest X-ray',
            r'\bCT\b': 'CT scan',
            r'\bMRI\b': 'MRI',
            r'\bRUL\b': 'right upper lobe',
            r'\bRML\b': 'right middle lobe', 
            r'\bRLL\b': 'right lower lobe',
            r'\bLUL\b': 'left upper lobe',
            r'\bLLL\b': 'left lower lobe',
            r'\bBIL\b': 'bilateral',
            r'\bPTX\b': 'pneumothorax',
            r'\bGGO\b': 'ground glass opacity'
        }
        
        expanded = text
        for abbrev, expansion in abbreviations.items():
            # Only expand if it's clearly medical context
            expanded = re.sub(abbrev, expansion, expanded)
        
        return expanded

    def _assess_text_quality(self, text: str) -> float:
        """Assess overall quality of extracted text"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        quality_factors = []
        
        # Length factor
        length_score = min(1.0, len(text) / 200)
        quality_factors.append(length_score * 0.2)
        
        # Medical content factor
        medical_score = 0.0
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    medical_score += 0.2
        
        medical_score = min(1.0, medical_score)
        quality_factors.append(medical_score * 0.4)
        
        # Structure factor (presence of sections, proper formatting)
        structure_score = 0.0
        if re.search(r'(?i)findings?:', text):
            structure_score += 0.3
        if re.search(r'(?i)impression:', text):
            structure_score += 0.3
        if re.search(r'[.!?]', text):  # Has proper punctuation
            structure_score += 0.2
        if re.search(r'\b[A-Z][a-z]+', text):  # Has proper capitalization
            structure_score += 0.2
        
        structure_score = min(1.0, structure_score)
        quality_factors.append(structure_score * 0.3)
        
        # Readability factor
        words = text.split()
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            readability_score = min(1.0, avg_word_length / 8)  # Optimal around 6-8 chars
        else:
            readability_score = 0.0
        
        quality_factors.append(readability_score * 0.1)
        
        return sum(quality_factors)

    def _basic_preprocess_image(self, image: Image.Image) -> Image.Image:
        """Basic preprocessing fallback"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Basic enhancements
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Basic noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception:
            return image

# Global instance
ultra_ocr_processor = UltraOCRProcessor()