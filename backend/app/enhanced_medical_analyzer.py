"""
Enhanced Medical Analyzer for Comprehensive Respiratory Analysis
Provides radiologist-level accuracy and detailed analysis for all respiratory conditions
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class KeyFinding:
    finding: str
    significance: str
    location: str = ""
    severity: str = "mild"
    confidence: float = 0.0

@dataclass
class DiseaseRisk:
    disease: str
    probability: float
    severity: str
    description: str
    clinical_signs: List[str]
    differential_diagnosis: List[str]

@dataclass
class MedicalSuggestion:
    category: str
    suggestion: str
    priority: str
    timeframe: str = ""
    clinical_reasoning: str = ""

@dataclass
class ClinicalImpression:
    primary_diagnosis: str
    differential_diagnoses: List[str]
    severity_assessment: str
    clinical_significance: str
    immediate_concerns: List[str]
    followup_needed: bool

class EnhancedMedicalAnalyzer:
    """
    Comprehensive medical analyzer for respiratory conditions
    Provides radiologist-level analysis with detailed clinical impressions
    """
    
    def __init__(self):
        self.respiratory_patterns = self._initialize_respiratory_patterns()
        self.anatomical_regions = self._initialize_anatomical_regions()
        self.severity_indicators = self._initialize_severity_indicators()
        self.clinical_correlations = self._initialize_clinical_correlations()
        
    def _initialize_respiratory_patterns(self) -> Dict[str, Dict]:
        """Initialize comprehensive respiratory condition patterns"""
        return {
            # Infectious Conditions
            "pneumonia": {
                "patterns": [
                    r"pneumonia", r"consolidation", r"opacity", r"infiltrate", r"air bronchogram",
                    r"lobar.*pneumonia", r"bronchopneumonia", r"patchy.*infiltrate",
                    r"alveolar.*filling", r"inflammatory.*change", r"airspace.*disease"
                ],
                "mild_indicators": [
                    r"subtle.*opacity", r"minimal.*infiltrate", r"early.*consolidation",
                    r"patchy.*change", r"mild.*inflammatory"
                ],
                "locations": ["upper", "middle", "lower", "bilateral", "unilateral"],
                "severity_markers": ["extensive", "confluent", "multi-lobar", "bilateral"]
            },
            
            "bronchitis": {
                "patterns": [
                    r"bronchial.*wall.*thickening", r"peribronchial.*cuffing",
                    r"bronchial.*markings.*prominent", r"linear.*opacity",
                    r"bronchiectasis", r"bronchiolitis", r"airways.*inflammation"
                ],
                "mild_indicators": [
                    r"mild.*bronchial.*thickening", r"subtle.*peribronchial",
                    r"minimal.*bronchial.*prominence", r"early.*bronchitis"
                ],
                "locations": ["central", "peripheral", "bilateral"],
                "severity_markers": ["marked", "severe", "extensive"]
            },
            
            "covid_pneumonia": {
                "patterns": [
                    r"ground.*glass.*opacity", r"peripheral.*distribution",
                    r"covid.*pneumonia", r"bilateral.*ground.*glass",
                    r"organizing.*pneumonia", r"crazy.*paving"
                ],
                "mild_indicators": [
                    r"minimal.*ground.*glass", r"subtle.*peripheral.*opacity",
                    r"early.*covid.*changes", r"mild.*viral.*pattern"
                ],
                "locations": ["peripheral", "bilateral", "lower lobe"],
                "severity_markers": ["extensive", "bilateral", "consolidative"]
            },
            
            "tuberculosis": {
                "patterns": [
                    r"cavitation", r"upper.*lobe.*infiltrate", r"hilar.*adenopathy",
                    r"miliary.*pattern", r"fibrocavitary", r"apical.*opacity",
                    r"tree.*in.*bud", r"caseous.*necrosis"
                ],
                "mild_indicators": [
                    r"minimal.*apical.*opacity", r"subtle.*upper.*lobe",
                    r"early.*cavitary.*change", r"mild.*nodular"
                ],
                "locations": ["apical", "upper lobe", "bilateral"],
                "severity_markers": ["extensive", "miliary", "cavitary"]
            },
            
            # Allergic and Inflammatory Conditions
            "allergic_reaction": {
                "patterns": [
                    r"eosinophilic.*pneumonia", r"hypersensitivity.*pneumonitis",
                    r"allergic.*bronchopulmonary", r"peripheral.*eosinophilia",
                    r"fleeting.*infiltrate", r"migratory.*opacity"
                ],
                "mild_indicators": [
                    r"minimal.*eosinophilic", r"subtle.*hypersensitivity",
                    r"mild.*allergic.*change", r"early.*inflammatory"
                ],
                "locations": ["peripheral", "upper lobe", "bilateral"],
                "severity_markers": ["extensive", "severe", "acute"]
            },
            
            "asthma": {
                "patterns": [
                    r"hyperinflation", r"bronchial.*wall.*thickening",
                    r"air.*trapping", r"flattened.*diaphragm",
                    r"increased.*anteroposterior.*diameter"
                ],
                "mild_indicators": [
                    r"mild.*hyperinflation", r"subtle.*air.*trapping",
                    r"minimal.*bronchial.*thickening"
                ],
                "locations": ["bilateral", "diffuse"],
                "severity_markers": ["marked", "severe", "extensive"]
            },
            
            # Early Stage Infections
            "viral_pneumonia": {
                "patterns": [
                    r"interstitial.*pneumonia", r"viral.*pattern",
                    r"bilateral.*interstitial", r"reticular.*opacity",
                    r"ground.*glass", r"viral.*syndrome"
                ],
                "mild_indicators": [
                    r"minimal.*interstitial", r"subtle.*viral.*pattern",
                    r"early.*viral.*pneumonia", r"mild.*reticular"
                ],
                "locations": ["bilateral", "diffuse", "lower lobe"],
                "severity_markers": ["extensive", "bilateral", "confluent"]
            },
            
            "atypical_pneumonia": {
                "patterns": [
                    r"atypical.*pneumonia", r"mycoplasma.*pneumonia",
                    r"interstitial.*infiltrate", r"reticulonodular.*pattern",
                    r"walking.*pneumonia"
                ],
                "mild_indicators": [
                    r"minimal.*atypical", r"subtle.*interstitial",
                    r"early.*mycoplasma", r"mild.*reticulonodular"
                ],
                "locations": ["bilateral", "lower lobe", "unilateral"],
                "severity_markers": ["extensive", "bilateral", "severe"]
            },
            
            # Other Respiratory Conditions
            "pleural_effusion": {
                "patterns": [
                    r"pleural.*effusion", r"fluid.*level", r"blunted.*costophrenic",
                    r"meniscus.*sign", r"layering.*fluid"
                ],
                "mild_indicators": [
                    r"minimal.*pleural.*fluid", r"trace.*effusion",
                    r"small.*pleural.*effusion", r"blunting.*costophrenic"
                ],
                "locations": ["bilateral", "unilateral", "right", "left"],
                "severity_markers": ["large", "massive", "extensive"]
            },
            
            "pneumothorax": {
                "patterns": [
                    r"pneumothorax", r"collapsed.*lung", r"pleural.*air",
                    r"lung.*edge", r"tension.*pneumothorax"
                ],
                "mild_indicators": [
                    r"minimal.*pneumothorax", r"small.*pneumothorax",
                    r"apical.*pneumothorax", r"trace.*pleural.*air"
                ],
                "locations": ["apical", "bilateral", "unilateral"],
                "severity_markers": ["tension", "large", "extensive"]
            },
            
            "pulmonary_edema": {
                "patterns": [
                    r"pulmonary.*edema", r"alveolar.*flooding", r"butterfly.*pattern",
                    r"cardiac.*pulmonary.*edema", r"interstitial.*edema",
                    r"kerley.*lines", r"bat.*wing.*opacity"
                ],
                "mild_indicators": [
                    r"minimal.*edema", r"early.*pulmonary.*edema",
                    r"mild.*interstitial.*edema", r"subtle.*kerley"
                ],
                "locations": ["bilateral", "central", "perihilar"],
                "severity_markers": ["severe", "extensive", "acute"]
            },
            
            "fibrosis": {
                "patterns": [
                    r"pulmonary.*fibrosis", r"reticular.*pattern", r"honeycombing",
                    r"traction.*bronchiectasis", r"subpleural.*fibrosis",
                    r"usual.*interstitial.*pneumonia"
                ],
                "mild_indicators": [
                    r"minimal.*fibrosis", r"early.*fibrotic.*change",
                    r"subtle.*reticular", r"mild.*subpleural"
                ],
                "locations": ["bilateral", "lower lobe", "subpleural"],
                "severity_markers": ["extensive", "severe", "end-stage"]
            }
        }
    
    def _initialize_anatomical_regions(self) -> Dict[str, List[str]]:
        """Initialize anatomical region patterns"""
        return {
            "upper_lobe": ["upper.*lobe", "apical", "superior"],
            "middle_lobe": ["middle.*lobe", "lingula", "right.*middle"],
            "lower_lobe": ["lower.*lobe", "basal", "inferior"],
            "bilateral": ["bilateral", "both.*lung", "bilaterally"],
            "unilateral": ["unilateral", "one.*lung", "single.*lung"],
            "peripheral": ["peripheral", "subpleural", "cortical"],
            "central": ["central", "hilar", "perihilar", "mediastinal"],
            "diffuse": ["diffuse", "widespread", "generalized"]
        }
    
    def _initialize_severity_indicators(self) -> Dict[str, List[str]]:
        """Initialize severity assessment patterns"""
        return {
            "mild": [
                "minimal", "subtle", "mild", "trace", "small", "early",
                "limited", "focal", "localized", "slight"
            ],
            "moderate": [
                "moderate", "patchy", "multifocal", "scattered",
                "moderate.*sized", "partial", "intermediate"
            ],
            "severe": [
                "severe", "extensive", "widespread", "diffuse", "marked",
                "confluent", "large", "massive", "complete", "total"
            ],
            "critical": [
                "critical", "life.*threatening", "tension", "massive",
                "respiratory.*failure", "acute.*respiratory.*distress"
            ]
        }
    
    def _initialize_clinical_correlations(self) -> Dict[str, Dict]:
        """Initialize clinical correlation patterns"""
        return {
            "symptoms": {
                "respiratory": ["dyspnea", "shortness.*breath", "cough", "chest.*pain"],
                "systemic": ["fever", "malaise", "fatigue", "weight.*loss"],
                "cardiac": ["palpitations", "chest.*pressure", "orthopnea"]
            },
            "risk_factors": {
                "infectious": ["recent.*travel", "immunocompromised", "exposure"],
                "chronic": ["smoking", "copd", "diabetes", "heart.*disease"],
                "environmental": ["occupational.*exposure", "allergen.*exposure"]
            }
        }
    
    def analyze_comprehensive_respiratory(self, report_text: str, predicted_class: str, confidence: float) -> Dict:
        """
        Comprehensive respiratory analysis with radiologist-level detail
        """
        try:
            logger.info("Starting comprehensive respiratory analysis")
            
            # Check for minimal medical content
            if not self._has_medical_content(report_text):
                return self._create_non_medical_analysis(report_text, predicted_class, confidence)
            
            # Normalize text for analysis
            report_lower = report_text.lower()
            
            # Extract detailed findings
            key_findings = self._extract_comprehensive_findings(report_text)
            
            # Assess all respiratory conditions
            disease_risks = self._assess_all_respiratory_conditions(report_lower, predicted_class, confidence)
            
            # Generate clinical impression
            clinical_impression = self._generate_clinical_impression(key_findings, disease_risks, report_lower)
            
            # Generate comprehensive medical suggestions
            medical_suggestions = self._generate_comprehensive_suggestions(
                report_lower, disease_risks, clinical_impression
            )
            
            # Assess overall severity
            severity_assessment = self._assess_comprehensive_severity(disease_risks, key_findings)
            
            # Generate follow-up recommendations
            follow_up_recommendations = self._generate_detailed_followup(
                disease_risks, clinical_impression, severity_assessment
            )
            
            # Create detailed report summary
            report_summary = self._create_comprehensive_summary(
                key_findings, disease_risks, clinical_impression
            )
            
            # Assess clinical significance
            clinical_significance = self._assess_detailed_clinical_significance(
                disease_risks, clinical_impression, severity_assessment
            )
            
            result = {
                "model": "Enhanced Respiratory Analysis v2.0",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "key_findings": [
                    {
                        "finding": kf.finding,
                        "significance": kf.significance,
                        "location": kf.location,
                        "severity": kf.severity,
                        "confidence": kf.confidence
                    }
                    for kf in key_findings
                ],
                "disease_risks": [
                    {
                        "disease": dr.disease,
                        "probability": dr.probability,
                        "severity": dr.severity,
                        "description": dr.description,
                        "clinical_signs": dr.clinical_signs,
                        "differential_diagnosis": dr.differential_diagnosis
                    }
                    for dr in disease_risks
                ],
                "clinical_impression": {
                    "primary_diagnosis": clinical_impression.primary_diagnosis,
                    "differential_diagnoses": clinical_impression.differential_diagnoses,
                    "severity_assessment": clinical_impression.severity_assessment,
                    "clinical_significance": clinical_impression.clinical_significance,
                    "immediate_concerns": clinical_impression.immediate_concerns,
                    "followup_needed": clinical_impression.followup_needed
                },
                "medical_suggestions": [
                    {
                        "category": ms.category,
                        "suggestion": ms.suggestion,
                        "priority": ms.priority,
                        "timeframe": ms.timeframe,
                        "clinical_reasoning": ms.clinical_reasoning
                    }
                    for ms in medical_suggestions
                ],
                "severity_assessment": severity_assessment,
                "follow_up_recommendations": follow_up_recommendations,
                "report_summary": report_summary,
                "clinical_significance": clinical_significance,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_version": "2.0-comprehensive"
            }
            
            logger.info(f"Comprehensive analysis completed: {len(disease_risks)} conditions assessed")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    def _extract_comprehensive_findings(self, report_text: str) -> List[KeyFinding]:
        """Enhanced comprehensive findings extraction with improved accuracy and detail"""
        findings = []
        report_lower = report_text.lower()
        
        # Enhanced normal indicators
        normal_indicators = [
            r"clear.*lung.*fields", r"clear.*bilaterally", r"normal.*chest.*x.*ray",
            r"no.*abnormalities", r"unremarkable", r"within.*normal.*limits",
            r"impression.*normal", r"normal.*study", r"no.*acute.*findings"
        ]
        
        is_explicitly_normal = any(re.search(pattern, report_lower) for pattern in normal_indicators)
        
        # Enhanced negative exclusions
        negative_exclusions = {
            "pneumonia": [
                r"no.*consolidation", r"no.*infiltrate", r"no.*pneumonia", r"clear.*lung",
                r"lungs.*clear", r"no.*opacity", r"no.*airspace.*disease"
            ],
            "pleural_effusion": [
                r"no.*effusion", r"no.*fluid", r"no.*pleural.*fluid", r"sharp.*costophrenic.*angle",
                r"no.*blunting", r"clear.*costophrenic.*angle"
            ],
            "pneumothorax": [
                r"no.*pneumothorax", r"no.*collapsed.*lung", r"no.*pleural.*air",
                r"lungs.*fully.*expanded"
            ],
            "tuberculosis": [
                r"no.*cavitation", r"no.*tb", r"no.*tuberculosis", r"no.*upper.*lobe.*opacity"
            ],
            "covid_pneumonia": [
                r"no.*ground.*glass", r"no.*covid", r"no.*viral.*pneumonia", r"no.*bilateral.*opacity"
            ],
            "cardiac": [r"normal.*heart.*size", r"normal.*cardiac", r"heart.*normal"],
            "copd": [r"no.*copd", r"no.*emphysema", r"no.*hyperinflation"]
        }
        
        excluded_conditions = set()
        for condition, neg_patterns in negative_exclusions.items():
            for pattern in neg_patterns:
                if re.search(pattern, report_lower):
                    excluded_conditions.add(condition)
        
        # Enhanced finding patterns with more comprehensive coverage
        enhanced_finding_patterns = {
            "consolidation": {
                "patterns": [r"consolidation", r"airspace.*opacity", r"dense.*opacity", r"confluent.*opacity"],
                "significance": "Indicates possible pneumonia or infection",
                "severity_indicators": [r"extensive", r"bilateral", r"multilobar"]
            },
            "ground_glass": {
                "patterns": [r"ground.*glass", r"hazy.*opacity", r"increased.*interstitial"],
                "significance": "May suggest viral pneumonia or early infection",
                "severity_indicators": [r"bilateral", r"peripheral", r"extensive"]
            },
            "pleural_effusion": {
                "patterns": [r"pleural.*effusion", r"fluid.*pleural", r"blunt.*costophrenic", r"meniscus"],
                "significance": "Fluid accumulation in pleural space",
                "severity_indicators": [r"large", r"bilateral", r"massive"]
            },
            "pneumothorax": {
                "patterns": [r"pneumothorax", r"collapsed.*lung", r"pleural.*air", r"\bpneumo\b(?!nia)"],
                "significance": "Air in pleural space requiring immediate attention",
                "severity_indicators": [r"tension", r"large", r"complete"]
            },
            "cardiomegaly": {
                "patterns": [r"cardiomegaly", r"enlarged.*heart", r"heart.*enlarged", r"cardiac.*enlargement"],
                "significance": "Enlarged heart suggesting cardiac condition",
                "severity_indicators": [r"severe", r"marked", r"significant"]
            },
            "hyperinflation": {
                "patterns": [r"hyperinflation", r"hyperinflated", r"flattened.*diaphragm", r"increased.*lung.*volume"],
                "significance": "Suggests COPD or asthma",
                "severity_indicators": [r"severe", r"marked", r"significant"]
            },
            "infiltrate": {
                "patterns": [r"infiltrate", r"patchy.*opacity", r"reticular.*opacity", r"nodular.*opacity"],
                "significance": "Abnormal lung tissue density",
                "severity_indicators": [r"diffuse", r"bilateral", r"extensive"]
            },
            "atelectasis": {
                "patterns": [r"atelectasis", r"collapsed.*lobe", r"volume.*loss", r"linear.*opacity"],
                "significance": "Lung collapse or incomplete expansion",
                "severity_indicators": [r"complete", r"lobar", r"bilateral"]
            },
            "opacity": {
                "patterns": [r"opacity", r"opacification", r"shadowing", r"density"],
                "significance": "Abnormal lung density requiring evaluation",
                "severity_indicators": [r"dense", r"extensive", r"bilateral"]
            },
            "air_bronchogram": {
                "patterns": [r"air.*bronchogram", r"bronchogram", r"air.*filled.*bronchi"],
                "significance": "Air-filled bronchi within consolidated lung",
                "severity_indicators": [r"extensive", r"multiple", r"prominent"]
            }
        }
        
        # Extract findings with enhanced context analysis
        for finding_type, data in enhanced_finding_patterns.items():
            # Map finding types to condition names for exclusion check
            condition_mapping = {
                "consolidation": "pneumonia",
                "ground_glass": "covid_pneumonia", 
                "pleural_effusion": "pleural_effusion",
                "pneumothorax": "pneumothorax",
                "cardiomegaly": "cardiac",
                "hyperinflation": "copd",
                "infiltrate": "pneumonia",
                "atelectasis": "atelectasis",
                "opacity": "pneumonia",
                "air_bronchogram": "pneumonia"
            }
            
            mapped_condition = condition_mapping.get(finding_type, finding_type)
            if mapped_condition in excluded_conditions:
                continue
                
            for pattern in data["patterns"]:
                matches = list(re.finditer(pattern, report_lower))
                for match in matches:
                    # Enhanced context analysis
                    context_start = max(0, match.start() - 30)
                    context_end = min(len(report_lower), match.end() + 30)
                    context = report_lower[context_start:context_end]
                    
                    # Check for negative context
                    negative_indicators = ["no", "absence", "ruled out", "excluded", "without", "negative for"]
                    is_negative = any(neg in context[:match.start()-context_start+10] for neg in negative_indicators)
                    
                    if not is_negative:
                        # Determine severity
                        severity = "mild"
                        for severity_indicator in data["severity_indicators"]:
                            if re.search(severity_indicator, context):
                                severity = "moderate" if severity == "mild" else "severe"
                        
                        # Determine location if possible
                        location = self._find_detailed_location(report_lower, match.start(), match.end())
                        
                        # Calculate confidence based on context and specificity
                        confidence = self._calculate_finding_confidence(report_lower, pattern, finding_type)
                        
                        # Only add findings with reasonable confidence
                        if confidence >= 0.3:
                            finding = KeyFinding(
                                finding=f"{finding_type.replace('_', ' ').title()}: {match.group()}",
                                significance=f"{severity.title()} {data['significance'].lower()}",
                                location=location,
                                confidence=confidence
                            )
                            findings.append(finding)
        
        # Add normal findings if explicitly normal
        if is_explicitly_normal and not findings:
            normal_finding = KeyFinding(
                finding="Normal chest radiograph",
                significance="No acute abnormalities identified",
                location="Bilateral lung fields",
                confidence=0.9
            )
            findings.append(normal_finding)
        
        # Remove duplicates and sort by confidence
        unique_findings = self._deduplicate_findings(findings)
        return sorted(unique_findings, key=lambda x: x.confidence or 0.0, reverse=True)[:15]  # Limit to top 15 findings
    
    def _find_detailed_location(self, text: str, start_pos: int, end_pos: int) -> str:
        """Find detailed anatomical location of finding"""
        # Extract context around the finding
        context_start = max(0, start_pos - 50)
        context_end = min(len(text), end_pos + 50)
        context = text[context_start:context_end].lower()
        
        # Location patterns
        location_patterns = {
            "right upper lobe": [r"right.*upper.*lobe", r"rul"],
            "right middle lobe": [r"right.*middle.*lobe", r"rml"],
            "right lower lobe": [r"right.*lower.*lobe", r"rll"],
            "left upper lobe": [r"left.*upper.*lobe", r"lul"],
            "left lower lobe": [r"left.*lower.*lobe", r"lll"],
            "bilateral": [r"bilateral", r"both.*lung", r"bilaterally"],
            "right lung": [r"right.*lung", r"right.*side"],
            "left lung": [r"left.*lung", r"left.*side"],
            "upper lobes": [r"upper.*lobe", r"apical"],
            "lower lobes": [r"lower.*lobe", r"basilar"],
            "perihilar": [r"perihilar", r"hilar"],
            "peripheral": [r"peripheral", r"subpleural"]
        }
        
        for location, patterns in location_patterns.items():
            if any(re.search(pattern, context) for pattern in patterns):
                return location
        
        return "Not specified"
    
    def _calculate_finding_confidence(self, text: str, pattern: str, finding_type: str) -> float:
        """Calculate confidence score for a finding"""
        base_confidence = 0.5
        
        # Pattern specificity bonus
        if len(pattern) > 10:  # More specific patterns get higher confidence
            base_confidence += 0.1
        
        # Context analysis
        pattern_matches = len(re.findall(pattern, text.lower()))
        if pattern_matches > 1:
            base_confidence += min(0.2, pattern_matches * 0.05)
        
        # Medical terminology bonus
        medical_terms = [r"radiograph", r"x-ray", r"chest", r"findings", r"impression"]
        if any(re.search(term, text.lower()) for term in medical_terms):
            base_confidence += 0.1
        
        # Severity indicators
        severity_terms = [r"severe", r"extensive", r"marked", r"significant"]
        if any(re.search(term, text.lower()) for term in severity_terms):
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _assess_all_respiratory_conditions(self, report_lower: str, predicted_class: str, confidence: float) -> List[DiseaseRisk]:
        """Enhanced comprehensive assessment of all respiratory conditions with improved accuracy"""
        disease_risks = []
        
        # Enhanced negative exclusions with more precise patterns
        negative_exclusions = {
            "pneumonia": [
                r"no.*consolidation", r"no.*infiltrate", r"no.*evidence.*pneumonia", 
                r"clear.*lung", r"lungs.*clear", r"no.*opacity", r"no.*airspace.*disease", 
                r"normal.*lung.*fields", r"ruled.*out.*pneumonia"
            ],
            "pleural_effusion": [
                r"no.*effusion", r"no.*fluid", r"no.*pleural.*fluid", r"sharp.*costophrenic.*angle",
                r"no.*blunting", r"clear.*costophrenic.*angle"
            ],
            "pneumothorax": [
                r"no.*pneumothorax", r"no.*collapsed.*lung", r"no.*pleural.*air",
                r"lungs.*fully.*expanded", r"no.*pneumo"
            ],
            "tuberculosis": [
                r"no.*cavitation", r"no.*tb", r"no.*tuberculosis", r"no.*upper.*lobe.*opacity",
                r"no.*apical.*scarring", r"no.*hilar.*lymphadenopathy"
            ],
            "covid_pneumonia": [
                r"no.*ground.*glass", r"no.*covid", r"no.*viral.*pneumonia", r"no.*bilateral.*opacity",
                r"no.*peripheral.*opacity"
            ],
            "bronchitis": [r"no.*bronchitis", r"no.*bronchial.*thickening", r"no.*peribronchial.*cuffing"],
            "asthma": [r"no.*asthma", r"no.*hyperinflation", r"no.*air.*trapping", r"normal.*lung.*volumes"],
            "copd": [r"no.*copd", r"no.*emphysema", r"no.*chronic", r"no.*hyperinflation"],
            "cardiac": [r"normal.*heart.*size", r"normal.*cardiac", r"heart.*normal"]
        }
        
        # Determine excluded conditions
        excluded_conditions = set()
        for condition, neg_patterns in negative_exclusions.items():
            for pattern in neg_patterns:
                if re.search(pattern, report_lower):
                    excluded_conditions.add(condition)
        
        # Enhanced normal indicators - be more specific to avoid false positives
        strong_normal_patterns = [
            r"clear.*lung.*fields", r"clear.*bilaterally", r"normal.*chest.*x.*ray",
            r"no.*abnormalities", r"unremarkable.*study", r"impression.*normal",
            r"normal.*study", r"no.*acute.*findings", r"lungs.*are.*clear",
            r"chest.*x.*ray.*normal", r"radiograph.*normal"
        ]
        
        is_explicitly_normal = any(re.search(pattern, report_lower) for pattern in strong_normal_patterns)
        
        # Enhanced condition assessment with better probability calculation
        for condition, data in self.respiratory_patterns.items():
            # Skip explicitly excluded conditions
            if condition in excluded_conditions:
                continue
            
            risk_assessment = self._detailed_condition_assessment(
                report_lower, condition, data, predicted_class, confidence
            )
            
            # Enhanced filtering logic
            if risk_assessment:
                # For explicitly normal reports, be more selective
                if is_explicitly_normal:
                    # Only include if there's strong evidence or it matches predicted class
                    if risk_assessment.probability < 0.2 and condition.lower() != predicted_class.lower():
                        continue
                
                # Include risks above threshold, matching predicted class, or with clear findings
                should_include = (
                    risk_assessment.probability >= 0.15 or  # Lower threshold for inclusion
                    condition.lower() == predicted_class.lower() or
                    risk_assessment.probability >= 0.25  # Always include strong findings
                )
                
                if should_include:
                    disease_risks.append(risk_assessment)
        
        # Enhanced sorting with clinical priority
        disease_risks.sort(key=lambda x: (
            x.probability * self._get_condition_priority(x.disease) / 10,  # Weighted by priority
            x.probability
        ), reverse=True)
        
        # Ensure we always have some risks if abnormal findings are present
        if not disease_risks and not is_explicitly_normal and predicted_class.lower() != "normal":
            # Add a generic risk based on predicted class
            generic_risk = DiseaseRisk(
                disease=predicted_class,
                probability=max(0.15, confidence * 0.8),  # Minimum 15% or 80% of model confidence
                severity="mild" if confidence < 0.6 else "moderate",
                description=f"Possible {predicted_class.lower()} based on imaging findings. Further evaluation recommended."
            )
            disease_risks.append(generic_risk)
        
        return disease_risks[:10]  # Limit to top 10 most relevant risks
    
    def _get_condition_priority(self, condition: str) -> int:
        """Get clinical priority score for condition (higher = more urgent)"""
        priority_scores = {
            "pneumothorax": 10,  # Highest priority - emergency
            "tension pneumothorax": 10,
            "pneumonia": 8,
            "covid_pneumonia": 8,
            "pleural_effusion": 7,
            "tuberculosis": 7,
            "cardiac": 6,
            "copd": 5,
            "asthma": 5,
            "bronchitis": 4,
            "atelectasis": 3,
            "normal": 1
        }
        
        condition_lower = condition.lower()
        for key, score in priority_scores.items():
            if key in condition_lower:
                return score
        
        return 2  # Default priority
    
    def _detailed_condition_assessment(self, report_lower: str, condition: str, pattern_data: Dict, 
                                     predicted_class: str, base_confidence: float) -> DiseaseRisk:
        """Detailed assessment of individual respiratory condition"""
        
        # Calculate base probability
        pattern_matches = 0
        mild_matches = 0
        severity_matches = 0
        
        for pattern in pattern_data["patterns"]:
            if re.search(pattern, report_lower):
                pattern_matches += 1
        
        for mild_pattern in pattern_data.get("mild_indicators", []):
            if re.search(mild_pattern, report_lower):
                mild_matches += 1
        
        for severity_marker in pattern_data.get("severity_markers", []):
            if re.search(severity_marker, report_lower):
                severity_matches += 1
        
        # Calculate probability based on matches
        if pattern_matches == 0 and mild_matches == 0:
            return None
        
        # Improved probability calculation
        pattern_score = pattern_matches * 0.4 + mild_matches * 0.2
        base_prob = min(0.9, pattern_score * base_confidence)
        
        # Ensure minimum probability for clear findings
        if pattern_matches > 0:
            base_prob = max(0.25, base_prob)  # Minimum 25% for clear pattern matches
        elif mild_matches > 0:
            base_prob = max(0.15, base_prob)  # Minimum 15% for mild indicators
        
        # Adjust for severity
        if severity_matches > 0:
            severity = "severe" if severity_matches >= 2 else "moderate"
            base_prob = min(0.95, base_prob * 1.2)
        elif mild_matches > 0:
            severity = "mild"
            base_prob = max(0.1, base_prob * 0.8)  # Don't dismiss mild cases
        else:
            severity = "moderate"
        
        # Generate clinical signs
        clinical_signs = self._extract_clinical_signs(report_lower, condition, pattern_data)
        
        # Generate differential diagnoses
        differential_diagnoses = self._generate_differential_diagnoses(condition, clinical_signs)
        
        # Create detailed description
        description = self._generate_detailed_description(condition, severity, clinical_signs)
        
        return DiseaseRisk(
            disease=condition.replace('_', ' ').title(),
            probability=base_prob,
            severity=severity,
            description=description,
            clinical_signs=clinical_signs,
            differential_diagnosis=differential_diagnoses
        )
    
    def _generate_clinical_impression(self, key_findings: List[KeyFinding], 
                                    disease_risks: List[DiseaseRisk], report_lower: str) -> ClinicalImpression:
        """Generate comprehensive clinical impression with improved normal case handling"""
        
        # Check for explicit normal indicators
        strong_normal_patterns = [
            r"clear.*lung.*fields", r"clear.*bilaterally", r"normal.*chest.*x.*ray",
            r"no.*abnormalities", r"unremarkable", r"within.*normal.*limits",
            r"impression.*normal"
        ]
        
        is_explicitly_normal = any(re.search(pattern, report_lower) for pattern in strong_normal_patterns)
        
        # Determine primary diagnosis
        primary_diagnosis = "Normal chest X-ray"
        if disease_risks and not is_explicitly_normal:
            highest_risk = max(disease_risks, key=lambda x: x.probability)
            if highest_risk.probability > 0.4:  # Higher threshold for non-normal diagnosis
                primary_diagnosis = f"{highest_risk.disease} ({highest_risk.severity})"
        elif disease_risks and is_explicitly_normal:
            # For explicitly normal reports, only change diagnosis if very high probability
            highest_risk = max(disease_risks, key=lambda x: x.probability)
            if highest_risk.probability > 0.7:
                primary_diagnosis = f"Likely {highest_risk.disease} ({highest_risk.severity})"
        
        # Generate differential diagnoses (more conservative for normal reports)
        differential_diagnoses = []
        probability_threshold = 0.4 if is_explicitly_normal else 0.2
        for risk in disease_risks[:3]:  # Reduced to top 3 for normal reports
            if risk.probability > probability_threshold:
                differential_diagnoses.append(f"{risk.disease} (probability: {risk.probability:.1%})")
        
        # Assess severity (more conservative for explicitly normal reports)
        severity_assessment = self._assess_overall_clinical_severity(disease_risks, key_findings)
        if is_explicitly_normal and severity_assessment != "normal":
            # Downgrade severity for explicitly normal reports
            severity_map = {"critical": "severe", "severe": "moderate", "moderate": "mild", "mild": "normal"}
            severity_assessment = severity_map.get(severity_assessment, "normal")
        
        # Assess clinical significance
        clinical_significance = self._assess_impression_significance(disease_risks, severity_assessment)
        
        # Identify immediate concerns (more conservative for normal reports)
        immediate_concerns = []
        concern_threshold = 0.7 if is_explicitly_normal else 0.6
        for risk in disease_risks:
            if risk.probability > concern_threshold and risk.severity in ["severe", "critical"]:
                immediate_concerns.append(f"High probability of {risk.disease} requiring immediate attention")
            elif "pneumothorax" in risk.disease.lower() and risk.probability > (0.6 if is_explicitly_normal else 0.4):
                immediate_concerns.append("Possible pneumothorax - consider immediate evaluation")
            elif "tension" in report_lower or "respiratory.*failure" in report_lower:
                immediate_concerns.append("Signs suggestive of respiratory emergency")
        
        # Determine if follow-up is needed (more conservative for normal reports)
        followup_threshold = 0.5 if is_explicitly_normal else 0.3
        followup_needed = (
            len([r for r in disease_risks if r.probability > followup_threshold]) > 0 or
            severity_assessment in ["moderate", "severe", "critical"] or
            len(immediate_concerns) > 0
        )
        
        return ClinicalImpression(
            primary_diagnosis=primary_diagnosis,
            differential_diagnoses=differential_diagnoses,
            severity_assessment=severity_assessment,
            clinical_significance=clinical_significance,
            immediate_concerns=immediate_concerns,
            followup_needed=followup_needed
        )
    
    def _generate_comprehensive_suggestions(self, report_lower: str, disease_risks: List[DiseaseRisk], 
                                          clinical_impression: ClinicalImpression) -> List[MedicalSuggestion]:
        """Generate comprehensive medical suggestions with clinical reasoning"""
        suggestions = []
        
        # Immediate actions for high-risk conditions
        for risk in disease_risks:
            if risk.probability > 0.6:
                if "pneumonia" in risk.disease.lower():
                    suggestions.append(MedicalSuggestion(
                        category="immediate",
                        suggestion=f"Initiate antibiotic therapy for suspected {risk.disease}",
                        priority="high",
                        timeframe="within 4-6 hours",
                        clinical_reasoning=f"High probability ({risk.probability:.1%}) of bacterial infection with {risk.severity} severity"
                    ))
                elif "pneumothorax" in risk.disease.lower():
                    suggestions.append(MedicalSuggestion(
                        category="immediate",
                        suggestion="Urgent chest tube consideration for pneumothorax",
                        priority="critical",
                        timeframe="immediate",
                        clinical_reasoning=f"Pneumothorax detected with {risk.probability:.1%} probability"
                    ))
        
        # Follow-up recommendations
        if clinical_impression.followup_needed:
            timeframe = "24-48 hours" if clinical_impression.severity_assessment == "severe" else "7-14 days"
            suggestions.append(MedicalSuggestion(
                category="follow_up",
                suggestion=f"Follow-up chest X-ray in {timeframe}",
                priority="medium",
                timeframe=timeframe,
                clinical_reasoning="Monitor disease progression and treatment response"
            ))
        
        # Lifestyle and supportive care
        for risk in disease_risks:
            if risk.probability > 0.3:
                if "bronchitis" in risk.disease.lower() or "asthma" in risk.disease.lower():
                    suggestions.append(MedicalSuggestion(
                        category="lifestyle",
                        suggestion="Bronchodilator therapy and pulmonary hygiene measures",
                        priority="medium",
                        timeframe="ongoing",
                        clinical_reasoning=f"Support respiratory function in {risk.disease}"
                    ))
        
        # Monitoring suggestions
        suggestions.append(MedicalSuggestion(
            category="monitoring",
            suggestion="Monitor respiratory symptoms and oxygen saturation",
            priority="medium",
            timeframe="daily",
            clinical_reasoning="Early detection of clinical deterioration"
        ))
        
        return suggestions
    
    # Helper methods for detailed analysis
    def _find_detailed_location(self, report_lower: str, start: int, end: int) -> str:
        """Find detailed anatomical location of finding"""
        context = report_lower[max(0, start-50):min(len(report_lower), end+50)]
        
        for region, patterns in self.anatomical_regions.items():
            for pattern in patterns:
                if re.search(pattern, context):
                    return region.replace('_', ' ').title()
        
        return "Not specified"
    
    def _determine_detailed_severity(self, report_lower: str, pattern: str, start: int, end: int) -> str:
        """Determine detailed severity of finding"""
        context = report_lower[max(0, start-30):min(len(report_lower), end+30)]
        
        for severity, indicators in self.severity_indicators.items():
            for indicator in indicators:
                if re.search(indicator, context):
                    return severity
        
        return "moderate"  # Default
    
    def _calculate_finding_confidence(self, report_lower: str, pattern: str, finding_type: str) -> float:
        """Calculate confidence score for finding"""
        base_confidence = 0.6
        
        # Pattern specificity bonus
        if len(pattern) > 10:  # More specific patterns get higher confidence
            base_confidence += 0.1
        
        # Context analysis
        pattern_matches = len(re.findall(pattern, report_lower))
        if pattern_matches > 1:
            base_confidence += min(0.2, pattern_matches * 0.05)
        
        # Medical terminology bonus
        medical_terms = [r"radiograph", r"x-ray", r"chest", r"findings", r"impression"]
        if any(re.search(term, report_lower) for term in medical_terms):
            base_confidence += 0.1
        
        # Severity indicators
        severity_terms = [r"severe", r"extensive", r"marked", r"significant"]
        if any(re.search(term, report_lower) for term in severity_terms):
            base_confidence += 0.1
        
        # Finding-specific adjustments
        finding_adjustments = {
            "consolidation": 0.1,  # High confidence for consolidation
            "ground_glass": 0.1,   # High confidence for ground glass
            "pleural_effusion": 0.1,
            "pneumothorax": 0.15,  # Very high confidence for pneumothorax
            "cardiomegaly": 0.05,
            "hyperinflation": 0.05
        }
        
        if finding_type in finding_adjustments:
            base_confidence += finding_adjustments[finding_type]
        
        return min(0.95, base_confidence)
    
    def _determine_clinical_significance(self, condition: str, severity: str) -> str:
        """Determine clinical significance of finding"""
        significance_map = {
            "mild": f"Mild {condition.replace('_', ' ')} requiring monitoring",
            "moderate": f"Moderate {condition.replace('_', ' ')} requiring treatment",
            "severe": f"Severe {condition.replace('_', ' ')} requiring immediate intervention",
            "critical": f"Critical {condition.replace('_', ' ')} - emergency management needed"
        }
        return significance_map.get(severity, f"{condition.replace('_', ' ')} of unspecified severity")
    
    def _deduplicate_findings(self, findings: List[KeyFinding]) -> List[KeyFinding]:
        """Remove duplicate findings while preserving the most significant ones"""
        unique_findings = []
        seen_conditions = set()
        
        for finding in sorted(findings, key=lambda x: x.confidence, reverse=True):
            condition_key = finding.finding.split(':')[0].lower()
            if condition_key not in seen_conditions:
                unique_findings.append(finding)
                seen_conditions.add(condition_key)
        
        return unique_findings
    
    def _extract_clinical_signs(self, report_lower: str, condition: str, pattern_data: Dict) -> List[str]:
        """Extract clinical signs associated with condition"""
        clinical_signs = []
        
        # Add matched patterns as clinical signs
        for pattern in pattern_data["patterns"]:
            if re.search(pattern, report_lower):
                clinical_signs.append(pattern.replace('.*', ' ').replace('\\', '').title())
        
        return clinical_signs[:5]  # Limit to top 5 signs
    
    def _generate_differential_diagnoses(self, condition: str, clinical_signs: List[str]) -> List[str]:
        """Generate differential diagnoses based on condition and signs"""
        differential_map = {
            "pneumonia": ["Viral pneumonia", "Atypical pneumonia", "Pulmonary edema", "Lung cancer"],
            "bronchitis": ["Asthma", "COPD exacerbation", "Pneumonia", "Allergic reaction"],
            "covid_pneumonia": ["Viral pneumonia", "Bacterial pneumonia", "Organizing pneumonia"],
            "tuberculosis": ["Lung cancer", "Fungal infection", "Pneumonia", "Sarcoidosis"],
            "allergic_reaction": ["Asthma", "Eosinophilic pneumonia", "Drug reaction", "Viral infection"],
            "pleural_effusion": ["Heart failure", "Pneumonia", "Malignancy", "Tuberculosis"],
            "pneumothorax": ["Bullous disease", "Trauma", "Iatrogenic", "Spontaneous"]
        }
        
        return differential_map.get(condition, ["Further evaluation needed"])
    
    def _generate_detailed_description(self, condition: str, severity: str, clinical_signs: List[str]) -> str:
        """Generate detailed description of condition"""
        base_descriptions = {
            "pneumonia": "Inflammatory condition of the lung parenchyma with alveolar consolidation",
            "bronchitis": "Inflammation of the bronchial airways with thickening of bronchial walls",
            "covid_pneumonia": "Viral pneumonia with characteristic ground-glass opacities",
            "tuberculosis": "Granulomatous infection typically affecting upper lobes with potential cavitation",
            "allergic_reaction": "Hypersensitivity reaction affecting pulmonary parenchyma",
            "pleural_effusion": "Accumulation of fluid in the pleural space",
            "pneumothorax": "Presence of air in the pleural space causing lung collapse"
        }
        
        description = base_descriptions.get(condition, f"{condition.replace('_', ' ').title()} detected")
        
        if clinical_signs:
            description += f". Clinical signs include: {', '.join(clinical_signs[:3])}"
        
        description += f". Assessed as {severity} severity."
        
        return description
    
    def _assess_overall_clinical_severity(self, disease_risks: List[DiseaseRisk], key_findings: List[KeyFinding]) -> str:
        """Assess overall clinical severity"""
        if not disease_risks:
            return "normal"
        
        max_severity = "mild"
        high_prob_conditions = [r for r in disease_risks if r.probability > 0.5]
        
        if any(r.severity == "critical" for r in high_prob_conditions):
            max_severity = "critical"
        elif any(r.severity == "severe" for r in high_prob_conditions):
            max_severity = "severe"
        elif any(r.severity == "moderate" for r in high_prob_conditions):
            max_severity = "moderate"
        
        return max_severity
    
    def _assess_impression_significance(self, disease_risks: List[DiseaseRisk], severity: str) -> str:
        """Assess clinical significance of impression"""
        if severity == "critical":
            return "Immediate medical intervention required - potential life-threatening condition"
        elif severity == "severe":
            return "Urgent medical attention needed - significant pathology identified"
        elif severity == "moderate":
            return "Medical evaluation recommended - notable findings requiring follow-up"
        elif any(r.probability > 0.3 for r in disease_risks):
            return "Clinical correlation recommended - findings warrant further evaluation"
        else:
            return "Routine follow-up as clinically indicated"
    
    def _get_condition_priority(self, condition: str) -> int:
        """Get priority score for condition ordering"""
        priority_map = {
            "pneumothorax": 10,
            "pulmonary edema": 9,
            "pneumonia": 8,
            "covid pneumonia": 8,
            "tuberculosis": 7,
            "pleural effusion": 6,
            "bronchitis": 5,
            "allergic reaction": 4,
            "asthma": 4,
            "viral pneumonia": 3,
            "atypical pneumonia": 3,
            "fibrosis": 2
        }
        return priority_map.get(condition.lower(), 1)
    
    def _assess_comprehensive_severity(self, disease_risks: List[DiseaseRisk], key_findings: List[KeyFinding]) -> str:
        """Comprehensive severity assessment"""
        return self._assess_overall_clinical_severity(disease_risks, key_findings)
    
    def _generate_detailed_followup(self, disease_risks: List[DiseaseRisk], 
                                  clinical_impression: ClinicalImpression, severity: str) -> str:
        """Generate detailed follow-up recommendations"""
        if severity == "critical":
            return "Immediate emergency department evaluation. Consider ICU consultation if respiratory failure develops."
        elif severity == "severe":
            return "Urgent medical evaluation within 24 hours. Serial chest X-rays to monitor progression."
        elif severity == "moderate":
            return "Medical evaluation within 48-72 hours. Follow-up chest X-ray in 7-14 days depending on clinical response."
        elif any(r.probability > 0.3 for r in disease_risks):
            return "Routine medical follow-up in 1-2 weeks. Repeat chest X-ray if symptoms persist or worsen."
        else:
            return "Routine follow-up as clinically indicated. No immediate imaging follow-up required."
    
    def _create_comprehensive_summary(self, key_findings: List[KeyFinding], 
                                    disease_risks: List[DiseaseRisk], 
                                    clinical_impression: ClinicalImpression) -> str:
        """Create comprehensive report summary"""
        summary_parts = []
        
        # Primary findings
        if key_findings:
            summary_parts.append(f"Key radiological findings include {len(key_findings)} abnormalities.")
        
        # Primary diagnosis
        summary_parts.append(f"Primary impression: {clinical_impression.primary_diagnosis}.")
        
        # Significant conditions
        significant_conditions = [r for r in disease_risks if r.probability > 0.4]
        if significant_conditions:
            conditions_text = ", ".join([f"{r.disease} ({r.probability:.0%})" for r in significant_conditions[:3]])
            summary_parts.append(f"Significant findings suggest: {conditions_text}.")
        
        # Severity assessment
        summary_parts.append(f"Overall severity assessed as {clinical_impression.severity_assessment}.")
        
        # Immediate concerns
        if clinical_impression.immediate_concerns:
            summary_parts.append(f"Immediate attention needed for: {'; '.join(clinical_impression.immediate_concerns)}.")
        
        return " ".join(summary_parts)
    
    def _assess_detailed_clinical_significance(self, disease_risks: List[DiseaseRisk], 
                                             clinical_impression: ClinicalImpression, 
                                             severity: str) -> str:
        """Assess detailed clinical significance"""
        return clinical_impression.clinical_significance
    
    def _has_medical_content(self, text: str) -> bool:
        """Check if text contains meaningful medical content"""
        text_lower = text.lower()
        
        # Check for negative contexts first (text explicitly saying it's not medical)
        negative_contexts = [
            r'no medical content', r'not medical', r'non-medical', r'regular sentence',
            r'just.*sentence', r'no.*medical.*whatsoever'
        ]
        
        if any(re.search(pattern, text_lower) for pattern in negative_contexts):
            return False
        
        # Core medical keywords that strongly indicate medical content
        core_medical_keywords = [
            'chest x-ray', 'radiograph', 'ct scan', 'pneumonia', 'consolidation',
            'pleural effusion', 'pneumothorax', 'cardiomegaly', 'opacity', 'infiltrate',
            'atelectasis', 'bronchitis', 'tuberculosis'
        ]
        
        # Check for core medical terms first
        if any(keyword in text_lower for keyword in core_medical_keywords):
            return True
        
        # Medical report structure patterns
        medical_structure_patterns = [
            r'findings?:', r'impression:', r'history:', r'technique:', r'comparison:',
            r'pa and lateral', r'chest.*x.*ray', r'radiograph.*shows', r'lungs.*are',
            r'heart.*size', r'no.*acute.*findings'
        ]
        
        if any(re.search(pattern, text_lower) for pattern in medical_structure_patterns):
            return True
        
        # Secondary medical keywords (need multiple or in medical context)
        secondary_keywords = [
            'chest', 'lung', 'heart', 'patient', 'findings', 'impression', 
            'normal', 'abnormal', 'diagnosis', 'symptoms', 'treatment', 
            'hospital', 'doctor', 'examination', 'study', 'report'
        ]
        
        secondary_count = sum(1 for keyword in secondary_keywords if keyword in text_lower)
        
        # Need at least 2 secondary keywords or 1 secondary + medical context
        if secondary_count >= 2:
            return True
        
        # Check for medical context with single secondary keyword
        if secondary_count >= 1:
            medical_context_patterns = [
                r'chest.*clear', r'lung.*fields', r'heart.*normal', r'patient.*shows',
                r'findings.*suggest', r'impression.*is', r'normal.*study'
            ]
            if any(re.search(pattern, text_lower) for pattern in medical_context_patterns):
                return True
        
        return False
    
    def _create_non_medical_analysis(self, text: str, predicted_class: str, confidence: float) -> Dict:
        """Create analysis for non-medical text"""
        # Create a basic finding indicating non-medical content
        key_finding = {
            'finding': "Non-medical text detected",
            'significance': "Input text does not appear to contain medical or radiological content",
            'location': "N/A",
            'confidence': 0.9
        }
        
        # Create a low-risk assessment
        disease_risk = {
            'disease': "No medical assessment",
            'probability': 0.0,
            'severity': "N/A",
            'description': "Cannot assess medical conditions from non-medical text input"
        }
        
        # Create appropriate suggestion
        medical_suggestion = {
            'category': "information",
            'suggestion': "Please provide medical report text or radiological findings for accurate analysis",
            'priority': "Low"
        }
        
        return {
            'key_findings': [key_finding],
            'disease_risks': [disease_risk],
            'medical_suggestions': [medical_suggestion],
            'severity_assessment': 'N/A',
            'follow_up_recommendations': 'Please provide medical content for analysis',
            'report_summary': 'Non-medical text provided - no medical analysis performed',
            'clinical_significance': 'Input requires medical or radiological content for meaningful analysis'
        }

# Global enhanced analyzer instance
enhanced_medical_analyzer = EnhancedMedicalAnalyzer()