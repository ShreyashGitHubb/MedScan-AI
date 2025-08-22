"""
Advanced Medical Analysis Engine for X-ray Reports
Provides detailed disease risk assessment, medical suggestions, and clinical insights
"""

import re
from typing import List, Dict, Tuple
from .schemas import DiseaseRisk, MedicalSuggestion, KeyFinding

class MedicalAnalyzer:
    def __init__(self):
        # Medical terminology and patterns
        self.pathology_patterns = {
            'pneumonia': [
                r'consolidation', r'infiltrat\w+', r'opacity', r'pneumonia',
                r'air space disease', r'alveolar filling', r'bronchopneumonia'
            ],
            'covid-19': [
                r'ground glass', r'bilateral', r'peripheral', r'covid',
                r'coronavirus', r'sars-cov-2', r'multifocal'
            ],
            'tuberculosis': [
                r'cavitation', r'upper lobe', r'apical', r'tuberculosis',
                r'tb', r'miliary', r'hilar lymphadenopathy'
            ],
            'lung_cancer': [
                r'mass', r'nodule', r'tumor', r'malignancy', r'cancer',
                r'neoplasm', r'lesion', r'suspicious'
            ],
            'pleural_effusion': [
                r'pleural effusion', r'fluid', r'blunting', r'meniscus',
                r'layering', r'pleural'
            ],
            'pneumothorax': [
                r'pneumothorax', r'collapsed lung', r'air', r'pleural space'
            ],
            'heart_failure': [
                r'cardiomegaly', r'enlarged heart', r'pulmonary edema',
                r'congestion', r'heart failure', r'chf'
            ],
            'fracture': [
                r'fracture', r'break', r'broken', r'displaced', r'comminuted'
            ]
        }
        
        self.severity_indicators = {
            'high': [
                r'severe', r'extensive', r'massive', r'large', r'significant',
                r'marked', r'prominent', r'widespread', r'bilateral'
            ],
            'moderate': [
                r'moderate', r'mild to moderate', r'patchy', r'focal',
                r'localized', r'small to moderate'
            ],
            'low': [
                r'mild', r'minimal', r'slight', r'small', r'trace',
                r'questionable', r'possible'
            ]
        }
        
        self.anatomical_locations = {
            'upper_lobe': [r'upper lobe', r'apical', r'apex'],
            'middle_lobe': [r'middle lobe', r'lingula'],
            'lower_lobe': [r'lower lobe', r'base', r'basilar'],
            'bilateral': [r'bilateral', r'both', r'bibasilar'],
            'right': [r'right'],
            'left': [r'left']
        }

    def analyze_report(self, report_text: str, predicted_class: str, confidence: float) -> Dict:
        """
        Comprehensive analysis of X-ray report
        """
        report_lower = report_text.lower()
        
        # Extract key findings
        key_findings = self._extract_key_findings(report_text)
        
        # Assess disease risks
        disease_risks = self._assess_disease_risks(report_lower, predicted_class, confidence)
        
        # Generate medical suggestions
        medical_suggestions = self._generate_medical_suggestions(report_lower, disease_risks, predicted_class)
        
        # Assess severity
        severity_assessment = self._assess_severity(report_lower, disease_risks)
        
        # Generate follow-up recommendations
        follow_up_recommendations = self._generate_follow_up_recommendations(disease_risks, severity_assessment)
        
        # Create report summary
        report_summary = self._create_report_summary(key_findings, disease_risks)
        
        # Assess clinical significance
        clinical_significance = self._assess_clinical_significance(disease_risks, severity_assessment)
        
        return {
            'key_findings': key_findings,
            'disease_risks': disease_risks,
            'medical_suggestions': medical_suggestions,
            'severity_assessment': severity_assessment,
            'follow_up_recommendations': follow_up_recommendations,
            'report_summary': report_summary,
            'clinical_significance': clinical_significance
        }

    def _extract_key_findings(self, report_text: str) -> List[KeyFinding]:
        """Extract and categorize key medical findings"""
        findings = []
        report_lower = report_text.lower()
        
        # Check for pathological findings
        for condition, patterns in self.pathology_patterns.items():
            for pattern in patterns:
                if re.search(pattern, report_lower):
                    location = self._find_anatomical_location(report_lower, pattern)
                    significance = self._determine_finding_significance(report_lower, pattern)
                    
                    findings.append(KeyFinding(
                        finding=pattern.replace(r'\w+', '').replace('\\', ''),
                        significance=significance,
                        location=location
                    ))
                    break  # Avoid duplicates for same condition
        
        # Add normal findings if no pathology detected
        if not findings:
            findings.append(KeyFinding(
                finding="Clear lung fields",
                significance="Normal - no acute abnormalities detected",
                location="Bilateral"
            ))
        
        return findings[:5]  # Limit to top 5 findings

    def _assess_disease_risks(self, report_lower: str, predicted_class: str, confidence: float) -> List[DiseaseRisk]:
        """Assess probability of various diseases based on findings"""
        risks = []
        
        # Base risk assessment on AI prediction
        if predicted_class.lower() == 'abnormal':
            base_abnormal_prob = confidence
        else:
            base_abnormal_prob = 1 - confidence
        
        # Assess specific disease risks
        disease_assessments = {
            'Pneumonia': self._assess_pneumonia_risk(report_lower, base_abnormal_prob),
            'COVID-19': self._assess_covid_risk(report_lower, base_abnormal_prob),
            'Tuberculosis': self._assess_tb_risk(report_lower, base_abnormal_prob),
            'Lung Cancer': self._assess_cancer_risk(report_lower, base_abnormal_prob),
            'Pleural Effusion': self._assess_effusion_risk(report_lower, base_abnormal_prob),
            'Heart Failure': self._assess_heart_failure_risk(report_lower, base_abnormal_prob),
            'Pneumothorax': self._assess_pneumothorax_risk(report_lower, base_abnormal_prob)
        }
        
        # Convert to DiseaseRisk objects
        for disease, (prob, severity, desc) in disease_assessments.items():
            if prob > 0.1:  # Only include risks above 10%
                risks.append(DiseaseRisk(
                    disease=disease,
                    probability=prob,
                    severity=severity,
                    description=desc
                ))
        
        # Sort by probability
        risks.sort(key=lambda x: x.probability, reverse=True)
        return risks[:6]  # Top 6 risks

    def _assess_pneumonia_risk(self, report_lower: str, base_prob: float) -> Tuple[float, str, str]:
        """Assess pneumonia risk"""
        pneumonia_indicators = ['consolidation', 'infiltrat', 'opacity', 'pneumonia', 'air space']
        severity_score = 0
        
        for indicator in pneumonia_indicators:
            if indicator in report_lower:
                severity_score += 0.2
        
        if 'bilateral' in report_lower:
            severity_score += 0.3
        
        prob = min(base_prob * (1 + severity_score), 0.95)
        
        if prob > 0.7:
            severity = "High"
            desc = "Strong indicators of pneumonia with consolidation patterns"
        elif prob > 0.4:
            severity = "Moderate"
            desc = "Moderate likelihood based on opacity patterns"
        else:
            severity = "Low"
            desc = "Low probability based on current findings"
        
        return prob, severity, desc

    def _assess_covid_risk(self, report_lower: str, base_prob: float) -> Tuple[float, str, str]:
        """Assess COVID-19 risk"""
        covid_indicators = ['ground glass', 'bilateral', 'peripheral', 'multifocal']
        severity_score = 0
        
        for indicator in covid_indicators:
            if indicator in report_lower:
                severity_score += 0.25
        
        prob = min(base_prob * (0.3 + severity_score), 0.85)
        
        if prob > 0.6:
            severity = "Moderate"
            desc = "Pattern consistent with viral pneumonia, consider COVID-19 testing"
        else:
            severity = "Low"
            desc = "Low probability based on imaging patterns"
        
        return prob, severity, desc

    def _assess_tb_risk(self, report_lower: str, base_prob: float) -> Tuple[float, str, str]:
        """Assess tuberculosis risk"""
        tb_indicators = ['cavitation', 'upper lobe', 'apical', 'miliary', 'hilar']
        severity_score = 0
        
        for indicator in tb_indicators:
            if indicator in report_lower:
                severity_score += 0.3
        
        prob = min(base_prob * (0.2 + severity_score), 0.8)
        
        if prob > 0.5:
            severity = "Moderate"
            desc = "Upper lobe involvement suggests possible tuberculosis"
        else:
            severity = "Low"
            desc = "Low probability based on distribution pattern"
        
        return prob, severity, desc

    def _assess_cancer_risk(self, report_lower: str, base_prob: float) -> Tuple[float, str, str]:
        """Assess lung cancer risk"""
        cancer_indicators = ['mass', 'nodule', 'tumor', 'suspicious', 'lesion']
        severity_score = 0
        
        for indicator in cancer_indicators:
            if indicator in report_lower:
                severity_score += 0.4
        
        prob = min(base_prob * (0.1 + severity_score), 0.9)
        
        if prob > 0.6:
            severity = "High"
            desc = "Suspicious mass or nodule requires immediate evaluation"
        elif prob > 0.3:
            severity = "Moderate"
            desc = "Nodular changes need follow-up imaging"
        else:
            severity = "Low"
            desc = "No obvious mass lesions identified"
        
        return prob, severity, desc

    def _assess_effusion_risk(self, report_lower: str, base_prob: float) -> Tuple[float, str, str]:
        """Assess pleural effusion risk"""
        effusion_indicators = ['effusion', 'fluid', 'blunting', 'pleural']
        severity_score = sum(0.3 for indicator in effusion_indicators if indicator in report_lower)
        
        prob = min(base_prob * (0.1 + severity_score), 0.85)
        
        if prob > 0.5:
            severity = "Moderate"
            desc = "Pleural fluid collection identified"
        else:
            severity = "Low"
            desc = "No significant pleural effusion"
        
        return prob, severity, desc

    def _assess_heart_failure_risk(self, report_lower: str, base_prob: float) -> Tuple[float, str, str]:
        """Assess heart failure risk"""
        hf_indicators = ['cardiomegaly', 'enlarged heart', 'pulmonary edema', 'congestion']
        severity_score = sum(0.25 for indicator in hf_indicators if indicator in report_lower)
        
        prob = min(base_prob * (0.15 + severity_score), 0.8)
        
        if prob > 0.4:
            severity = "Moderate"
            desc = "Cardiac enlargement with possible congestion"
        else:
            severity = "Low"
            desc = "Normal cardiac silhouette"
        
        return prob, severity, desc

    def _assess_pneumothorax_risk(self, report_lower: str, base_prob: float) -> Tuple[float, str, str]:
        """Assess pneumothorax risk"""
        ptx_indicators = ['pneumothorax', 'collapsed', 'air']
        severity_score = sum(0.4 for indicator in ptx_indicators if indicator in report_lower)
        
        prob = min(base_prob * (0.05 + severity_score), 0.9)
        
        if prob > 0.5:
            severity = "High"
            desc = "Possible pneumothorax - immediate evaluation needed"
        else:
            severity = "Low"
            desc = "No evidence of pneumothorax"
        
        return prob, severity, desc

    def _generate_medical_suggestions(self, report_lower: str, disease_risks: List[DiseaseRisk], predicted_class: str) -> List[MedicalSuggestion]:
        """Generate actionable medical suggestions"""
        suggestions = []
        
        # High-risk findings require immediate attention
        high_risk_diseases = [risk for risk in disease_risks if risk.severity == "High"]
        if high_risk_diseases:
            suggestions.append(MedicalSuggestion(
                category="immediate",
                suggestion=f"Immediate clinical correlation recommended for {high_risk_diseases[0].disease.lower()}",
                priority="High"
            ))
        
        # Moderate risks need follow-up
        moderate_risks = [risk for risk in disease_risks if risk.severity == "Moderate"]
        if moderate_risks:
            suggestions.append(MedicalSuggestion(
                category="follow_up",
                suggestion="Follow-up imaging in 4-6 weeks to monitor progression",
                priority="Medium"
            ))
        
        # Specific suggestions based on findings
        if any('pneumonia' in risk.disease.lower() for risk in disease_risks if risk.probability > 0.4):
            suggestions.append(MedicalSuggestion(
                category="immediate",
                suggestion="Consider antibiotic therapy and supportive care",
                priority="High"
            ))
        
        if any('covid' in risk.disease.lower() for risk in disease_risks if risk.probability > 0.3):
            suggestions.append(MedicalSuggestion(
                category="immediate",
                suggestion="COVID-19 testing and isolation protocols recommended",
                priority="High"
            ))
        
        if any('cancer' in risk.disease.lower() for risk in disease_risks if risk.probability > 0.3):
            suggestions.append(MedicalSuggestion(
                category="follow_up",
                suggestion="CT chest with contrast for further characterization",
                priority="High"
            ))
        
        # General monitoring suggestions
        if predicted_class.lower() == 'abnormal':
            suggestions.append(MedicalSuggestion(
                category="monitoring",
                suggestion="Regular monitoring of symptoms and vital signs",
                priority="Medium"
            ))
        
        # Lifestyle recommendations
        suggestions.append(MedicalSuggestion(
            category="lifestyle",
            suggestion="Maintain good respiratory hygiene and avoid smoking",
            priority="Low"
        ))
        
        return suggestions[:5]  # Limit to 5 suggestions

    def _assess_severity(self, report_lower: str, disease_risks: List[DiseaseRisk]) -> str:
        """Assess overall severity of findings"""
        high_risk_count = sum(1 for risk in disease_risks if risk.severity == "High")
        moderate_risk_count = sum(1 for risk in disease_risks if risk.severity == "Moderate")
        
        # Check for severity indicators in text
        severity_score = 0
        for severity, indicators in self.severity_indicators.items():
            for indicator in indicators:
                if re.search(indicator, report_lower):
                    if severity == 'high':
                        severity_score += 3
                    elif severity == 'moderate':
                        severity_score += 2
                    else:
                        severity_score += 1
                    break
        
        if high_risk_count > 0 or severity_score > 5:
            return "High - Requires immediate medical attention"
        elif moderate_risk_count > 0 or severity_score > 2:
            return "Moderate - Follow-up recommended within 1-2 weeks"
        else:
            return "Low - Routine follow-up as clinically indicated"

    def _generate_follow_up_recommendations(self, disease_risks: List[DiseaseRisk], severity: str) -> str:
        """Generate specific follow-up recommendations"""
        if "High" in severity:
            return "Immediate clinical evaluation recommended. Consider emergency department visit if symptomatic."
        elif "Moderate" in severity:
            return "Schedule follow-up with primary care physician within 1-2 weeks. Monitor symptoms closely."
        else:
            return "Routine follow-up as clinically indicated. Return if symptoms worsen."

    def _create_report_summary(self, key_findings: List[KeyFinding], disease_risks: List[DiseaseRisk]) -> str:
        """Create a concise summary of the report"""
        if not key_findings:
            return "Normal chest X-ray with no acute abnormalities identified."
        
        main_finding = key_findings[0].finding
        top_risk = disease_risks[0] if disease_risks else None
        
        if top_risk and top_risk.probability > 0.5:
            return f"Abnormal chest X-ray showing {main_finding}. Most likely diagnosis: {top_risk.disease} ({top_risk.probability:.1%} probability)."
        else:
            return f"Chest X-ray shows {main_finding}. Clinical correlation recommended for definitive diagnosis."

    def _assess_clinical_significance(self, disease_risks: List[DiseaseRisk], severity: str) -> str:
        """Assess the clinical significance of findings"""
        if "High" in severity:
            return "Clinically significant findings requiring immediate attention and treatment."
        elif "Moderate" in severity:
            return "Moderately significant findings that warrant close monitoring and follow-up."
        else:
            return "Findings of low clinical significance. Routine care and monitoring appropriate."

    def _find_anatomical_location(self, report_lower: str, pattern: str) -> str:
        """Find anatomical location of findings"""
        for location, indicators in self.anatomical_locations.items():
            for indicator in indicators:
                if re.search(indicator, report_lower):
                    return location.replace('_', ' ').title()
        return "Unspecified"

    def _determine_finding_significance(self, report_lower: str, pattern: str) -> str:
        """Determine the clinical significance of a finding"""
        if any(re.search(ind, report_lower) for ind in self.severity_indicators['high']):
            return "High clinical significance - requires immediate attention"
        elif any(re.search(ind, report_lower) for ind in self.severity_indicators['moderate']):
            return "Moderate clinical significance - follow-up recommended"
        else:
            return "Mild clinical significance - monitor as needed"