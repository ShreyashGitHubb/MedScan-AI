import { useState } from "react";
import { ArrowLeft, FileText, Upload, Stethoscope, Activity, Heart, Zap, AlertTriangle, CheckCircle, Clock, User, Calendar, TrendingUp, Shield, Brain, Sparkles, Star, Bot, CheckCircle2, XCircle, AlertCircle } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import "../styles/xray-analysis.css";

interface XrayPrediction {
  model: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface DiseaseRisk {
  disease: string;
  probability: number;
  severity: string;
  description: string;
}

interface MedicalSuggestion {
  category: string;
  suggestion: string;
  priority: string;
}

interface KeyFinding {
  finding: string;
  significance: string;
  location?: string;
}

interface EnhancedXrayAnalysis {
  model: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  key_findings: string[];
  disease_risks: Record<string, {
    probability: number;
    severity: string;
    description: string;
  }>;
  medical_suggestions: string[];
  severity_assessment: string;
  follow_up_recommendations: string;
  report_summary: string;
  clinical_significance: string;
  
  // Gemini AI enhancements
  gemini_enhanced_findings?: string[];
  gemini_corrected_diagnosis?: string;
  gemini_confidence_assessment?: number;
  gemini_clinical_recommendations?: string[];
  gemini_contradictions_found?: string[];
  gemini_missing_elements?: string[];
  gemini_report_quality_score?: number;
  gemini_enhanced_summary?: string;
  gemini_differential_diagnoses?: string[];
  gemini_urgency_level?: string;
  gemini_follow_up_timeline?: string;
  gemini_clinical_reasoning?: string;
  
  // Quality metrics
  analysis_quality_score?: number;
  gemini_review_status?: string;
  processing_timestamp?: string;
  
  // Advanced Gemini features
  patient_summary?: {
    condition_explanation?: string;
    simplified_findings?: string[];
    next_steps?: string[];
    questions_for_doctor?: string[];
  };
  clinical_decision_support?: {
    treatment_guidelines?: string[];
    monitoring_protocols?: string[];
    red_flags?: string[];
    contraindications?: string[];
  };
  validation_results?: {
    consistency_score?: number;
    findings_correlation?: string;
    quality_metrics?: Record<string, number>;
  };
  enhanced_confidence_metrics?: {
    diagnostic_certainty?: number;
    clinical_correlation?: number;
    evidence_strength?: number;
    recommendation_confidence?: number;
  };
}

const XrayAnalysis = () => {
  const [textInput, setTextInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<XrayPrediction | null>(null);
  const [enhancedAnalysis, setEnhancedAnalysis] = useState<EnhancedXrayAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'basic' | 'enhanced'>('enhanced');
  const { toast } = useToast();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Accept image files and PDFs
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp', 'application/pdf'];
      const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.pdf'];
      
      const isValidType = validTypes.includes(file.type) || validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
      
      if (isValidType) {
        setSelectedFile(file);
        
        // Create image preview for image files
        if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = (e) => {
            setImagePreview(e.target?.result as string);
          };
          reader.readAsDataURL(file);
        } else {
          // For PDF files, we can't preview but we can show file info
          setImagePreview(null);
        }
        
        toast({
          title: "File Selected",
          description: `${file.name} ready for analysis`,
        });
      } else {
        toast({
          title: "Invalid file type",
          description: "Please select an image file (JPG, PNG, BMP, TIFF, WebP) or PDF",
          variant: "destructive",
        });
      }
    }
  };

  const analyzeText = async (text: string, method: 'direct' | 'file', retryCount = 0) => {
    setIsLoading(true);
    setPrediction(null);
    setEnhancedAnalysis(null);

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 300000); // 5 minute timeout for enhanced analysis (increased for improved accuracy)

    try {
      let response;
      
      if (analysisMode === 'enhanced') {
        // Enhanced analysis with Gemini AI medical insights
        if (method === 'direct') {
          const formData = new FormData();
          formData.append('text', text);
          
          response = await fetch('http://localhost:8000/gemini-analyze-text/xray', {
            method: 'POST',
            body: formData,
            signal: controller.signal,
          });
        } else {
          if (!selectedFile) {
            throw new Error('No file selected');
          }
          
          const formData = new FormData();
          formData.append('file', selectedFile);
          
          // Try the new ultra-enhanced endpoint first, fallback to regular if needed
          try {
            response = await fetch('http://localhost:8000/ultra-enhanced-analyze/xray', {
              method: 'POST',
              body: formData,
              signal: controller.signal,
            });
          } catch (ultraError) {
            console.log('Ultra-enhanced endpoint not available, using regular endpoint');
            response = await fetch('http://localhost:8000/gemini-analyze/xray-image', {
              method: 'POST',
              body: formData,
              signal: controller.signal,
            });
          }
        }

        if (!response.ok) {
          throw new Error(`Enhanced analysis failed: ${response.statusText}`);
        }

        const result = await response.json();
        setEnhancedAnalysis(result);
        
        toast({
          title: "Gemini Enhanced Analysis Complete",
          description: `Advanced AI medical analysis completed with Gemini enhancements - ${Object.keys(result.disease_risks || {}).length} disease risks identified`,
        });
      } else {
        // Basic analysis
        if (method === 'direct') {
          const formData = new FormData();
          formData.append('text', text);
          
          response = await fetch('http://localhost:8000/predict-text/xray', {
            method: 'POST',
            body: formData,
            signal: controller.signal,
          });
        } else {
          if (!selectedFile) {
            throw new Error('No file selected');
          }
          
          const formData = new FormData();
          formData.append('file', selectedFile);
          
          response = await fetch('http://localhost:8000/predict/xray-image', {
            method: 'POST',
            body: formData,
            signal: controller.signal,
          });
        }

        if (!response.ok) {
          throw new Error(`Analysis failed: ${response.statusText}`);
        }

        const result = await response.json();
        setPrediction(result);
        
        toast({
          title: "Analysis Complete",
          description: `X-ray report classified as: ${result.predicted_class}`,
        });
      }
    } catch (error) {
      console.error('Analysis error:', error);
      
      // Retry logic for network errors (but not for AbortError/timeout)
      if (retryCount < 2 && error instanceof Error && 
          (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) &&
          !error.name.includes('AbortError')) {
        console.log(`Retrying analysis (attempt ${retryCount + 1}/2)...`);
        clearTimeout(timeoutId);
        setIsLoading(false);
        setTimeout(() => analyzeText(text, method, retryCount + 1), 2000);
        return;
      }
      
      let errorMessage = "An error occurred during analysis";
      let errorTitle = "Analysis Failed";
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorTitle = "Analysis Timeout";
          errorMessage = "The enhanced analysis is taking longer than expected (>5 minutes). This might be due to complex medical analysis or server load. Please try with Basic Analysis mode or try again later.";
        } else if (error.message.includes('Failed to fetch')) {
          errorTitle = "Connection Error";
          errorMessage = "Cannot connect to the backend server. Please ensure the backend is running on http://localhost:8000";
        } else if (error.message.includes('NetworkError')) {
          errorTitle = "Network Error";
          errorMessage = "Network connection failed. Please check your internet connection and try again.";
        } else if (error.message.includes('500')) {
          errorTitle = "Server Error";
          errorMessage = "Internal server error occurred. The backend might be processing or experiencing issues.";
        } else {
          errorMessage = error.message;
        }
      }
      
      toast({
        title: errorTitle,
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      clearTimeout(timeoutId);
      setIsLoading(false);
    }
  };

  const handleDirectAnalysis = () => {
    if (!textInput.trim()) {
      toast({
        title: "No text provided",
        description: "Please enter some text to analyze",
        variant: "destructive",
      });
      return;
    }
    analyzeText(textInput, 'direct');
  };

  const handleFileAnalysis = () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select a text file to analyze",
        variant: "destructive",
      });
      return;
    }
    analyzeText(textInput, 'file');
  };

  const getClassificationColor = (className: string) => {
    return className.toLowerCase() === 'normal' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getSeverityColor = (severity: string) => {
    if (severity.toLowerCase().includes('high')) return 'bg-red-100 text-red-800 border-red-200';
    if (severity.toLowerCase().includes('moderate')) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    return 'bg-green-100 text-green-800 border-green-200';
  };

  const getPriorityColor = (priority: string) => {
    if (priority.toLowerCase() === 'high') return 'bg-red-50 border-red-200 text-red-700';
    if (priority.toLowerCase() === 'medium') return 'bg-yellow-50 border-yellow-200 text-yellow-700';
    return 'bg-blue-50 border-blue-200 text-blue-700';
  };

  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case 'immediate': return '🚨';
      case 'follow_up': return '📅';
      case 'lifestyle': return '🏃';
      case 'monitoring': return '👁️';
      default: return '💡';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <Header />
      
      {/* Hero Section */}
      <section className="py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        <div className="container mx-auto px-4 relative">
          <div className="flex items-center mb-8">
            <Link to="/">
              <Button variant="ghost" size="sm" className="mr-4 hover:bg-white/20 transition-colors">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Skin Analysis
              </Button>
            </Link>
          </div>
          
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative mb-8">
              <div className="w-20 h-20 rounded-full bg-gradient-primary flex items-center justify-center mx-auto mb-6 shadow-glow animate-pulse">
                <Stethoscope className="w-10 h-10 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
                <Brain className="w-3 h-3 text-white" />
              </div>
            </div>
            <h1 className="text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-6">
              Advanced X-ray Analysis
            </h1>
            <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
              Professional AI-powered analysis enhanced by Google's Gemini AI. Get comprehensive medical insights, 
              disease risk assessment, corrected diagnoses, and clinical recommendations. Upload medical report images or PDFs - our OCR technology will extract and analyze the text.
            </p>
            <div className="flex items-center justify-center space-x-8 text-sm text-gray-500 dark:text-gray-400">
              <div className="flex items-center space-x-2">
                <Shield className="w-4 h-4 text-accent" />
                <span className="text-muted-foreground">HIPAA Compliant</span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-primary" />
                <span className="text-muted-foreground">Instant Results</span>
              </div>
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-primary" />
                <span className="text-muted-foreground">Medical Grade</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Analysis Interface */}
      <section className="py-20 relative">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 gap-12 max-w-5xl mx-auto">
            
            {/* Input & Samples */}
            <div className="space-y-8">
              <Card className="shadow-card hover:shadow-medical">
                <CardHeader className="bg-card">
                  <CardTitle className="flex items-center text-xl">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center mr-3 bg-primary">
                      <FileText className="w-4 h-4 text-white" />
                    </div>
                    X-ray Report Analysis
                  </CardTitle>
                  <CardDescription className="text-base">
                    Choose your preferred method: type text directly or upload medical report images/PDFs for OCR processing and AI analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {/* Analysis Mode Selector */}
                  <div className="mb-6 p-6 rounded-xl border bg-card shadow-sm">
                    <div className="flex items-center mb-4">
                      <Brain className="w-5 h-5 text-primary mr-2" />
                      <Label className="text-base font-semibold">Analysis Mode</Label>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <label className={`flex items-start space-x-3 p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                        analysisMode === 'enhanced' 
                          ? 'border-primary bg-secondary/30 shadow-sm' 
                          : 'border-border hover:border-primary/60'
                      }`}>
                        <input
                          type="radio"
                          name="analysisMode"
                          value="enhanced"
                          checked={analysisMode === 'enhanced'}
                          onChange={(e) => setAnalysisMode(e.target.value as 'basic' | 'enhanced')}
                          className="mt-1 text-primary focus:ring-primary"
                        />
                        <div className="flex-1">
                          <div className="flex items-center mb-2">
                            <Bot className="w-4 h-4 text-purple-600 mr-2" />
                            <span className="font-semibold">Gemini Enhanced Analysis</span>
                            <Badge className="ml-2 bg-purple-100 text-purple-800 text-xs">🤖 AI Powered</Badge>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            Google Gemini AI medical review with enhanced findings, corrected diagnoses, clinical reasoning, differential diagnosis, and quality assessment
                          </p>
                        </div>
                      </label>
                      
                      <label className={`flex items-start space-x-3 p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                        analysisMode === 'basic' 
                          ? 'border-primary bg-secondary/30 shadow-sm' 
                          : 'border-border hover:border-primary/60'
                      }`}>
                        <input
                          type="radio"
                          name="analysisMode"
                          value="basic"
                          checked={analysisMode === 'basic'}
                          onChange={(e) => setAnalysisMode(e.target.value as 'basic' | 'enhanced')}
                          className="mt-1 text-primary focus:ring-primary"
                        />
                        <div className="flex-1">
                          <div className="flex items-center mb-2">
                            <CheckCircle className="w-4 h-4 text-blue-600 mr-2" />
                            <span className="font-semibold">Basic Analysis</span>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            Simple classification with confidence scores and probability breakdown
                          </p>
                        </div>
                      </label>
                    </div>
                  </div>

                  <Tabs defaultValue="text" className="w-full">
                    <TabsList role="tablist" aria-orientation="horizontal" className="grid w-full grid-cols-2 bg-secondary/40 p-1 rounded-lg">
                      <TabsTrigger value="text" className="flex items-center space-x-2 data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:shadow-sm transition-all">
                        <FileText className="w-4 h-4" />
                        <span>Direct Text</span>
                      </TabsTrigger>
                      <TabsTrigger value="file" className="flex items-center space-x-2 data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:shadow-sm transition-all">
                        <Upload className="w-4 h-4" />
                        <span>Upload File</span>
                      </TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="text" className="space-y-6 mt-6">
                      <div className="space-y-3">
                        <div className="flex items-center space-x-2">
                          <FileText className="w-4 h-4 text-blue-600" />
                          <Label htmlFor="text-input" className="font-medium">Paste X-ray Report</Label>
                        </div>
                        <Textarea
                          id="text-input"
                          placeholder="Enter your X-ray report here... 

Example:
CLINICAL HISTORY: 45-year-old male with cough and fever.

FINDINGS: 
The lungs show patchy consolidation in the right lower lobe. 
There is increased opacity consistent with pneumonia. 
No pleural effusion is identified.

IMPRESSION: 
Right lower lobe pneumonia."
                          value={textInput}
                          onChange={(e) => setTextInput(e.target.value)}
                          className="min-h-[200px] resize-none border-2 focus:border-blue-500 transition-colors"
                        />
                        <div className="flex justify-between items-center">
                          <p className="text-sm text-muted-foreground">
                            {textInput.length}/1000 characters
                          </p>
                          {textInput.length > 800 && (
                            <p className="text-sm text-amber-600">
                              <AlertTriangle className="w-3 h-3 inline mr-1" />
                              Approaching limit
                            </p>
                          )}
                        </div>
                      </div>
                      <Button 
                        onClick={handleDirectAnalysis}
                        disabled={isLoading || !textInput.trim()}
                        className="w-full h-12 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium shadow-lg hover:shadow-xl transition-all duration-200"
                      >
                        {isLoading ? (
                          <div className="flex items-center space-x-2">
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            <span>{analysisMode === 'enhanced' ? 'Enhanced AI Analysis in Progress...' : 'Analyzing Report...'}</span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <Activity className="w-4 h-4" />
                            <span>Analyze Text</span>
                          </div>
                        )}
                      </Button>
                    </TabsContent>
                    
                    <TabsContent value="file" className="space-y-6 mt-6">
                      <div className="space-y-3">
                        <div className="flex items-center space-x-2">
                          <Upload className="w-4 h-4 text-blue-600" />
                          <Label htmlFor="file-input" className="font-medium">Upload Medical Report Image</Label>
                        </div>
                        <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl p-8 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors duration-200 bg-gray-50/50 dark:bg-gray-800/50">
                          <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Upload className="w-8 h-8 text-blue-600" />
                          </div>
                          <p className="text-base font-medium text-gray-900 dark:text-gray-100 mb-2">
                            Drop your medical report image here
                          </p>
                          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                            or click to browse files
                          </p>
                          <p className="text-xs text-gray-400 dark:text-gray-500 mb-6">
                            Supports: JPG, PNG, BMP, TIFF, WebP, PDF (max 10MB)
                          </p>
                          <input
                            id="file-input"
                            type="file"
                            accept="image/*,.pdf"
                            onChange={handleFileSelect}
                            className="hidden"
                          />
                          <Button
                            variant="outline"
                            onClick={() => document.getElementById('file-input')?.click()}
                            className="border-2 border-blue-200 hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                          >
                            <Upload className="w-4 h-4 mr-2" />
                            Select File
                          </Button>
                        </div>
                        {selectedFile && (
                          <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                            <div className="flex items-start space-x-3">
                              <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                                <FileText className="w-5 h-5 text-green-600" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-green-900 dark:text-green-100 truncate">{selectedFile.name}</p>
                                <p className="text-xs text-green-600 dark:text-green-400">
                                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for OCR analysis
                                </p>
                                {imagePreview && (
                                  <div className="mt-3">
                                    <img 
                                      src={imagePreview} 
                                      alt="Preview" 
                                      className="max-w-full h-32 object-contain rounded-md border border-green-200 dark:border-green-700"
                                    />
                                  </div>
                                )}
                              </div>
                              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                            </div>
                          </div>
                        )}
                      </div>
                      <Button 
                        onClick={handleFileAnalysis}
                        disabled={isLoading || !selectedFile}
                        className="w-full h-12 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium shadow-lg hover:shadow-xl transition-all duration-200"
                      >
                        {isLoading ? (
                          <div className="flex items-center space-x-2">
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            <span>{analysisMode === 'enhanced' ? 'Enhanced AI Analysis in Progress...' : 'Analyzing File...'}</span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <Upload className="w-4 h-4" />
                            <span>Analyze File</span>
                          </div>
                        )}
                      </Button>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>

              {/* Sample Reports */}
              <Card className="border-2 border-gray-100 dark:border-gray-800 shadow-sm">
                <CardHeader className="pb-4">
                  <div className="flex items-center space-x-2">
                    <FileText className="w-5 h-5 text-indigo-600" />
                    <CardTitle className="text-lg">Sample Reports</CardTitle>
                  </div>
                  <CardDescription>
                    Try these example reports to test the enhanced analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-3">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setTextInput("CLINICAL HISTORY: 32-year-old female with chest pain.\n\nFINDINGS: Clear lung fields bilaterally. No consolidation or infiltrates. Normal heart size and mediastinal contours. No pleural effusion or pneumothorax.\n\nIMPRESSION: Normal chest X-ray.")}
                      className="w-full justify-start text-left h-auto p-4 border-2 hover:border-green-300 hover:bg-green-50 dark:hover:bg-green-900/20 transition-all duration-200"
                    >
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                          <CheckCircle className="w-4 h-4 text-green-600" />
                        </div>
                        <div className="flex-1">
                          <p className="font-semibold text-green-800 dark:text-green-200">Normal Chest X-ray</p>
                          <p className="text-sm text-green-600 dark:text-green-400 mt-1">Clear lung fields, no abnormalities detected</p>
                          <Badge className="mt-2 bg-green-100 text-green-800 text-xs">Expected: Normal</Badge>
                        </div>
                      </div>
                    </Button>
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setTextInput("CLINICAL HISTORY: 45-year-old male with cough and fever for 5 days.\n\nFINDINGS: The lungs show patchy consolidation in the right lower lobe. There is increased opacity consistent with pneumonia. No pleural effusion is identified.\n\nIMPRESSION: Right lower lobe pneumonia.")}
                      className="w-full justify-start text-left h-auto p-4 border-2 hover:border-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-200"
                    >
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                          <AlertTriangle className="w-4 h-4 text-red-600" />
                        </div>
                        <div className="flex-1">
                          <p className="font-semibold text-red-800 dark:text-red-200">Pneumonia Case</p>
                          <p className="text-sm text-red-600 dark:text-red-400 mt-1">Right lower lobe consolidation with fever</p>
                          <Badge className="mt-2 bg-red-100 text-red-800 text-xs">Expected: Abnormal</Badge>
                        </div>
                      </div>
                    </Button>
                  </div>
                  
                  <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center">
                      <Zap className="w-3 h-3 mr-1" />
                      Click any sample to auto-fill and test the enhanced analysis
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Results */}
            <div className="space-y-6">
              {enhancedAnalysis ? (
                <div className="enhanced-analysis-wrapper">
                  {(() => {
                    try {
                      return (
                        <>
                  {/* Report Summary */}
                  <Card className="border-2 border-blue-100 dark:border-blue-900 shadow-lg">
                    <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50">
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Brain className="w-5 h-5 text-blue-600" />
                          <span>Analysis Summary</span>
                        </div>
                        <Badge className={`${getClassificationColor(enhancedAnalysis.predicted_class)} text-sm px-3 py-1 shadow-sm`}>
                          {enhancedAnalysis.predicted_class.toUpperCase()}
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6 pt-6">
                      <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 rounded-xl border border-blue-200 dark:border-blue-800">
                        <div className="flex items-start space-x-3">
                          <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-lg flex items-center justify-center flex-shrink-0">
                            <FileText className="w-4 h-4 text-blue-600" />
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">Clinical Summary</p>
                            <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">{enhancedAnalysis.report_summary}</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-3">
                          <div className="flex items-center space-x-2">
                            <TrendingUp className="w-4 h-4 text-green-600" />
                            <Label className="text-sm font-semibold">Confidence Score</Label>
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-gray-500">Prediction Accuracy</span>
                              <span className={`text-sm font-bold ${getConfidenceColor(enhancedAnalysis.confidence)}`}>
                                {(enhancedAnalysis.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress value={enhancedAnalysis.confidence * 100} className="h-3 bg-gray-200 dark:bg-gray-700" />
                          </div>
                        </div>
                        
                        <div className="space-y-3">
                          <div className="flex items-center space-x-2">
                            <AlertTriangle className="w-4 h-4 text-orange-600" />
                            <Label className="text-sm font-semibold">Clinical Severity</Label>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Badge className={`${getSeverityColor(enhancedAnalysis.severity_assessment)} px-3 py-1 text-sm shadow-sm`}>
                              {enhancedAnalysis.severity_assessment.split(' - ')[0]}
                            </Badge>
                            <span className="text-xs text-gray-500">Risk Level</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Disease Risks */}
                  <Card className="border-2 border-orange-100 dark:border-orange-900 shadow-lg">
                    <CardHeader className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-950/50 dark:to-red-950/50">
                      <CardTitle className="flex items-center space-x-2">
                        <Heart className="w-5 h-5 text-red-600" />
                        <span>Disease Risk Assessment</span>
                      </CardTitle>
                      <CardDescription>
                        AI-powered probability analysis for potential medical conditions
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="pt-6">
                      <div className="space-y-4">
                        {Object.entries(enhancedAnalysis.disease_risks || {}).map(([diseaseName, riskData], index) => {
                          const risk = {
                            disease: diseaseName,
                            probability: riskData.probability,
                            severity: riskData.severity,
                            description: riskData.description
                          };
                          const severityLower = risk.severity.toLowerCase();
                          return (
                            <div key={index} className={`p-5 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                              severityLower === 'high' ? 'border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-900/10' :
                              severityLower === 'moderate' ? 'border-yellow-200 bg-yellow-50/50 dark:border-yellow-800 dark:bg-yellow-900/10' :
                              'border-green-200 bg-green-50/50 dark:border-green-800 dark:bg-green-900/10'
                            }`}>
                              <div className="flex justify-between items-start mb-3">
                                <div className="flex items-center space-x-3">
                                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                                    severityLower === 'high' ? 'bg-red-100 dark:bg-red-900/30' :
                                    severityLower === 'moderate' ? 'bg-yellow-100 dark:bg-yellow-900/30' :
                                    'bg-green-100 dark:bg-green-900/30'
                                  }`}>
                                    {severityLower === 'high' ? <AlertTriangle className="w-5 h-5 text-red-600" /> :
                                     severityLower === 'moderate' ? <Clock className="w-5 h-5 text-yellow-600" /> :
                                     <CheckCircle className="w-5 h-5 text-green-600" />}
                                  </div>
                                  <h4 className="font-semibold text-gray-900 dark:text-gray-100">{risk.disease}</h4>
                                </div>
                                <div className="flex items-center space-x-3">
                                  <Badge className={`${getSeverityColor(risk.severity)} px-3 py-1 text-xs font-medium shadow-sm`}>
                                    {risk.severity.charAt(0).toUpperCase() + risk.severity.slice(1)}
                                  </Badge>
                                  <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                                    {(risk.probability * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                              <div className="space-y-2 mb-3">
                                <div className="flex justify-between text-xs text-gray-500">
                                  <span>Risk Probability</span>
                                  <span>{risk.probability > 0.7 ? 'High Risk' : risk.probability > 0.4 ? 'Moderate Risk' : 'Low Risk'}</span>
                                </div>
                                <Progress value={risk.probability * 100} className="h-2" />
                              </div>
                              <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">{risk.description}</p>
                            </div>
                          );
                        })}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Medical Suggestions */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Medical Recommendations</CardTitle>
                      <CardDescription>
                        Actionable suggestions based on findings
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {enhancedAnalysis.medical_suggestions.map((suggestion, index) => (
                          <div key={index} className="p-4 border border-blue-200 rounded-lg bg-blue-50/50 dark:border-blue-800 dark:bg-blue-900/10">
                            <div className="flex items-start space-x-3">
                              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-lg flex items-center justify-center flex-shrink-0">
                                <Stethoscope className="w-4 h-4 text-blue-600" />
                              </div>
                              <div className="flex-1">
                                <p className="text-sm text-gray-700 dark:text-gray-300">{suggestion}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Key Findings */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Key Findings</CardTitle>
                      <CardDescription>
                        Important observations from the report
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {enhancedAnalysis.key_findings.map((finding, index) => (
                          <div key={index} className="p-4 border border-green-200 rounded-lg bg-green-50/50 dark:border-green-800 dark:bg-green-900/10">
                            <div className="flex items-start space-x-3">
                              <div className="w-8 h-8 bg-green-100 dark:bg-green-900/50 rounded-lg flex items-center justify-center flex-shrink-0">
                                <Activity className="w-4 h-4 text-green-600" />
                              </div>
                              <div className="flex-1">
                                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{finding}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Enhanced Confidence Metrics */}
                  {enhancedAnalysis.enhanced_confidence_metrics && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-2">
                          <TrendingUp className="w-5 h-5 text-blue-600" />
                          <span>Enhanced Confidence Metrics</span>
                        </CardTitle>
                        <CardDescription>
                          Advanced confidence assessment with multiple factors
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-4">
                          {enhancedAnalysis.enhanced_confidence_metrics.diagnostic_certainty && (
                            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                              <div className="text-sm font-medium text-blue-900 dark:text-blue-100">Diagnostic Certainty</div>
                              <div className="text-lg font-bold text-blue-700 dark:text-blue-300">
                                {(enhancedAnalysis.enhanced_confidence_metrics.diagnostic_certainty * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                          {enhancedAnalysis.enhanced_confidence_metrics.clinical_correlation && (
                            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <div className="text-sm font-medium text-green-900 dark:text-green-100">Clinical Correlation</div>
                              <div className="text-lg font-bold text-green-700 dark:text-green-300">
                                {(enhancedAnalysis.enhanced_confidence_metrics.clinical_correlation * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                          {enhancedAnalysis.enhanced_confidence_metrics.evidence_strength && (
                            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                              <div className="text-sm font-medium text-purple-900 dark:text-purple-100">Evidence Strength</div>
                              <div className="text-lg font-bold text-purple-700 dark:text-purple-300">
                                {(enhancedAnalysis.enhanced_confidence_metrics.evidence_strength * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                          {enhancedAnalysis.enhanced_confidence_metrics.recommendation_confidence && (
                            <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                              <div className="text-sm font-medium text-orange-900 dark:text-orange-100">Recommendation Confidence</div>
                              <div className="text-lg font-bold text-orange-700 dark:text-orange-300">
                                {(enhancedAnalysis.enhanced_confidence_metrics.recommendation_confidence * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Patient-Friendly Summary */}
                  {enhancedAnalysis.patient_summary && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-2">
                          <User className="w-5 h-5 text-green-600" />
                          <span>Patient-Friendly Summary</span>
                        </CardTitle>
                        <CardDescription>
                          Clear, easy-to-understand explanation of your results
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {enhancedAnalysis.patient_summary.condition_explanation && (
                            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <h4 className="font-medium text-green-900 dark:text-green-100 mb-2">What This Means:</h4>
                              <p className="text-sm text-green-800 dark:text-green-200">{enhancedAnalysis.patient_summary.condition_explanation}</p>
                            </div>
                          )}
                          
                          {enhancedAnalysis.patient_summary.simplified_findings && (
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Key Points:</h4>
                              <ul className="space-y-1">
                                {(Array.isArray(enhancedAnalysis.patient_summary.simplified_findings) 
                                  ? enhancedAnalysis.patient_summary.simplified_findings 
                                  : [enhancedAnalysis.patient_summary.simplified_findings]
                                ).map((finding, index) => (
                                  <li key={index} className="text-sm text-gray-700 dark:text-gray-300 flex items-start space-x-2">
                                    <span className="text-green-500 mt-1">•</span>
                                    <span>{finding}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {enhancedAnalysis.patient_summary.next_steps && (
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Next Steps:</h4>
                              <ul className="space-y-1">
                                {(Array.isArray(enhancedAnalysis.patient_summary.next_steps) 
                                  ? enhancedAnalysis.patient_summary.next_steps 
                                  : [enhancedAnalysis.patient_summary.next_steps]
                                ).map((step, index) => (
                                  <li key={index} className="text-sm text-blue-700 dark:text-blue-300 flex items-start space-x-2">
                                    <span className="text-blue-500 mt-1">→</span>
                                    <span>{step}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {enhancedAnalysis.patient_summary.questions_for_doctor && (
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Questions to Ask Your Doctor:</h4>
                              <ul className="space-y-1">
                                {(Array.isArray(enhancedAnalysis.patient_summary.questions_for_doctor) 
                                  ? enhancedAnalysis.patient_summary.questions_for_doctor 
                                  : [enhancedAnalysis.patient_summary.questions_for_doctor]
                                ).map((question, index) => (
                                  <li key={index} className="text-sm text-purple-700 dark:text-purple-300 flex items-start space-x-2">
                                    <span className="text-purple-500 mt-1">?</span>
                                    <span>{question}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Clinical Decision Support */}
                  {enhancedAnalysis.clinical_decision_support && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-2">
                          <Stethoscope className="w-5 h-5 text-red-600" />
                          <span>Clinical Decision Support</span>
                        </CardTitle>
                        <CardDescription>
                          Evidence-based medical guidance and protocols
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {enhancedAnalysis.clinical_decision_support.treatment_guidelines && (
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Treatment Guidelines:</h4>
                              <ul className="space-y-2">
                                {(Array.isArray(enhancedAnalysis.clinical_decision_support.treatment_guidelines) 
                                  ? enhancedAnalysis.clinical_decision_support.treatment_guidelines 
                                  : [enhancedAnalysis.clinical_decision_support.treatment_guidelines]
                                ).map((guideline, index) => (
                                  <li key={index} className="text-sm text-gray-700 dark:text-gray-300 p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                                    {guideline}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {enhancedAnalysis.clinical_decision_support.red_flags && (
                            <div>
                              <h4 className="font-medium text-red-900 dark:text-red-100 mb-2">⚠️ Red Flags:</h4>
                              <ul className="space-y-2">
                                {(Array.isArray(enhancedAnalysis.clinical_decision_support.red_flags) 
                                  ? enhancedAnalysis.clinical_decision_support.red_flags 
                                  : [enhancedAnalysis.clinical_decision_support.red_flags]
                                ).map((flag, index) => (
                                  <li key={index} className="text-sm text-red-700 dark:text-red-300 p-2 bg-red-50 dark:bg-red-900/20 rounded border-l-4 border-red-500">
                                    {flag}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {enhancedAnalysis.clinical_decision_support.monitoring_protocols && (
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Monitoring Protocols:</h4>
                              <ul className="space-y-1">
                                {(Array.isArray(enhancedAnalysis.clinical_decision_support.monitoring_protocols) 
                                  ? enhancedAnalysis.clinical_decision_support.monitoring_protocols 
                                  : [enhancedAnalysis.clinical_decision_support.monitoring_protocols]
                                ).map((protocol, index) => (
                                  <li key={index} className="text-sm text-gray-700 dark:text-gray-300 flex items-start space-x-2">
                                    <span className="text-blue-500 mt-1">📋</span>
                                    <span>{protocol}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Gemini Enhanced Findings */}
                  {enhancedAnalysis.gemini_enhanced_findings && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-2">
                          <Sparkles className="w-5 h-5 text-purple-600" />
                          <span>Gemini AI Enhanced Findings</span>
                        </CardTitle>
                        <CardDescription>
                          Advanced AI analysis with enhanced medical insights
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          {(Array.isArray(enhancedAnalysis.gemini_enhanced_findings) 
                            ? enhancedAnalysis.gemini_enhanced_findings 
                            : [enhancedAnalysis.gemini_enhanced_findings]
                          ).map((finding, index) => (
                            <div key={index} className="p-4 border border-purple-200 rounded-lg bg-purple-50/50 dark:border-purple-800 dark:bg-purple-900/10">
                              <div className="flex items-start space-x-3">
                                <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900/50 rounded-lg flex items-center justify-center flex-shrink-0">
                                  <Sparkles className="w-4 h-4 text-purple-600" />
                                </div>
                                <div className="flex-1">
                                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{finding}</p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        {enhancedAnalysis.gemini_clinical_reasoning && (
                          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Clinical Reasoning:</h4>
                            <p className="text-sm text-gray-700 dark:text-gray-300">{enhancedAnalysis.gemini_clinical_reasoning}</p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  )}

                  {/* Follow-up Recommendations */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Follow-up Plan</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                          <h4 className="font-medium text-yellow-800 mb-2">Next Steps</h4>
                          <p className="text-sm text-yellow-700">{enhancedAnalysis.follow_up_recommendations}</p>
                        </div>
                        <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                          <h4 className="font-medium text-gray-800 mb-2">Clinical Significance</h4>
                          <p className="text-sm text-gray-700">{enhancedAnalysis.clinical_significance}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                        </>
                      );
                    } catch (error) {
                      console.error('Error rendering enhanced analysis:', error);
                      return (
                        <Card className="border-2 border-red-200 bg-red-50">
                          <CardHeader>
                            <CardTitle className="text-red-800">Display Error</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-red-700">
                              There was an error displaying the enhanced analysis results. 
                              The analysis completed successfully, but some advanced features 
                              may not be shown correctly.
                            </p>
                            <div className="mt-4 p-3 bg-white rounded border">
                              <h4 className="font-medium mb-2">Basic Results:</h4>
                              <p><strong>Prediction:</strong> {enhancedAnalysis?.predicted_class || 'Unknown'}</p>
                              <p><strong>Confidence:</strong> {enhancedAnalysis?.confidence ? (enhancedAnalysis.confidence * 100).toFixed(1) + '%' : 'Unknown'}</p>
                            </div>
                          </CardContent>
                        </Card>
                      );
                    }
                  })()}
                </div>
              ) : prediction ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      Basic Analysis Results
                      <Badge className={getClassificationColor(prediction.predicted_class)}>
                        {prediction.predicted_class.toUpperCase()}
                      </Badge>
                    </CardTitle>
                    <CardDescription>
                      Simple AI classification of the X-ray report
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Confidence Score */}
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <Label>Confidence Score</Label>
                        <span className={`font-bold ${getConfidenceColor(prediction.confidence)}`}>
                          {(prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress 
                        value={prediction.confidence * 100} 
                        className="h-2"
                      />
                    </div>

                    {/* Probability Breakdown */}
                    <div>
                      <Label className="mb-3 block">Classification Probabilities</Label>
                      <div className="space-y-3">
                        {Object.entries(prediction.probabilities).map(([className, probability]) => (
                          <div key={className} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span className="capitalize font-medium">{className}</span>
                              <span>{(probability * 100).toFixed(1)}%</span>
                            </div>
                            <Progress value={probability * 100} className="h-1" />
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Model Information */}
                    <div className="pt-4 border-t">
                      <p className="text-sm text-muted-foreground">
                        <strong>Model:</strong> {prediction.model.toUpperCase()} (BERT-based Text Classifier)
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card className="border-2 border-dashed border-gray-300 dark:border-gray-600 shadow-lg">
                  <CardContent className="py-16 text-center">
                    <div className="relative mb-6">
                      <div className="w-20 h-20 bg-gradient-to-r from-blue-100 to-indigo-100 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse-slow">
                        <Stethoscope className="w-10 h-10 text-blue-600 animate-float" />
                      </div>
                      <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center animate-pulse">
                        <Activity className="w-3 h-3 text-white" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-3">Ready for Analysis</h3>
                    <p className="text-gray-600 dark:text-gray-300 mb-6 max-w-md mx-auto">
                      Enter or upload an X-ray report to get comprehensive AI-powered medical analysis with disease risk assessment
                    </p>
                    <div className="flex items-center justify-center space-x-6 text-sm text-gray-500">
                      <div className="flex items-center space-x-2">
                        <Brain className="w-4 h-4 text-blue-500" />
                        <span>AI Analysis</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Heart className="w-4 h-4 text-red-500" />
                        <span>Risk Assessment</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Shield className="w-4 h-4 text-green-500" />
                        <span>Medical Grade</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Information Card */}
              <Card className="border-2 border-indigo-100 dark:border-indigo-900 shadow-lg hover-lift">
                <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/50 dark:to-purple-950/50">
                  <CardTitle className="flex items-center text-lg">
                    <div className="w-8 h-8 bg-indigo-100 dark:bg-indigo-900/50 rounded-lg flex items-center justify-center mr-3">
                      <Brain className="w-4 h-4 text-indigo-600" />
                    </div>
                    About X-ray Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6 pt-6">
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                        <Zap className="w-4 h-4 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">How it works</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
                          Advanced AI model analyzes X-ray report text using natural language processing, 
                          providing classification, disease risk assessment, and clinical recommendations.
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">Supported Classifications</h4>
                        <div className="flex gap-3">
                          <Badge className="bg-green-100 text-green-800 px-3 py-1 shadow-sm">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Normal
                          </Badge>
                          <Badge className="bg-red-100 text-red-800 px-3 py-1 shadow-sm">
                            <AlertTriangle className="w-3 h-3 mr-1" />
                            Abnormal
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>

                  <Alert className="border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20">
                    <Shield className="w-4 h-4 text-amber-600" />
                    <AlertDescription className="text-sm text-amber-800 dark:text-amber-200">
                      <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
                      Always consult with qualified medical professionals for clinical decisions and diagnosis.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Floating Help Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <Button
          size="lg"
          className="w-14 h-14 rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-110"
          onClick={() => {
            const element = document.querySelector('[data-help-section]');
            element?.scrollIntoView({ behavior: 'smooth' });
          }}
        >
          <Brain className="w-6 h-6 text-white" />
        </Button>
      </div>

      {/* Enhanced Footer Section */}
      <section className="py-16 bg-gradient-to-r from-gray-900 to-blue-900 text-white" data-help-section>
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <div className="mb-8">
              <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Stethoscope className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-3xl font-bold mb-4">Advanced Medical AI Analysis</h2>
              <p className="text-xl text-gray-300 leading-relaxed">
                Empowering healthcare professionals with AI-driven insights for better patient outcomes
              </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-8 mb-12">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <Brain className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="font-semibold mb-2">AI-Powered</h3>
                <p className="text-sm text-gray-400">Advanced machine learning algorithms for accurate analysis</p>
              </div>
              
              <div className="text-center">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <Shield className="w-6 h-6 text-green-400" />
                </div>
                <h3 className="font-semibold mb-2">Secure & Private</h3>
                <p className="text-sm text-gray-400">HIPAA compliant with end-to-end encryption</p>
              </div>
              
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <Heart className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="font-semibold mb-2">Medical Grade</h3>
                <p className="text-sm text-gray-400">Designed for healthcare professionals and researchers</p>
              </div>
            </div>
            
            <div className="border-t border-white/10 pt-8">
              <p className="text-sm text-gray-400">
                © 2024 MedScanAI. Built with advanced AI for medical analysis and research.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default XrayAnalysis;