import React, { useState } from "react";
import { ArrowLeft, Brain, Activity, Zap, Shield, CheckCircle, AlertTriangle, Upload, Bot, FileText, Info, Clock, TrendingUp, Users, Heart, Stethoscope, Sparkles, Star, CheckCircle2, XCircle, AlertCircle } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

interface MRIPrediction {
  model: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  gradcam_png?: string;
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
  confidence?: number;
}

interface EnhancedMRIAnalysis {
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
  follow_up_recommendations: string[];
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
  };
}

const MRIAnalysis: React.FC = () => {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<MRIPrediction | null>(null);
  const [enhancedAnalysis, setEnhancedAnalysis] = useState<EnhancedMRIAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'basic' | 'enhanced' | 'gemini'>('gemini');
  const { toast } = useToast();

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setImage(file);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      
      toast({
        title: "Image Selected",
        description: `${file.name} ready for analysis`,
      });
    }
  };

  const handleSubmit = async () => {
    if (!image) {
      toast({
        title: "No image selected",
        description: "Please upload an MRI image first",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    setResult(null);
    setEnhancedAnalysis(null);

    const formData = new FormData();
    formData.append("file", image);

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 300000); // 5 minute timeout for enhanced analysis

    try {
      let response;
      
      if (analysisMode === 'gemini') {
        // Try Gemini enhanced analysis first
        try {
          response = await fetch("http://127.0.0.1:8000/gemini-analyze/mri-image", {
            method: "POST",
            body: formData,
            signal: controller.signal,
          });
          
          if (response.ok) {
            const enhancedData = await response.json();
            setEnhancedAnalysis(enhancedData);
            setResult({
              model: enhancedData.model,
              predicted_class: enhancedData.predicted_class,
              confidence: enhancedData.confidence,
              probabilities: enhancedData.probabilities
            });
            
            toast({
              title: "Gemini Analysis Complete",
              description: `MRI analyzed with advanced AI: ${enhancedData.predicted_class}`,
            });
            return;
          }
        } catch (enhancedError) {
          console.log('Gemini analysis not available, falling back to enhanced');
        }
      }
      
      if (analysisMode === 'enhanced' || analysisMode === 'gemini') {
        // Try enhanced analysis
        try {
          response = await fetch("http://127.0.0.1:8000/analyze/mri", {
            method: "POST",
            body: formData,
            signal: controller.signal,
          });
          
          if (response.ok) {
            const enhancedData = await response.json();
            setEnhancedAnalysis(enhancedData);
            setResult({
              model: enhancedData.model,
              predicted_class: enhancedData.predicted_class,
              confidence: enhancedData.confidence,
              probabilities: enhancedData.probabilities
            });
            
            toast({
              title: "Enhanced Analysis Complete",
              description: `MRI analyzed with medical insights: ${enhancedData.predicted_class}`,
            });
            return;
          }
        } catch (enhancedError) {
          console.log('Enhanced analysis not available, falling back to basic');
        }
      }
      
      // Basic analysis fallback
      response = await fetch("http://127.0.0.1:8000/predict/mri", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
      
      toast({
        title: "Analysis Complete",
        description: `MRI classified as: ${data.predicted_class}`,
      });
      
    } catch (err) {
      if (err.name === 'AbortError') {
        toast({
          title: "Analysis Timeout",
          description: "Analysis took too long. Please try again with a smaller image.",
          variant: "destructive",
        });
      } else {
        console.error(err);
        toast({
          title: "Analysis Failed",
          description: "Error analyzing MRI. Please try again.",
          variant: "destructive",
        });
      }
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  const getClassificationColor = (className: string) => {
    const lowerClass = className.toLowerCase();
    if (lowerClass.includes('notumor') || lowerClass.includes('normal')) {
      return 'bg-green-100 text-green-800';
    }
    return 'bg-red-100 text-red-800';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <Header />
      
      {/* Hero Section */}
      <section className="py-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        <div className="container mx-auto px-4 relative">
          <div className="flex items-center mb-8">
            <Link to="/">
              <Button variant="ghost" size="sm" className="mr-4 hover:bg-white/20 transition-colors">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Home
              </Button>
            </Link>
          </div>
          
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative mb-8">
              <div className="w-20 h-20 rounded-full bg-gradient-primary flex items-center justify-center mx-auto mb-6 shadow-glow animate-pulse">
                <Brain className="w-10 h-10 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center animate-bounce">
                <CheckCircle className="w-4 h-4 text-white" />
              </div>
            </div>
            
            <h1 className="text-5xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
              Advanced MRI Analysis
            </h1>
            
            <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
              Professional AI-powered analysis for MRI scans. Get tumor classification, probability breakdown, urgency estimation, 
              and AI-driven doctor recommendations.
            </p>
            
            <div className="flex flex-wrap justify-center gap-6 text-sm">
              <div className="flex items-center space-x-2 bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                <Shield className="w-4 h-4 text-blue-600" />
                <span className="text-gray-700 dark:text-gray-300">Medical Grade</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                <Zap className="w-4 h-4 text-yellow-600" />
                <span className="text-gray-700 dark:text-gray-300">Instant Results</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                <Activity className="w-4 h-4 text-green-600" />
                <span className="text-gray-700 dark:text-gray-300">HIPAA Compliant</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Analysis Interface */}
      <section className="py-20 relative">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-12 max-w-7xl mx-auto">
            
            {/* Left Column - Upload Interface */}
            <div className="space-y-8">
              <Card className="border-2 border-gray-200 dark:border-gray-700 shadow-xl hover-lift">
                <CardHeader className="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/30">
                  <CardTitle className="flex items-center text-xl">
                    <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-lg flex items-center justify-center mr-3">
                      <Brain className="w-4 h-4 text-blue-600" />
                    </div>
                    MRI Brain Scan Analysis
                  </CardTitle>
                  <CardDescription className="text-base">
                    Upload your MRI brain scan image for AI-powered tumor detection and classification
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {/* Analysis Mode Selector */}
                  <div className="mb-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50 rounded-xl border border-blue-200 dark:border-blue-800 shadow-sm">
                    <div className="flex items-center mb-4">
                      <Brain className="w-5 h-5 text-blue-600 mr-2" />
                      <Label className="text-base font-semibold text-blue-900 dark:text-blue-100">Analysis Mode</Label>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <label className={`flex items-start space-x-3 p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                        analysisMode === 'gemini' 
                          ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30 shadow-md' 
                          : 'border-gray-200 dark:border-gray-700 hover:border-purple-300 dark:hover:border-purple-600'
                      }`}>
                        <input
                          type="radio"
                          name="analysisMode"
                          value="gemini"
                          checked={analysisMode === 'gemini'}
                          onChange={(e) => setAnalysisMode(e.target.value as 'basic' | 'enhanced' | 'gemini')}
                          className="mt-1 text-purple-600 focus:ring-purple-500"
                        />
                        <div className="flex-1">
                          <div className="flex items-center mb-2">
                            <Sparkles className="w-4 h-4 text-purple-600 mr-2" />
                            <span className="font-semibold text-gray-900 dark:text-gray-100">Gemini AI Analysis</span>
                            <Badge className="ml-2 bg-purple-100 text-purple-800 text-xs">✨ Advanced</Badge>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            Google Gemini AI with comprehensive medical insights and clinical recommendations
                          </p>
                        </div>
                      </label>
                      
                      {/* <label className={`flex items-start space-x-3 p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                        analysisMode === 'enhanced' 
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 shadow-md' 
                          : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600'
                      }`}>
                        <input
                          type="radio"
                          name="analysisMode"
                          value="enhanced"
                          checked={analysisMode === 'enhanced'}
                          onChange={(e) => setAnalysisMode(e.target.value as 'basic' | 'enhanced' | 'gemini')}
                          className="mt-1 text-blue-600 focus:ring-blue-500"
                        />
                        <div className="flex-1">
                          <div className="flex items-center mb-2">
                            <Bot className="w-4 h-4 text-blue-600 mr-2" />
                            <span className="font-semibold text-gray-900 dark:text-gray-100">Enhanced Analysis</span>
                            <Badge className="ml-2 bg-blue-100 text-blue-800 text-xs">🔬 Medical</Badge>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            Enhanced medical analysis with key findings and clinical assessment
                          </p>
                        </div>
                      </label> */}
                      
                      <label className={`flex items-start space-x-3 p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                        analysisMode === 'basic' 
                          ? 'border-green-500 bg-green-50 dark:bg-green-900/30 shadow-md' 
                          : 'border-gray-200 dark:border-gray-700 hover:border-green-300 dark:hover:border-green-600'
                      }`}>
                        <input
                          type="radio"
                          name="analysisMode"
                          value="basic"
                          checked={analysisMode === 'basic'}
                          onChange={(e) => setAnalysisMode(e.target.value as 'basic' | 'enhanced' | 'gemini')}
                          className="mt-1 text-green-600 focus:ring-green-500"
                        />
                        <div className="flex-1">
                          <div className="flex items-center mb-2">
                            <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                            <span className="font-semibold text-gray-900 dark:text-gray-100">Basic Analysis</span>
                            <Badge className="ml-2 bg-green-100 text-green-800 text-xs">⚡ Fast</Badge>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            Quick tumor classification with confidence scores
                          </p>
                        </div>
                      </label>
                    </div>
                  </div>

                  {/* Upload Section */}
                  <div className="space-y-6">
                    <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl p-8 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-all duration-200 bg-gradient-to-br from-gray-50 to-blue-50/30 dark:from-gray-800 dark:to-blue-900/20">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                        id="mri-upload"
                      />
                      <label htmlFor="mri-upload" className="cursor-pointer">
                        <div className="space-y-4">
                          <div className="w-16 h-16 mx-auto bg-gradient-to-r from-blue-100 to-indigo-100 dark:from-blue-900/50 dark:to-indigo-900/50 rounded-full flex items-center justify-center shadow-lg">
                            <Upload className="w-8 h-8 text-blue-600" />
                          </div>
                          <div>
                            <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
                              {image ? image.name : "Choose MRI brain scan image"}
                            </p>
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              Supports PNG, JPG, JPEG formats • Max 10MB
                            </p>
                            <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
                              <p className="text-xs text-amber-700 dark:text-amber-300 flex items-center justify-center">
                                <AlertTriangle className="w-4 h-4 mr-1" />
                                Please upload actual MRI brain scan images only
                              </p>
                            </div>
                          </div>
                        </div>
                      </label>
                    </div>

                    {imagePreview && (
                      <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg border shadow-sm">
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Image Preview</p>
                        <img
                          src={imagePreview}
                          alt="MRI Preview"
                          className="max-w-xs mx-auto rounded-lg shadow-md border"
                        />
                      </div>
                    )}

                    <Button
                      onClick={handleSubmit}
                      disabled={loading || !image}
                      className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white py-4 text-lg font-medium shadow-lg hover:shadow-xl transition-all duration-200"
                    >
                      {loading ? (
                        <div className="flex items-center justify-center gap-3">
                          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                          {analysisMode === 'gemini' ? 'Running Gemini AI Analysis...' : 
                           analysisMode === 'enhanced' ? 'Running Enhanced Analysis...' : 'Analyzing MRI...'}
                        </div>
                      ) : (
                        <div className="flex items-center justify-center gap-2">
                          <Brain className="w-5 h-5" />
                          {analysisMode === 'gemini' ? 'Run Gemini AI Analysis' : 
                           analysisMode === 'enhanced' ? 'Run Enhanced Analysis' : 'Analyze MRI'}
                        </div>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Column - Information */}
            <div className="space-y-6">
              <Card className="border-2 border-blue-200 dark:border-blue-800 shadow-lg bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50">
                <CardHeader>
                  <CardTitle className="flex items-center text-blue-900 dark:text-blue-100">
                    <Info className="w-5 h-5 mr-2" />
                    About MRI Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-sm text-blue-800 dark:text-blue-200">
                  <div className="flex items-start space-x-3">
                    <Brain className="w-4 h-4 mt-0.5 text-blue-600" />
                    <div>
                      <p className="font-medium">Tumor Detection</p>
                      <p className="text-blue-700 dark:text-blue-300">Identifies glioma, meningioma, pituitary tumors, and normal brain tissue</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <TrendingUp className="w-4 h-4 mt-0.5 text-blue-600" />
                    <div>
                      <p className="font-medium">Confidence Scoring</p>
                      <p className="text-blue-700 dark:text-blue-300">Provides probability breakdown for each tumor type</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Clock className="w-4 h-4 mt-0.5 text-blue-600" />
                    <div>
                      <p className="font-medium">Fast Results</p>
                      <p className="text-blue-700 dark:text-blue-300">Basic analysis in seconds, enhanced analysis in 1-2 minutes</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-2 border-amber-200 dark:border-amber-800 shadow-lg bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-950/50 dark:to-orange-950/50">
                <CardHeader>
                  <CardTitle className="flex items-center text-amber-900 dark:text-amber-100">
                    <AlertTriangle className="w-5 h-5 mr-2" />
                    Important Notice
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-amber-800 dark:text-amber-200">
                  <p>• This tool is for educational and research purposes only</p>
                  <p>• Not a substitute for professional medical diagnosis</p>
                  <p>• Always consult qualified healthcare professionals</p>
                  <p>• Upload only actual MRI brain scan images</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {(result || enhancedAnalysis) && (
        <section className="py-16 bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-blue-950">
          <div className="container mx-auto px-4">
            <div className="max-w-7xl mx-auto space-y-8">
              
              {/* Results Header */}
              <div className="text-center mb-12">
                <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg animate-pulse">
                  <CheckCircle className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
                  MRI Analysis Complete
                </h2>
                <p className="text-xl text-gray-600 dark:text-gray-400 mb-6">
                  {enhancedAnalysis ? 'Comprehensive AI-powered medical analysis with clinical insights' : 'Basic MRI classification results'}
                </p>
                
                {/* Analysis Quality Indicators */}
                <div className="flex justify-center gap-4 mb-8">
                  <div className="flex items-center space-x-2 bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                    <CheckCircle2 className="w-4 h-4 text-green-600" />
                    <span className="text-sm font-medium">Analysis Complete</span>
                  </div>
                  {enhancedAnalysis && (
                    <div className="flex items-center space-x-2 bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                      <Sparkles className="w-4 h-4 text-purple-600" />
                      <span className="text-sm font-medium">AI Enhanced</span>
                    </div>
                  )}
                  <div className="flex items-center space-x-2 bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                    <Shield className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium">Medical Grade</span>
                  </div>
                </div>
              </div>

              {/* Comprehensive Results with Tabs */}
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-5 bg-white/50 backdrop-blur-sm border border-gray-200 dark:border-gray-700 rounded-xl p-1">
                  <TabsTrigger value="overview" className="flex items-center gap-2 data-[state=active]:bg-blue-500 data-[state=active]:text-white">
                    <Brain className="w-4 h-4" />
                    Overview
                  </TabsTrigger>
                  <TabsTrigger value="analysis" className="flex items-center gap-2 data-[state=active]:bg-green-500 data-[state=active]:text-white">
                    <Activity className="w-4 h-4" />
                    Analysis
                  </TabsTrigger>
                  <TabsTrigger value="medical" className="flex items-center gap-2 data-[state=active]:bg-purple-500 data-[state=active]:text-white">
                    <Stethoscope className="w-4 h-4" />
                    Medical
                  </TabsTrigger>
                  <TabsTrigger value="gemini" className="flex items-center gap-2 data-[state=active]:bg-indigo-500 data-[state=active]:text-white" disabled={!enhancedAnalysis?.gemini_enhanced_findings}>
                    <Sparkles className="w-4 h-4" />
                    Gemini AI
                  </TabsTrigger>
                  <TabsTrigger value="summary" className="flex items-center gap-2 data-[state=active]:bg-orange-500 data-[state=active]:text-white">
                    <FileText className="w-4 h-4" />
                    Summary
                  </TabsTrigger>
                </TabsList>

                {/* Overview Tab */}
                <TabsContent value="overview" className="mt-8">
                  <div className="grid lg:grid-cols-3 gap-8">
                    {/* Primary Classification */}
                    <Card className="lg:col-span-1 border-2 border-blue-200 dark:border-blue-800 shadow-xl bg-gradient-to-br from-white to-blue-50 dark:from-gray-800 dark:to-blue-950/50">
                      <CardHeader className="text-center pb-4">
                        <CardTitle className="flex items-center justify-center gap-2 text-xl">
                          <Brain className="w-6 h-6 text-blue-600" />
                          Classification
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="text-center space-y-6">
                        <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50 rounded-xl border border-blue-200 dark:border-blue-700">
                          <Badge className={`text-xl px-6 py-3 font-bold ${getClassificationColor(result?.predicted_class || '')}`}>
                            {(result?.predicted_class || '').toUpperCase()}
                          </Badge>
                          
                          <div className="mt-6">
                            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Confidence Level</p>
                            <div className={`text-4xl font-bold mb-3 ${getConfidenceColor(result?.confidence || 0)}`}>
                              {((result?.confidence || 0) * 100).toFixed(1)}%
                            </div>
                            <Progress 
                              value={(result?.confidence || 0) * 100} 
                              className="h-3 bg-gray-200 dark:bg-gray-700"
                            />
                          </div>
                        </div>

                        {/* Confidence Warnings */}
                        {(result?.confidence || 0) < 0.7 && (
                          <Alert className="border-red-200 bg-red-50 dark:bg-red-950/50">
                            <AlertTriangle className="h-4 w-4 text-red-600" />
                            <AlertDescription className="text-red-800 dark:text-red-200">
                              {(result?.confidence || 0) < 0.5 ? (
                                <>
                                  <strong>Very Low Confidence:</strong> This image may not be suitable for MRI analysis. 
                                  Please ensure you're uploading a proper MRI brain scan image.
                                </>
                              ) : (
                                <>
                                  <strong>Low Confidence:</strong> Results should be interpreted with caution. 
                                  Consider consulting with a medical professional.
                                </>
                              )}
                            </AlertDescription>
                          </Alert>
                        )}
                      </CardContent>
                    </Card>

                    {/* Probability Breakdown */}
                    <Card className="lg:col-span-2 shadow-xl border-2 border-gray-200 dark:border-gray-700">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-xl">
                          <Activity className="w-6 h-6 text-green-600" />
                          Detailed Probability Analysis
                        </CardTitle>
                        <CardDescription>
                          Breakdown of probabilities for each tumor type
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid md:grid-cols-2 gap-6">
                          {Object.entries(result?.probabilities || {})
                            .sort(([,a], [,b]) => b - a)
                            .map(([className, probability]) => (
                              <div key={className} className="space-y-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                                <div className="flex justify-between items-center">
                                  <span className="font-semibold capitalize text-lg">{className}</span>
                                  <span className="text-lg font-bold text-blue-600">
                                    {(probability * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <Progress 
                                  value={probability * 100} 
                                  className="h-3"
                                />
                                <p className="text-xs text-gray-600 dark:text-gray-400">
                                  {className === 'glioma' && 'Most common primary brain tumor'}
                                  {className === 'meningioma' && 'Usually benign, slow-growing tumor'}
                                  {className === 'pituitary' && 'Tumor in the pituitary gland'}
                                  {className === 'notumor' && 'Normal brain tissue detected'}
                                </p>
                              </div>
                            ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                {/* Analysis Tab */}
                <TabsContent value="analysis" className="mt-8">
                  {enhancedAnalysis ? (
                    <div className="grid lg:grid-cols-2 gap-8">
                      {/* Key Findings */}
                      <Card className="shadow-xl border-2 border-green-200 dark:border-green-800 bg-gradient-to-br from-green-50 to-teal-50 dark:from-green-950/50 dark:to-teal-950/50">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2 text-green-900 dark:text-green-100">
                            <CheckCircle className="w-6 h-6 text-green-600" />
                            Key Findings
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {enhancedAnalysis.key_findings?.map((finding, index) => (
                              <div key={index} className="flex items-start space-x-3 p-3 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                                <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                                <p className="text-green-800 dark:text-green-200 text-sm">{finding}</p>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>

                      {/* Disease Risks */}
                      <Card className="shadow-xl border-2 border-red-200 dark:border-red-800 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-950/50 dark:to-pink-950/50">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2 text-red-900 dark:text-red-100">
                            <AlertTriangle className="w-6 h-6 text-red-600" />
                            Risk Assessment
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            {Object.entries(enhancedAnalysis.disease_risks || {}).map(([disease, risk]) => (
                              <div key={disease} className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                                <div className="flex justify-between items-center mb-2">
                                  <span className="font-semibold capitalize text-red-900 dark:text-red-100">{disease}</span>
                                  <Badge className={`${
                                    risk.severity === 'High' ? 'bg-red-100 text-red-800' :
                                    risk.severity === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-green-100 text-green-800'
                                  }`}>
                                    {risk.severity}
                                  </Badge>
                                </div>
                                <p className="text-sm text-red-700 dark:text-red-300 mb-2">{risk.description}</p>
                                <Progress value={risk.probability * 100} className="h-2" />
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <Card className="shadow-xl border-2 border-gray-200 dark:border-gray-700">
                      <CardContent className="text-center py-12">
                        <Info className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-600 dark:text-gray-400">Enhanced analysis not available. Please use Enhanced or Gemini AI analysis mode for detailed insights.</p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Medical Tab */}
                <TabsContent value="medical" className="mt-8">
                  {enhancedAnalysis ? (
                    <div className="grid lg:grid-cols-2 gap-8">
                      {/* Medical Suggestions */}
                      <Card className="shadow-xl border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2 text-blue-900 dark:text-blue-100">
                            <Stethoscope className="w-6 h-6 text-blue-600" />
                            Medical Recommendations
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {enhancedAnalysis.medical_suggestions?.map((suggestion, index) => (
                              <div key={index} className="flex items-start space-x-3 p-3 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                                <Heart className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                                <p className="text-blue-800 dark:text-blue-200 text-sm">{suggestion}</p>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>

                      {/* Clinical Significance */}
                      <Card className="shadow-xl border-2 border-purple-200 dark:border-purple-800 bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-950/50 dark:to-indigo-950/50">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2 text-purple-900 dark:text-purple-100">
                            <Activity className="w-6 h-6 text-purple-600" />
                            Clinical Assessment
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                            <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">Clinical Significance</h4>
                            <p className="text-purple-800 dark:text-purple-200 text-sm">{enhancedAnalysis.clinical_significance}</p>
                          </div>
                          <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                            <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">Severity Assessment</h4>
                            <p className="text-purple-800 dark:text-purple-200 text-sm">{enhancedAnalysis.severity_assessment}</p>
                          </div>
                          <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                            <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">Follow-up Recommendations</h4>
                            <div className="space-y-2">
                              {enhancedAnalysis.follow_up_recommendations?.map((rec, index) => (
                                <div key={index} className="flex items-start space-x-2">
                                  <CheckCircle className="w-4 h-4 text-purple-600 mt-0.5 flex-shrink-0" />
                                  <p className="text-purple-700 dark:text-purple-300 text-sm">{rec}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <Card className="shadow-xl border-2 border-gray-200 dark:border-gray-700">
                      <CardContent className="text-center py-12">
                        <Stethoscope className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-600 dark:text-gray-400">Medical analysis not available. Please use Enhanced or Gemini AI analysis mode for medical insights.</p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Gemini AI Tab */}
                <TabsContent value="gemini" className="mt-8">
                  {enhancedAnalysis?.gemini_enhanced_findings ? (
                    <div className="space-y-8">
                      {/* Gemini Enhanced Findings */}
                      <Card className="shadow-xl border-2 border-indigo-200 dark:border-indigo-800 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-950/50 dark:to-purple-950/50">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2 text-indigo-900 dark:text-indigo-100">
                            <Sparkles className="w-6 h-6 text-indigo-600" />
                            Gemini AI Enhanced Analysis
                          </CardTitle>
                          <CardDescription className="text-indigo-700 dark:text-indigo-300">
                            Advanced AI insights powered by Google Gemini
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                          {/* Enhanced Summary */}
                          <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                            <h4 className="font-semibold text-indigo-900 dark:text-indigo-100 mb-2">AI Summary</h4>
                            <p className="text-indigo-800 dark:text-indigo-200 text-sm">{enhancedAnalysis.gemini_enhanced_summary}</p>
                          </div>

                          {/* Clinical Reasoning */}
                          <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                            <h4 className="font-semibold text-indigo-900 dark:text-indigo-100 mb-2">Clinical Reasoning</h4>
                            <p className="text-indigo-800 dark:text-indigo-200 text-sm">{enhancedAnalysis.gemini_clinical_reasoning}</p>
                          </div>

                          {/* Gemini Recommendations */}
                          <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                            <h4 className="font-semibold text-indigo-900 dark:text-indigo-100 mb-3">AI Recommendations</h4>
                            <div className="space-y-2">
                              {enhancedAnalysis.gemini_clinical_recommendations?.map((rec, index) => (
                                <div key={index} className="flex items-start space-x-2">
                                  <Star className="w-4 h-4 text-indigo-600 mt-0.5 flex-shrink-0" />
                                  <p className="text-indigo-700 dark:text-indigo-300 text-sm">{rec}</p>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Quality Metrics */}
                          <div className="grid md:grid-cols-3 gap-4">
                            <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg text-center">
                              <div className="text-2xl font-bold text-indigo-600 mb-1">
                                {((enhancedAnalysis.gemini_confidence_assessment || 0) * 100).toFixed(0)}%
                              </div>
                              <p className="text-xs text-indigo-700 dark:text-indigo-300">AI Confidence</p>
                            </div>
                            <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg text-center">
                              <div className="text-2xl font-bold text-indigo-600 mb-1">
                                {((enhancedAnalysis.gemini_report_quality_score || 0) * 100).toFixed(0)}%
                              </div>
                              <p className="text-xs text-indigo-700 dark:text-indigo-300">Report Quality</p>
                            </div>
                            <div className="p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg text-center">
                              <Badge className={`${
                                enhancedAnalysis.gemini_urgency_level === 'High' ? 'bg-red-100 text-red-800' :
                                enhancedAnalysis.gemini_urgency_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-green-100 text-green-800'
                              }`}>
                                {enhancedAnalysis.gemini_urgency_level}
                              </Badge>
                              <p className="text-xs text-indigo-700 dark:text-indigo-300 mt-1">Urgency Level</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <Card className="shadow-xl border-2 border-gray-200 dark:border-gray-700">
                      <CardContent className="text-center py-12">
                        <Sparkles className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-600 dark:text-gray-400">Gemini AI analysis not available. Please use Gemini AI analysis mode for advanced insights.</p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Summary Tab */}
                <TabsContent value="summary" className="mt-8">
                  <Card className="shadow-xl border-2 border-orange-200 dark:border-orange-800 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-950/50 dark:to-red-950/50">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-orange-900 dark:text-orange-100">
                        <FileText className="w-6 h-6 text-orange-600" />
                        Analysis Summary Report
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Executive Summary */}
                      <div className="p-6 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                        <h4 className="font-bold text-orange-900 dark:text-orange-100 mb-3 text-lg">Executive Summary</h4>
                        <p className="text-orange-800 dark:text-orange-200 mb-4">
                          {enhancedAnalysis?.report_summary || `MRI brain scan analysis completed. The AI model classified this scan as ${result?.predicted_class} with ${((result?.confidence || 0) * 100).toFixed(1)}% confidence.`}
                        </p>
                        
                        {/* Key Metrics */}
                        <div className="grid md:grid-cols-4 gap-4 mt-4">
                          <div className="text-center p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                            <div className="text-xl font-bold text-orange-600">{result?.predicted_class?.toUpperCase()}</div>
                            <p className="text-xs text-orange-700 dark:text-orange-300">Classification</p>
                          </div>
                          <div className="text-center p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                            <div className="text-xl font-bold text-orange-600">{((result?.confidence || 0) * 100).toFixed(1)}%</div>
                            <p className="text-xs text-orange-700 dark:text-orange-300">Confidence</p>
                          </div>
                          <div className="text-center p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                            <div className="text-xl font-bold text-orange-600">
                              {analysisMode === 'gemini' ? 'Gemini AI' : 
                               analysisMode === 'enhanced' ? 'Enhanced' : 'Basic'}
                            </div>
                            <p className="text-xs text-orange-700 dark:text-orange-300">Analysis Type</p>
                          </div>
                          <div className="text-center p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                            <div className="text-xl font-bold text-orange-600">{enhancedAnalysis?.gemini_enhanced_findings ? 'Yes' : 'No'}</div>
                            <p className="text-xs text-orange-700 dark:text-orange-300">Gemini AI</p>
                          </div>
                        </div>
                      </div>

                      {/* Important Disclaimers */}
                      <Alert className="border-amber-200 bg-amber-50 dark:bg-amber-950/50">
                        <AlertTriangle className="h-4 w-4 text-amber-600" />
                        <AlertDescription className="text-amber-800 dark:text-amber-200">
                          <strong>Medical Disclaimer:</strong> This AI analysis is for educational and research purposes only. 
                          It is not a substitute for professional medical diagnosis. Always consult qualified healthcare professionals 
                          for medical advice and treatment decisions.
                        </AlertDescription>
                      </Alert>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </section>
      )}

      <Footer />
    </div>
  );
};

export default MRIAnalysis;
