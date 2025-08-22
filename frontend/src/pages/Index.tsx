import { useState } from "react";
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import ImageUploader from "@/components/ImageUploader";
import ModelSelector from "@/components/ModelSelector";
import PredictionResults from "@/components/PredictionResults";
import ModelInfo from "@/components/ModelInfo";
import Footer from "@/components/Footer";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [predictionResult, setPredictionResult] = useState<any | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  const handleImageSelect = (file: File) => {
    setSelectedFile(file);
    setPredictionResult(null);
  };

  const handlePredict = (result: any) => {
    setPredictionResult(result);
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <Header />
      <Hero />

      {/* Main Prediction Interface for Skin Analysis */}
      <section id="predictor" className="py-16 bg-gradient-subtle">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-3 tracking-tight">
              AI-Powered Skin Analysis
            </h2>
            <div className="h-1 w-24 bg-gradient-primary mx-auto rounded-full mb-4" />
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Upload a dermoscopic image to get instant analysis with visual
              explanations
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
            {/* Left Column - Upload & Controls */}
            <div className="space-y-6">
              <ImageUploader onImageSelect={handleImageSelect} />
              <ModelSelector
                selectedFile={selectedFile}
                onPredict={handlePredict}
                onLoadingChange={setIsPredicting}
              />
            </div>

            {/* Right Column - Results */}
            <div>
              <PredictionResults prediction={predictionResult} />
              {/* Skeleton loader overlay when predicting */}
              {isPredicting && (
                <div className="mt-6 p-4 rounded-lg border bg-muted/60 backdrop-blur animate-pulse">
                  <div className="h-5 w-40 bg-muted rounded mb-3" />
                  <div className="h-3 w-full bg-muted rounded mb-2" />
                  <div className="h-3 w-5/6 bg-muted rounded mb-2" />
                  <div className="h-3 w-2/3 bg-muted rounded" />
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <ModelInfo />
      <Footer />
    </div>
  );
};

export default Index;
