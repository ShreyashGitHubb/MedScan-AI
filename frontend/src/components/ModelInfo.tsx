import { Database, TrendingUp, Shield, ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const ModelInfo = () => {
  return (
    <section id="model" className="py-16 bg-gradient-subtle">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-foreground mb-4">
            Model & Dataset Information
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Our AI models are trained on validated medical datasets and thoroughly tested for accuracy.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {/* Model Architecture */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="w-5 h-5 text-accent" />
                <span>Model Architecture</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Architecture</span>
                  <Badge variant="outline">EfficientNet B0</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Parameters</span>
                  <span className="text-sm text-muted-foreground">5.3M</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Input Size</span>
                  <span className="text-sm text-muted-foreground">224×224</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Pretrained</span>
                  <Badge variant="secondary">ImageNet</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Dataset Information */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Database className="w-5 h-5 text-accent" />
                <span>Training Dataset</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Dataset</span>
                  <Badge variant="outline">ISIC 2018</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Images</span>
                  <span className="text-sm text-muted-foreground">10,015</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Classes</span>
                  <span className="text-sm text-muted-foreground">7</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Validation</span>
                  <Badge variant="secondary">Dermatologist</Badge>
                </div>
              </div>
              <Button variant="outline" size="sm" className="w-full">
                <ExternalLink className="w-4 h-4 mr-2" />
                <a href="https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000">View Dataset</a>
              </Button>
            </CardContent>
          </Card>

          {/* Performance Metrics */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="w-5 h-5 text-accent" />
                <span>Performance Metrics</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Accuracy</span>
                  <Badge variant="default">89.2%</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">AUC</span>
                  <span className="text-sm text-muted-foreground">0.923</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Precision</span>
                  <span className="text-sm text-muted-foreground">0.887</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Recall</span>
                  <span className="text-sm text-muted-foreground">0.901</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Disclaimer */}
        <Card className="max-w-4xl mx-auto mt-12 border-destructive/20 bg-destructive/5">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-destructive">
              <Shield className="w-5 h-5" />
              <span>Important Disclaimer</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p id="disclaimer" className="text-sm text-foreground leading-relaxed">
              <strong>This is a research prototype and educational tool.</strong> The predictions made by this AI system 
              are not intended for medical diagnosis, treatment, or clinical decision-making. Always consult with 
              qualified healthcare professionals for medical advice. The model may produce false positives or false 
              negatives, and should never replace professional medical examination.
            </p>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default ModelInfo;