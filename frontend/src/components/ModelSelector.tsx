import { useEffect, useState } from "react";
import { Cpu, Settings, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface ModelOption {
  id: string;
  name: string;
  accuracy: string;
  description: string;
  recommended?: boolean;
}

const models: ModelOption[] = [
  {
    id: "dermnet-resnet50",
    name: "DermNet ResNet50",
    accuracy: "92.5%",
    description: "Specialized for skin conditions with 23 classes",
    recommended: true,
  },
  {
    id: "efficientnet-b0",
    name: "EfficientNet B0",
    accuracy: "89.2%",
    description: "Balanced accuracy and speed",
  },
  {
    id: "resnet18",
    name: "ResNet-18",
    accuracy: "87.8%",
    description: "Robust feature extraction",
  },
];

const ModelSelector = ({
  selectedFile,
  onPredict,
  onLoadingChange,
}: {
  selectedFile: File | null;
  onPredict: (result: any) => void;
  onLoadingChange?: (loading: boolean) => void;
}) => {
  const [selectedModel, setSelectedModel] = useState("dermnet-resnet50");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onLoadingChange?.(isLoading);
  }, [isLoading, onLoadingChange]);

  const handlePredict = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    try {
      // Map UI model id to backend model name
      let backendModel = "efficientnet";
      if (selectedModel === "dermnet-resnet50") backendModel = "dermnet_resnet50";
      else if (selectedModel.includes("resnet")) backendModel = "resnet18";
      // Prepare form data
      const formData = new FormData();
      formData.append("file", selectedFile);
      // Call backend
      const response = await fetch(
        `http://localhost:8000/predict/${backendModel}`,
        {
          method: "POST",
          body: formData,
        }
      );
      if (!response.ok) {
        throw new Error((await response.json()).detail || "Prediction failed");
      }
      const data = await response.json();
      onPredict(data);
    } catch (err: any) {
      setError(err.message || "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  const selectedModelInfo = models.find((m) => m.id === selectedModel);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Settings className="w-5 h-5" />
          <span>Model Configuration</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Model Selection */}
        <div className="space-y-3">
          <label className="text-sm font-medium text-foreground">
            Select AI Model
          </label>
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger>
              <SelectValue placeholder="Choose a model" />
            </SelectTrigger>
            <SelectContent>
              {models.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <div className="flex items-center space-x-2">
                    <span>{model.name}</span>
                    {model.recommended && (
                      <Badge variant="secondary" className="text-xs">
                        Recommended
                      </Badge>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Selected Model Info */}
        {selectedModelInfo && (
          <div className="p-4 bg-muted rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-foreground">
                {selectedModelInfo.name}
              </h4>
              <Badge variant="outline">{selectedModelInfo.accuracy}</Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              {selectedModelInfo.description}
            </p>
          </div>
        )}

        {/* Predict Button */}
        <Button
          variant="medical"
          size="lg"
          className="w-full"
          onClick={handlePredict}
          disabled={!selectedFile || isLoading}
          aria-busy={isLoading}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Cpu className="w-5 h-5 mr-2" />
              Predict Diagnosis
            </>
          )}
        </Button>
        {error && (
          <p className="text-xs text-destructive text-center mt-2" role="alert">{error}</p>
        )}

        {!selectedFile && (
          <p className="text-xs text-muted-foreground text-center">
            Upload an image first to start prediction
          </p>
        )}
      </CardContent>
    </Card>
  );
};

export default ModelSelector;