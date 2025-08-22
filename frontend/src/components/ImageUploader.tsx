import { useState, useRef } from "react";
import { Upload, Image, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const ImageUploader = ({ onImageSelect }: { onImageSelect: (file: File) => void }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      setSelectedFile(file);
      onImageSelect(file);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviewUrl(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground">Upload Skin Image</h3>
          
          {!selectedFile ? (
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${
                dragActive
                  ? "border-accent bg-accent/10"
                  : "border-border/70 bg-white/70 backdrop-blur-sm hover:border-accent hover:bg-accent/5 shadow-card"
              }`}
              role="button"
              tabIndex={0}
              aria-label="Upload image by dropping a file or clicking to browse"
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onButtonClick();
                }
              }}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={onButtonClick}
            >
              <Upload className="w-12 h-12 text-accent mx-auto mb-4" />
              <p className="text-lg font-medium text-foreground mb-2">
                Drop your image here, or click to browse
              </p>
              <p className="text-sm text-muted-foreground">
                Supports JPG, PNG, JPEG formats
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={previewUrl!}
                  alt="Preview"
                  className="w-full h-64 object-cover rounded-lg border shadow-card"
                />
                <Button
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2"
                  onClick={clearFile}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                <Image className="w-4 h-4" />
                <span>{selectedFile.name}</span>
                <span>({(selectedFile.size / 1024).toFixed(1)} KB)</span>
              </div>
            </div>
          )}
          
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept="image/*"
            onChange={handleChange}
          />
          <p className="text-xs text-destructive mt-2">
            <strong>Note:</strong> This tool is only for dermoscopic images of
            skin lesions. Uploading other images will result in invalid
            predictions.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ImageUploader;