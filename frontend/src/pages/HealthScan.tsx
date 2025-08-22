import React, { useMemo, useRef, useState } from "react";
import { Upload, FileText, Image as ImageIcon, History, Loader2, Copy, Download, Sparkles, Wand2, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/use-toast";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

type ScanResult = {
  kind: "image" | "document";
  summary: string;
  detected: string[];
  risks: string[];
  precautions: string[];
  disclaimer: string;
  filename?: string;
  mime_type?: string;
  timestamp: string;
};

const STORAGE_KEY = "healthscan_history_v1";

const HealthScan: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"auto" | "gemini" | "fast">("auto");
  const [timeout, setTimeoutMs] = useState<number>(10);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { toast } = useToast();

  const history: ScanResult[] = useMemo(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw) as ScanResult[];
      return Array.isArray(arr) ? arr.slice(0, 5) : [];
    } catch {
      return [];
    }
  }, []);

  const saveHistory = (res: ScanResult) => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const arr = raw ? (JSON.parse(raw) as ScanResult[]) : [];
      arr.unshift(res);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(arr.slice(0, 5)));
    } catch {
      // ignore history persistence errors
    }
  };

  const onUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const url = new URL("http://localhost:8000/healthscan/analyze");
      url.searchParams.set("mode", mode);
      url.searchParams.set("timeout", String(timeout));
      const res = await fetch(url.toString(), {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      const data: ScanResult = await res.json();
      setResult(data);
      saveHistory(data);
      toast({ title: "Analysis complete", description: `Mode: ${mode.toUpperCase()}` });
    } catch (e) {
      setResult({
        kind: "document",
        summary: "Failed to analyze file. Please try again.",
        detected: [],
        risks: ["Network or server error"],
        precautions: ["Retry later"],
        disclaimer: "This is not a medical diagnosis. Please consult a doctor for confirmation.",
        filename: file?.name,
        mime_type: file?.type,
        timestamp: new Date().toISOString(),
      });
      toast({ variant: "destructive", title: "Analysis failed", description: "Please try a clearer scan or switch mode." });
    } finally {
      setLoading(false);
    }
  };

  const fileHint = file?.type?.startsWith("image/") ? "Image selected" : file ? "Document selected" : "";

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="container mx-auto px-4 py-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold flex items-center gap-2"><Sparkles className="w-5 h-5 text-primary"/> HealthScan AI</h2>
        <div className="hidden sm:flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => result && navigator.clipboard.writeText(result.summary)}>
            <Copy className="w-4 h-4 mr-1"/> Copy Summary
          </Button>
          {file && (
            <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
              <Upload className="w-4 h-4 mr-1"/> New Upload
            </Button>
          )}
        </div>
      </div>

      <Card className="border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" /> Upload Medical Image or Report
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
            <Input
              ref={inputRef}
              type="file"
              accept="image/*,application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <div className="flex items-center gap-2">
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as "auto" | "gemini" | "fast")}
                className="border rounded px-2 py-2 text-sm bg-background"
              >
                <option value="auto">Auto (Balanced)</option>
                <option value="gemini">Gemini (Best)</option>
                <option value="fast">Fast (No LLM)</option>
              </select>
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="w-4 h-4"/>
                <input
                  type="number"
                  min={6}
                  max={20}
                  value={timeout}
                  onChange={(e) => setTimeoutMs(Math.max(6, Math.min(20, Number(e.target.value) || 10)))}
                  className="w-16 border rounded px-2 py-1 bg-background"
                />
                s
              </div>
              <Button onClick={onUpload} disabled={!file || loading}>
              {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                Analyze
              </Button>
            </div>
            {fileHint && <span className="text-sm text-muted-foreground">{fileHint}</span>}
          </div>
          <p className="text-xs text-muted-foreground">Supported: PDF, JPG, PNG. Your file is processed securely.</p>
        </CardContent>
      </Card>

      {result && (
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {result.kind === "image" ? <ImageIcon className="w-5 h-5" /> : <FileText className="w-5 h-5" />}
              Scan Result
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {(result.keywords || []).slice(0, 10).map((k, i) => (
                <Badge key={i} variant="secondary" className="text-xs">{k}</Badge>
              ))}
            </div>

            <Tabs defaultValue="summary" className="w-full">
              <TabsList className="grid grid-cols-3 w-full">
                <TabsTrigger value="summary">Summary</TabsTrigger>
                <TabsTrigger value="findings">Findings</TabsTrigger>
                <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
              </TabsList>
              <TabsContent value="summary" className="space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="font-semibold">Summary</h3>
                  <Button variant="ghost" size="sm" onClick={() => navigator.clipboard.writeText(result.summary)}>
                    <Copy className="w-4 h-4 mr-1"/> Copy
                  </Button>
                </div>
                <p className="text-sm text-foreground/90 leading-relaxed">{result.summary}</p>
              </TabsContent>
              <TabsContent value="findings" className="space-y-3">
                <h3 className="font-semibold">Detected findings</h3>
                {result.detected?.length ? (
                  <ul className="list-disc pl-5 space-y-1 text-sm">
                    {result.detected.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No specific findings extracted.</p>
                )}
                {result.sections?.findings && (
                  <div className="mt-2 p-3 rounded bg-muted/40 text-sm whitespace-pre-wrap">
                    {result.sections.findings}
                  </div>
                )}
              </TabsContent>
              <TabsContent value="recommendations" className="space-y-3">
                <h3 className="font-semibold">Recommendations</h3>
                {result.precautions?.length ? (
                  <ul className="list-disc pl-5 space-y-1 text-sm">
                    {result.precautions.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No recommendations available.</p>
                )}
                {result.follow_up?.length ? (
                  <div className="mt-2 text-xs text-muted-foreground">{result.follow_up.join(" | ")}</div>
                ) : null}
              </TabsContent>
            </Tabs>

            <div className="flex justify-between items-center pt-3 border-t">
              <p className="text-xs text-muted-foreground">{result.disclaimer}</p>
              {/* Download removed per request */}
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="w-5 h-5" /> Recent Scans
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          {history.length === 0 && <p className="text-sm text-muted-foreground">No history yet.</p>}
          {history.map((h, i) => (
            <div key={i} className="rounded-lg border p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">{h.filename || "Upload"}</span>
                <span className="text-xs text-muted-foreground">{new Date(h.timestamp).toLocaleString()}</span>
              </div>
              <p className="text-sm line-clamp-3">{h.summary}</p>
              {h.detected?.length > 0 && (
                <p className="text-xs text-muted-foreground mt-2">Findings: {h.detected.slice(0,3).join(", ")}</p>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
      </main>
      <Footer />
    </div>
  );
};

export default HealthScan;
