import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import XrayAnalysis from "./pages/XrayAnalysis";
import NotFound from "./pages/NotFound";
import MRIAnalysis from "./pages/MRIAnalysis";
import HealthHub from "./pages/HealthHub";
import HealthScan from "./pages/HealthScan";
import MealPlanner from "./pages/MealPlanner";



const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
      <Routes>
  <Route path="/" element={<Index />} />
  <Route path="/xray-analysis" element={<XrayAnalysis />} />
  <Route path="/mri-analysis" element={<MRIAnalysis />} /> {/* New route */}
  <Route path="/health-hub" element={<HealthHub />} />
  <Route path="/health-scan" element={<HealthScan />} />
  <Route path="/meal-planner" element={<MealPlanner />} />
  <Route path="*" element={<NotFound />} />
</Routes>


      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
