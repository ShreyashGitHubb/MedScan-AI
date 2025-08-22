import { Brain, Menu, Stethoscope, Activity, Cpu, Utensils, ShieldPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link, useLocation } from "react-router-dom";
import { Drawer, DrawerTrigger, DrawerContent, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";

const Header = () => {
  const location = useLocation();

  const NavLinks = () => (
    <>
      <Link
        to="/"
        className={`text-sm font-medium transition-colors ${
          location.pathname === "/" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        Skin Analysis
      </Link>

      <Link
        to="/xray-analysis"
        className={`text-sm font-medium transition-colors flex items-center ${
          location.pathname === "/xray-analysis" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        <Stethoscope className="w-4 h-4 mr-1" />
        X-ray Analysis
      </Link>

      <Link
        to="/mri-analysis"
        className={`text-sm font-medium transition-colors flex items-center ${
          location.pathname === "/mri-analysis" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        <Activity className="w-4 h-4 mr-1" />
        MRI Analysis
      </Link>

      <Link
        to="/health-hub"
        className={`text-sm font-medium transition-colors flex items-center ${
          location.pathname === "/health-hub" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        <Cpu className="w-4 h-4 mr-1" />
        Health Hub
      </Link>

      <Link
        to="/health-scan"
        className={`text-sm font-medium transition-colors flex items-center ${
          location.pathname === "/health-scan" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        <ShieldPlus className="w-4 h-4 mr-1" />
        HealthScan
      </Link>

      <Link
        to="/meal-planner"
        className={`text-sm font-medium transition-colors flex items-center ${
          location.pathname === "/meal-planner" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        <Utensils className="w-4 h-4 mr-1" />
        Meal Planner
      </Link>
    </>
  );

  return (
    <header className="sticky top-0 z-50 bg-background/70 backdrop-blur supports-[backdrop-filter]:bg-background/50 border-b border-border/60">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo & Title */}
        <Link to="/" className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-primary flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">MedScanAI</h1>
            <p className="text-xs text-muted-foreground">Intelligent Medical Diagnosis</p>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center space-x-6">
          <NavLinks />
        </nav>

        {/* Mobile Menu */}
        <div className="md:hidden">
          <Drawer>
            <DrawerTrigger asChild>
              <Button variant="ghost" size="icon" aria-label="Open menu">
                <Menu className="w-5 h-5" />
              </Button>
            </DrawerTrigger>
            <DrawerContent>
              <DrawerHeader>
                <DrawerTitle className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-md bg-gradient-primary flex items-center justify-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <span>MedScanAI</span>
                </DrawerTitle>
              </DrawerHeader>
              <nav className="px-4 pb-6 grid gap-4">
                <NavLinks />
              </nav>
            </DrawerContent>
          </Drawer>
        </div>
      </div>
    </header>
  );
};

export default Header;
