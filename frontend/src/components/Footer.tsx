import { Brain } from "lucide-react";

const Footer = () => {
  return (
    <footer id="about" role="contentinfo" className="bg-primary text-primary-foreground mt-16">
      {/* Top subtle divider curve effect */}
      <div className="w-full h-6 bg-gradient-to-t from-primary/80 to-primary" aria-hidden="true" />

      <div className="container max-w-6xl mx-auto px-4 py-14">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded-lg bg-accent flex items-center justify-center shadow-glow">
                <Brain className="w-5 h-5 text-accent-foreground" />
              </div>
              <span className="text-xl font-bold tracking-tight">MedScanAI</span>
            </div>
            <p className="text-sm/6 text-primary-foreground/85">
              Advanced Medical AI Analysis
            </p>
            <p className="text-xs text-primary-foreground/75">
              Empowering healthcare professionals with AI‑driven insights for better patient outcomes.
            </p>
          </div>

          {/* AI-Powered */}
          <div className="space-y-2">
            <h3 className="font-semibold tracking-tight">AI‑Powered</h3>
            <p className="text-xs text-primary-foreground/80">
              Advanced machine learning algorithms for accurate analysis
            </p>
          </div>

          {/* Secure & Private */}
          <div className="space-y-2">
            <h3 className="font-semibold tracking-tight">Secure & Private</h3>
            <p className="text-xs text-primary-foreground/80">
              HIPAA compliant with end‑to‑end encryption
            </p>
          </div>

          {/* Medical Grade */}
          <div className="space-y-2">
            <h3 className="font-semibold tracking-tight">Medical Grade</h3>
            <p className="text-xs text-primary-foreground/80">
              Designed for healthcare professionals and researchers
            </p>
          </div>
        </div>

        <div className="border-t border-primary-foreground/20 mt-10 pt-6">
          <p className="text-center text-sm text-primary-foreground/70">
            © 2024 MedScanAI. Built with advanced AI for medical analysis and research.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;