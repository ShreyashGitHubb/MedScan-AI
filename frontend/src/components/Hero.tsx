import { useEffect, useState } from "react";
import { ArrowDown, Stethoscope, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import medicalHeroBg from "@/assets/medical-hero-bg.jpg";

const Hero = () => {
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const mql = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mql.matches) return;

    const onScroll = () => setOffset(window.scrollY * 0.15);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const scrollToPredictor = () => {
    document.getElementById("predictor")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative h-[620px] md:h-[700px] flex items-center justify-center overflow-hidden">
      {/* Parallax Background Image with clarity enhancements */}
      <div
        className="absolute inset-0 bg-cover bg-center will-change-transform"
        style={{
          backgroundImage: `url(${medicalHeroBg})`,
          transform: `translateY(${offset}px)`,
          transition: "transform 0.2s ease-out",
          filter: "contrast(1.1) saturate(1.05) brightness(0.9)",
        }}
        aria-hidden="true"
      />

      {/* Stronger dark scrim for text contrast */}
      <div
        className="absolute inset-0 bg-gradient-to-r from-black/70 via-black/50 to-black/20 md:from-black/80 md:via-black/55 md:to-black/25"
        aria-hidden="true"
      />

      {/* Decorative blurred orbs */}
      <div className="pointer-events-none absolute -top-10 -right-10 w-72 h-72 bg-accent/40 rounded-full blur-3xl" aria-hidden="true" />
      <div className="pointer-events-none absolute bottom-0 -left-10 w-80 h-80 bg-gradient-accent rounded-full blur-3xl opacity-40" aria-hidden="true" />

      {/* Content */}
      <div className="relative z-10 text-center text-white max-w-4xl mx-auto px-4">
        <h1 className="text-5xl md:text-6xl font-bold mb-6 animate-fade-in tracking-tight drop-shadow-[0_6px_24px_rgba(0,0,0,0.55)]">
          AI-Powered Medical
          <span className="text-accent block mt-2">Diagnosis Platform</span>
        </h1>

        <p className="text-xl md:text-2xl mb-8 text-white/95 animate-fade-in/75 drop-shadow-[0_4px_18px_rgba(0,0,0,0.5)]">
          Advanced skin lesion analysis, X-ray report classification, and MRI brain tumor detection with AI
        </p>

        {/* Prominent quick-action cards instead of flat white buttons */}
        <div className="flex flex-col sm:flex-row gap-5 justify-center animate-scale-in">
          {/* Skin Analysis CTA */}
          <div className="rounded-xl border border-white/15 bg-white/10 backdrop-blur-md shadow-[0_10px_30px_rgba(0,0,0,0.35)] px-1 py-1">
            <Button
              variant="medical"
              size="lg"
              onClick={scrollToPredictor}
              className="text-lg px-8 py-4 min-w-[220px]"
            >
              Skin Analysis
              <ArrowDown className="w-5 h-5 ml-2" />
            </Button>
          </div>

          {/* X-ray Analysis CTA */}
          <Link to="/xray-analysis" className="rounded-xl border border-white/15 bg-white/10 backdrop-blur-md shadow-[0_10px_30px_rgba(0,0,0,0.35)] px-1 py-1">
            <Button variant="accent" size="lg" className="text-lg px-8 py-4 min-w-[220px]">
              <Stethoscope className="w-5 h-5 mr-2" />
              X-ray Analysis
            </Button>
          </Link>

          {/* MRI Analysis CTA */}
          <Link to="/mri-analysis" className="rounded-xl border border-white/15 bg-white/10 backdrop-blur-md shadow-[0_10px_30px_rgba(0,0,0,0.35)] px-1 py-1">
            <Button variant="default" size="lg" className="text-lg px-8 py-4 min-w-[220px]">
              <Brain className="w-5 h-5 mr-2" />
              MRI Analysis
            </Button>
          </Link>
        </div>
      </div>

      {/* Bottom Gradient Fade */}
      <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-background/90 via-background/30 to-transparent" aria-hidden="true" />
    </section>
  );
};

export default Hero;