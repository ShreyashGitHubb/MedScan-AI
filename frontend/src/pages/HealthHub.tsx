import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Activity, Droplets, Dumbbell, Salad, Moon, Scale, Ruler, Flame } from "lucide-react";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { RadialBarChart, RadialBar, PolarAngleAxis, Legend } from "recharts";

type Gender = "male" | "female";

function bmiCategory(bmi: number) {
  if (!isFinite(bmi) || bmi <= 0) return { label: "—", color: "text-muted-foreground", tip: "Enter your details to see insights." };
  if (bmi < 18.5) return { label: "Underweight", color: "text-blue-600", tip: "Add nutrient-dense foods and strength training." };
  if (bmi < 25) return { label: "Normal", color: "text-green-600", tip: "Great job! Maintain with balanced diet and activity." };
  if (bmi < 30) return { label: "Overweight", color: "text-amber-600", tip: "Aim for small calorie deficit and daily walks." };
  return { label: "Obese", color: "text-red-600", tip: "Consult a professional; start with gentle activity." };
}

function calcBMI(heightCm: number, weightKg: number) {
  if (!heightCm || !weightKg) return 0;
  const h = heightCm / 100;
  return +(weightKg / (h * h)).toFixed(1);
}

function calcBMR({ gender, weightKg, heightCm, age }: { gender: Gender; weightKg: number; heightCm: number; age: number; }) {
  // Mifflin-St Jeor
  const s = gender === "male" ? 5 : -161;
  return Math.round(10 * weightKg + 6.25 * heightCm - 5 * age + s);
}

const ACTIVITY_FACTORS: Record<string, number> = {
  sedentary: 1.2,
  light: 1.375,
  moderate: 1.55,
  active: 1.725,
  very_active: 1.9,
};

function calcWaterLiters(weightKg: number) {
  if (!weightKg) return 0;
  return +(weightKg * 0.033).toFixed(2); // 33 ml per kg
}

function calcBodyFat({ gender, heightCm, neckCm, waistCm, hipCm }: { gender: Gender; heightCm: number; neckCm?: number; waistCm?: number; hipCm?: number; }) {
  // US Navy Method (cm)
  const h = heightCm;
  if (!h || !waistCm || !neckCm) return undefined;
  const log10 = (n: number) => Math.log10(n);
  if (gender === "male") {
    if (waistCm - neckCm <= 0) return undefined;
    return +(86.01 * log10(waistCm - neckCm) - 70.041 * log10(h) + 36.76).toFixed(1);
  }
  if (!hipCm || waistCm + hipCm - neckCm <= 0) return undefined;
  return +(163.205 * log10(waistCm + hipCm - neckCm) - 97.684 * log10(h) - 78.387).toFixed(1);
}

function idealWeightRangeKg(heightCm: number) {
  if (!heightCm) return { min: 0, max: 0 };
  const h = heightCm / 100;
  return {
    min: +(18.5 * h * h).toFixed(1),
    max: +(24.9 * h * h).toFixed(1),
  };
}

const tipBank = {
  workout: [
    "Start with 20–30 min brisk walk daily.",
    "Add 2 days of strength training per week.",
    "Short on time? Try 10-min HIIT blocks.",
  ],
  diet: [
    "Fill half your plate with veggies.",
    "Prioritize protein each meal.",
    "Hydrate before snacking—thirst mimics hunger.",
  ],
  quote: [
    "Small steps lead to big change.",
    "Your future self will thank you.",
    "Consistency beats intensity.",
  ],
};

// Removed in favor of dedicated Meal Planner page

const HealthHub: React.FC = () => {
  const [gender, setGender] = React.useState<Gender>("male");
  const [age, setAge] = React.useState<number>(28);
  const [heightCm, setHeightCm] = React.useState<number>(175);
  const [weightKg, setWeightKg] = React.useState<number>(70);

  const [activity, setActivity] = React.useState<string>("light");
  const [waterDrank, setWaterDrank] = React.useState<number>(0); // liters today

  const [steps, setSteps] = React.useState<number>(4000);
  const [sleep, setSleep] = React.useState<number>(7.0);

  const [neckCm, setNeckCm] = React.useState<number | undefined>();
  const [waistCm, setWaistCm] = React.useState<number | undefined>();
  const [hipCm, setHipCm] = React.useState<number | undefined>();

  const bmi = calcBMI(heightCm, weightKg);
  const cat = bmiCategory(bmi);
  const bmr = calcBMR({ gender, weightKg, heightCm, age });
  const tdee = Math.round(bmr * (ACTIVITY_FACTORS[activity] ?? 1.375));
  const waterGoal = calcWaterLiters(weightKg);
  const waterPct = Math.min(100, Math.round((waterDrank / Math.max(waterGoal, 0.01)) * 100));
  const bf = calcBodyFat({ gender, heightCm, neckCm, waistCm, hipCm });
  const iw = idealWeightRangeKg(heightCm);

  // Generate a small calories trend around TDEE
  const caloriesData = React.useMemo(() => {
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    return days.map((d, i) => ({ day: d, calories: Math.round(tdee + (Math.sin(i) * 120)) }));
  }, [tdee]);

  const tips = React.useMemo(() => {
    const idx = new Date().getDay() % 3;
    return {
      workout: tipBank.workout[idx],
      diet: tipBank.diet[idx],
      quote: tipBank.quote[idx],
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="container mx-auto px-4 py-6 grid gap-6 animate-in fade-in slide-in-from-bottom-2">
        {/* Title */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Health Hub</h1>
            <p className="text-sm text-muted-foreground">Your daily snapshot with smart calculators and healthy nudges.</p>
          </div>
        </div>

        {/* Top: BMI Calculator */}
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Scale className="w-5 h-5" /> BMI Calculator</CardTitle>
            <CardDescription>Enter your details to see your BMI, category, and tailored tip.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-6 md:grid-cols-2">
            <div className="grid gap-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="height">Height (cm)</Label>
                  <Input id="height" type="number" inputMode="decimal" value={heightCm}
                    onChange={(e) => setHeightCm(parseFloat(e.target.value) || 0)} />
                </div>
                <div>
                  <Label htmlFor="weight">Weight (kg)</Label>
                  <Input id="weight" type="number" inputMode="decimal" value={weightKg}
                    onChange={(e) => setWeightKg(parseFloat(e.target.value) || 0)} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="age">Age</Label>
                  <Input id="age" type="number" value={age} onChange={(e) => setAge(parseInt(e.target.value) || 0)} />
                </div>
                <div>
                  <Label>Gender</Label>
                  <Select value={gender} onValueChange={(v: Gender) => setGender(v)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select gender" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <div className="flex flex-col justify-between">
              <div>
                <div className="flex items-baseline gap-3">
                  <div className="text-4xl font-bold tabular-nums">{isFinite(bmi) && bmi > 0 ? bmi.toFixed(1) : "—"}</div>
                  <div className={cn("text-sm px-2 py-1 rounded-full bg-muted", cat.color)}>{cat.label}</div>
                </div>
                <p className="text-sm text-muted-foreground mt-2">{cat.tip}</p>
                <div className="mt-4">
                  <Progress value={Math.min(100, Math.round((bmi / 40) * 100))} />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>0</span><span>20</span><span>40</span>
                  </div>
                </div>
              </div>
              <div className="text-xs text-muted-foreground mt-4">Note: BMI is a screening tool and doesn’t account for body composition.</div>
            </div>
          </CardContent>
        </Card>

        {/* Grid: Calculators and Widgets */}
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {/* Calorie Needs */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Flame className="w-5 h-5" /> Daily Calorie Needs</CardTitle>
              <CardDescription>BMR and TDEE using Mifflin-St Jeor.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <Label>Age</Label>
                  <Input type="number" value={age} onChange={(e) => setAge(parseInt(e.target.value) || 0)} />
                </div>
                <div>
                  <Label>Height (cm)</Label>
                  <Input type="number" value={heightCm} onChange={(e) => setHeightCm(parseFloat(e.target.value) || 0)} />
                </div>
                <div>
                  <Label>Weight (kg)</Label>
                  <Input type="number" value={weightKg} onChange={(e) => setWeightKg(parseFloat(e.target.value) || 0)} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Gender</Label>
                  <Select value={gender} onValueChange={(v: Gender) => setGender(v)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Activity</Label>
                  <Select value={activity} onValueChange={(v) => setActivity(v)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Activity level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sedentary">Sedentary</SelectItem>
                      <SelectItem value="light">Light (1-3 d/wk)</SelectItem>
                      <SelectItem value="moderate">Moderate (3-5 d/wk)</SelectItem>
                      <SelectItem value="active">Active (6-7 d/wk)</SelectItem>
                      <SelectItem value="very_active">Very Active</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-3 rounded-md bg-muted"><span className="text-muted-foreground">BMR</span> <div className="text-lg font-semibold">{bmr} kcal</div></div>
                <div className="p-3 rounded-md bg-muted"><span className="text-muted-foreground">TDEE</span> <div className="text-lg font-semibold">{tdee} kcal</div></div>
              </div>

              {/* Mini line chart */}
              <ChartContainer
                className="h-44"
                config={{ calories: { label: "Calories", color: "hsl(var(--primary))" } }}
              >
                <LineChart data={caloriesData} margin={{ left: 6, right: 6, top: 6, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" tickLine={false} axisLine={false} />
                  <YAxis width={30} tickLine={false} axisLine={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line type="monotone" dataKey="calories" stroke="var(--color-calories)" strokeWidth={2} dot={false} />
                </LineChart>
              </ChartContainer>
            </CardContent>
          </Card>

          {/* Water Intake */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Droplets className="w-5 h-5" /> Water Intake</CardTitle>
              <CardDescription>Suggested goal based on weight.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Weight (kg)</Label>
                  <Input type="number" value={weightKg} onChange={(e) => setWeightKg(parseFloat(e.target.value) || 0)} />
                </div>
                <div>
                  <Label>Drank today (L)</Label>
                  <Input type="number" inputMode="decimal" step="0.1" value={waterDrank}
                    onChange={(e) => setWaterDrank(parseFloat(e.target.value) || 0)} />
                </div>
              </div>
              <div className="text-sm text-muted-foreground">Goal: <span className="text-foreground font-medium">{waterGoal} L</span></div>
              <div className="flex items-center justify-center">
                <RadialBarChart
                  width={220}
                  height={180}
                  innerRadius={70}
                  outerRadius={90}
                  data={[{ name: "water", value: waterPct }]}
                  startAngle={90}
                  endAngle={-270}
                >
                  <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                  <RadialBar background dataKey="value" cornerRadius={8} fill="hsl(var(--primary))" />
                </RadialBarChart>
              </div>
              <div className="text-center text-sm">{waterPct}% of daily goal</div>
              <Button variant="secondary" onClick={() => setWaterDrank(Math.min(waterGoal, +(waterDrank + 0.25).toFixed(2)))}>+ 250 ml</Button>
            </CardContent>
          </Card>

          {/* Body Fat % */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Ruler className="w-5 h-5" /> Body Fat % (estimate)</CardTitle>
              <CardDescription>US Navy method; measurements in cm. Hip is for female.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Gender</Label>
                  <Select value={gender} onValueChange={(v: Gender) => setGender(v)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Height (cm)</Label>
                  <Input type="number" value={heightCm} onChange={(e) => setHeightCm(parseFloat(e.target.value) || 0)} />
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <Label>Neck (cm)</Label>
                  <Input type="number" value={neckCm ?? ''} onChange={(e) => setNeckCm(parseFloat(e.target.value) || undefined)} />
                </div>
                <div>
                  <Label>Waist (cm)</Label>
                  <Input type="number" value={waistCm ?? ''} onChange={(e) => setWaistCm(parseFloat(e.target.value) || undefined)} />
                </div>
                <div>
                  <Label>Hip (cm) {gender === 'female' ? '' : '(optional)'}</Label>
                  <Input type="number" value={hipCm ?? ''} onChange={(e) => setHipCm(parseFloat(e.target.value) || undefined)} />
                </div>
              </div>
              <div className="text-sm">Estimated Body Fat: <span className="font-semibold">{bf ?? '—'}%</span></div>
            </CardContent>
          </Card>

          {/* Ideal Weight */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5" /> Ideal Weight Range</CardTitle>
              <CardDescription>Based on BMI 18.5–24.9 for your height.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Height (cm)</Label>
                  <Input type="number" value={heightCm} onChange={(e) => setHeightCm(parseFloat(e.target.value) || 0)} />
                </div>
                <div>
                  <Label>Current Weight (kg)</Label>
                  <Input type="number" value={weightKg} onChange={(e) => setWeightKg(parseFloat(e.target.value) || 0)} />
                </div>
              </div>
              <div className="text-sm flex flex-wrap items-center gap-2">
                <span className="text-muted-foreground">Target:</span>
                <span className="font-semibold">{iw.min}–{iw.max} kg</span>
              </div>
              <div className="text-xs text-muted-foreground">Age can influence ideal ranges; consult a clinician for personalized targets.</div>
            </CardContent>
          </Card>

          {/* Steps Widget */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Dumbbell className="w-5 h-5" /> Steps</CardTitle>
              <CardDescription>Manual entry for now.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              <div className="flex items-center gap-2">
                <Input type="number" value={steps} onChange={(e) => setSteps(parseInt(e.target.value) || 0)} />
                <Button variant="secondary" onClick={() => setSteps(steps + 500)}>+500</Button>
              </div>
              <Progress value={Math.min(100, Math.round((steps / 10000) * 100))} />
              <div className="text-xs text-muted-foreground">Goal: 10,000 steps</div>
            </CardContent>
          </Card>

          {/* Smart Meal & Nutrition Planner moved to its own page */}
          <Card className="shadow-sm bg-gradient-to-br from-emerald-50 to-green-100/60 dark:from-emerald-900/20 dark:to-green-900/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">Smart Meal & Nutrition Planner</CardTitle>
              <CardDescription>Analyze meals and generate creative weekly plans with preferences and tips.</CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-between gap-4">
              <p className="text-sm text-muted-foreground">Open the dedicated planner to build your week with AI.</p>
              <Link to="/meal-planner">
                <Button>Open Meal Planner</Button>
              </Link>
            </CardContent>
          </Card>

          {/* HealthScan quick link */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">HealthScan</CardTitle>
              <CardDescription>Upload an image or report and get a quick risk summary.</CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-between gap-4">
              <p className="text-sm text-muted-foreground">Image and PDF supported.</p>
              <Link to="/health-scan">
                <Button variant="secondary">Open HealthScan</Button>
              </Link>
            </CardContent>
          </Card>

          {/* Sleep Widget */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Moon className="w-5 h-5" /> Sleep</CardTitle>
              <CardDescription>Hours slept last night.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              <div className="flex items-center gap-2">
                <Input type="number" inputMode="decimal" step="0.5" value={sleep} onChange={(e) => setSleep(parseFloat(e.target.value) || 0)} />
                <span className="text-sm text-muted-foreground">hours</span>
              </div>
              <Progress value={Math.min(100, Math.round((sleep / 8) * 100))} />
              <div className="text-xs text-muted-foreground">Target: 7–9 hours</div>
            </CardContent>
          </Card>
        </div>

        {/* Tips */}
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Salad className="w-5 h-5" /> Health Tips</CardTitle>
            <CardDescription>Quick nudges for workout, diet, and mindset.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-3">
            <div className="p-4 rounded-lg bg-muted/70">
              <div className="text-xs text-muted-foreground">Workout</div>
              <div className="mt-1 font-medium">{tips.workout}</div>
            </div>
            <div className="p-4 rounded-lg bg-muted/70">
              <div className="text-xs text-muted-foreground">Diet</div>
              <div className="mt-1 font-medium">{tips.diet}</div>
            </div>
            <div className="p-4 rounded-lg bg-muted/70">
              <div className="text-xs text-muted-foreground">Motivation</div>
              <div className="mt-1 font-medium">{tips.quote}</div>
            </div>
          </CardContent>
        </Card>
      </main>
      <Footer />
    </div>
  );
};

export default HealthHub;
