import React from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Apple, ListChecks, Utensils, Goal, Download, Save, ShoppingCart, Trash2, Plus } from "lucide-react";

// Types and small client-only API helper
type Macro = { protein_g?: number; carbs_g?: number; fat_g?: number };
type AnalysisItem = { name: string; calories?: number; protein_g?: number; carbs_g?: number; fat_g?: number };
type AnalysisResponse = { error?: string; total_calories?: number; macros?: Macro; items?: AnalysisItem[]; suggestions?: string[]; swaps?: string[] };
type Meal = { name: string; description?: string; calories?: number; protein_g?: number; carbs_g?: number; fat_g?: number; ingredients?: string[] };
type PlanDay = { day?: string; meals?: Meal[] };
type MealPlanResponse = { error?: string; goal?: string; calories_target?: number; dietary_preference?: string; week?: PlanDay[]; days?: PlanDay[] };
type SavedPlan = { id: number; createdAt: string; data: MealPlanResponse; meta: { goal: string; pref?: string; calTarget?: number; days: number; mealsPerDay: number } };

function useNutritionApi() {
  const analyze = async (text: string) => {
    const res = await fetch("http://localhost:8000/nutrition/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("Nutrition analyze failed");
  return res.json() as Promise<AnalysisResponse>;
  };
  const plan = async (payload: { goal: string; calories_target?: number; dietary_preference?: string; days?: number; meals_per_day?: number }) => {
    const res = await fetch("http://localhost:8000/nutrition/meal-plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Meal plan failed");
  return res.json() as Promise<MealPlanResponse>;
  };
  return { analyze, plan };
}

const MealPlannerPage: React.FC = () => {
  const { analyze, plan } = useNutritionApi();

  // Analyze tab
  const [mealText, setMealText] = React.useState("");
  const [analysis, setAnalysis] = React.useState<AnalysisResponse | null>(null);
  const [loadingAnalyze, setLoadingAnalyze] = React.useState(false);

  // Plan tab
  const [goal, setGoal] = React.useState("weight_loss");
  const [pref, setPref] = React.useState<string | undefined>();
  const [calTarget, setCalTarget] = React.useState<number | undefined>();
  const [days, setDays] = React.useState(7);
  const [mealsPerDay, setMealsPerDay] = React.useState(3);
  const [planRes, setPlanRes] = React.useState<MealPlanResponse | null>(null);
  const [savedPlans, setSavedPlans] = React.useState<SavedPlan[]>(() => {
    try { return JSON.parse(localStorage.getItem("meal_plans") || "[]") as SavedPlan[]; } catch { return []; }
  });
  const [grocery, setGrocery] = React.useState<string[]>([]);
  const [loadingPlan, setLoadingPlan] = React.useState(false);
  const [checked, setChecked] = React.useState<Record<string, boolean>>({});
  const [newGrocery, setNewGrocery] = React.useState("");


  // Extras
  const [snacks, setSnacks] = React.useState(true);
  const [cookTime, setCookTime] = React.useState("any"); // any, quick, slow
  const [budget, setBudget] = React.useState("medium"); // low, medium, high

  const onAnalyze = async () => {
    setLoadingAnalyze(true);
    try {
      const data = await analyze(mealText);
      setAnalysis(data);
    } catch (e) {
      setAnalysis({ error: "Could not analyze meal" });
    } finally {
      setLoadingAnalyze(false);
    }
  };

  const onPlan = async () => {
    setLoadingPlan(true);
    try {
      const data = await plan({
        goal,
        calories_target: calTarget,
        dietary_preference: pref === "none" ? undefined : pref,
        days,
        meals_per_day: mealsPerDay,
      });
      setPlanRes(data);
      // Build grocery list from returned plan
  const items = new Set<string>();
      (data.week ?? data.days ?? []).forEach((d: PlanDay) => {
        (d.meals ?? []).forEach((m: Meal) => {
          (m.ingredients ?? []).forEach((ing: string) => items.add(ing));
        });
      });
  const list = Array.from(items);
  setGrocery(list);
  // reset checkboxes for any new list
  const nextChecked: Record<string, boolean> = {};
  list.forEach((i) => (nextChecked[i] = checked[i] ?? false));
  setChecked(nextChecked);
    } catch (e) {
      setPlanRes({ error: "Could not generate plan" });
    } finally {
      setLoadingPlan(false);
    }
  };

  const saveCurrentPlan = () => {
  if (!planRes || planRes.error) return;
  const entry: SavedPlan = { id: Date.now(), createdAt: new Date().toISOString(), data: planRes, meta: { goal, pref, calTarget, days, mealsPerDay } };
    const updated = [entry, ...savedPlans].slice(0, 10);
    setSavedPlans(updated);
    localStorage.setItem("meal_plans", JSON.stringify(updated));
  };

  const exportPlan = () => {
    if (!planRes || planRes.error) return;
    const blob = new Blob([JSON.stringify(planRes, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `meal-plan-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleGrocery = (item: string) => {
    setChecked((prev) => ({ ...prev, [item]: !prev[item] }));
  };
  const addGrocery = () => {
    const name = newGrocery.trim();
    if (!name) return;
    if (!grocery.includes(name)) {
      const next = [...grocery, name];
      setGrocery(next);
      setChecked((p) => ({ ...p, [name]: false }));
    }
    setNewGrocery("");
  };
  const clearChecked = () => {
    const remain = grocery.filter((g) => !checked[g]);
    const nextChecked: Record<string, boolean> = {};
    remain.forEach((r) => (nextChecked[r] = checked[r] ?? false));
    setGrocery(remain);
    setChecked(nextChecked);
  };

  const deleteSaved = (id: number) => {
    const updated = savedPlans.filter((p) => p.id !== id);
    setSavedPlans(updated);
    localStorage.setItem("meal_plans", JSON.stringify(updated));
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="container max-w-6xl mx-auto px-4 py-8 grid gap-6 animate-in fade-in slide-in-from-bottom-2">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight flex items-center gap-2">
              <Utensils className="w-6 h-6" /> Smart Meal & Nutrition Planner
            </h1>
            <p className="text-sm text-muted-foreground">
              AI-assisted meal analysis and creative weekly plans tailored to your goals.
            </p>
            <div className="h-1 w-24 bg-gradient-primary rounded-full mt-3" />
          </div>
        </div>

        <Tabs defaultValue="analyze" className="w-full">
          <TabsList className="grid grid-cols-4 max-w-4xl rounded-xl bg-muted/60 p-1">
            <TabsTrigger value="analyze" className="flex items-center gap-2"><Apple className="w-4 h-4" /> Analyze meal</TabsTrigger>
            <TabsTrigger value="plan" className="flex items-center gap-2"><ListChecks className="w-4 h-4" /> Build plan</TabsTrigger>
            <TabsTrigger value="grocery" className="flex items-center gap-2"><ShoppingCart className="w-4 h-4" /> Grocery</TabsTrigger>
            <TabsTrigger value="saved" className="flex items-center gap-2"><Save className="w-4 h-4" /> Saved</TabsTrigger>
          </TabsList>

          {/* Analyze */}
          <TabsContent value="analyze" className="mt-4">
            <Card className="shadow-card hover:shadow-medical">
              <CardHeader>
                <CardTitle>Quick Nutrition Analysis</CardTitle>
                <CardDescription>Paste or describe a meal; get macros and healthier swaps.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="mealText">What did you eat?</Label>
                  <Input id="mealText" placeholder="e.g., 2 eggs, toast with butter, orange juice" value={mealText} onChange={(e) => setMealText(e.target.value)} />
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button onClick={onAnalyze} disabled={!mealText || loadingAnalyze}>{loadingAnalyze ? "Analyzing..." : "Analyze"}</Button>
                </div>
                {analysis && (
                  <div className="grid gap-2">
                    {analysis.error ? (
                      <div className="text-sm text-red-600">{analysis.error}</div>
                    ) : (
                      <>
                        <div className="text-sm">Total Calories: <span className="font-semibold">{analysis.total_calories ?? "—"}</span></div>
                        <div className="text-xs text-muted-foreground">Macros: P {analysis.macros?.protein_g ?? "—"}g / C {analysis.macros?.carbs_g ?? "—"}g / F {analysis.macros?.fat_g ?? "—"}g</div>
                        {(analysis.items?.length ?? 0) > 0 && (
                          <div>
                            <div className="text-sm font-medium mt-2">Items</div>
                            <ul className="text-sm list-disc pl-5">
                              {analysis.items.map((it: AnalysisItem, i: number) => (
                                <li key={i}>{it.name} {it.calories ? `- ${it.calories} kcal` : ""}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {(analysis.swaps?.length ?? 0) > 0 && (
                          <div className="mt-2">
                            <div className="text-sm font-medium flex items-center gap-1"><Apple className="w-4 h-4" /> Healthy Swaps</div>
                            <ul className="text-sm list-disc pl-5">
                              {analysis.swaps.map((s: string, i: number) => <li key={i}>{s}</li>)}
                            </ul>
                          </div>
                        )}
                        {(analysis.suggestions?.length ?? 0) > 0 && (
                          <div className="mt-2">
                            <div className="text-sm font-medium">Suggestions</div>
                            <ul className="text-sm list-disc pl-5">
                              {analysis.suggestions.map((s: string, i: number) => <li key={i}>{s}</li>)}
                            </ul>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Plan */}
          <TabsContent value="plan" className="mt-4">
            <Card className="shadow-card hover:shadow-medical">
              <CardHeader>
                <CardTitle>Creative Weekly Plan</CardTitle>
                <CardDescription>Pick your goal and style; we’ll craft a plan you’ll actually want to follow.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="grid md:grid-cols-3 gap-3">
                  <div>
                    <Label>Goal</Label>
                    <Select value={goal} onValueChange={(v) => setGoal(v)}>
                      <SelectTrigger><SelectValue placeholder="Goal" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="weight_loss">Weight Loss</SelectItem>
                        <SelectItem value="muscle_gain">Muscle Gain</SelectItem>
                        <SelectItem value="diabetic_diet">Diabetic Diet</SelectItem>
                        <SelectItem value="balanced">Balanced</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Preference</Label>
                    <Select value={pref ?? 'none'} onValueChange={(v) => setPref(v === 'none' ? undefined : v)}>
                      <SelectTrigger><SelectValue placeholder="Select preference" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">None</SelectItem>
                        <SelectItem value="vegetarian">Vegetarian</SelectItem>
                        <SelectItem value="vegan">Vegan</SelectItem>
                        <SelectItem value="pescatarian">Pescatarian</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Calories (optional)</Label>
                    <Input type="number" value={calTarget ?? ''} onChange={(e) => setCalTarget(parseInt(e.target.value) || undefined)} />
                  </div>
                </div>
                <div className="grid md:grid-cols-4 gap-3">
                  <div>
                    <Label>Days</Label>
                    <Input type="number" value={days} onChange={(e) => setDays(Math.max(1, Math.min(14, parseInt(e.target.value) || 7)))} />
                  </div>
                  <div>
                    <Label>Meals / day</Label>
                    <Input type="number" value={mealsPerDay} onChange={(e) => setMealsPerDay(Math.max(1, Math.min(6, parseInt(e.target.value) || 3)))} />
                  </div>
                  <div>
                    <Label>Include snacks?</Label>
                    <Select value={snacks ? 'yes' : 'no'} onValueChange={(v) => setSnacks(v === 'yes')}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="yes">Yes</SelectItem>
                        <SelectItem value="no">No</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Cook time</Label>
                    <Select value={cookTime} onValueChange={(v) => setCookTime(v)}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="any">Any</SelectItem>
                        <SelectItem value="quick">Quick (&lt;20m)</SelectItem>
                        <SelectItem value="slow">Slow (batch/crock)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="grid md:grid-cols-3 gap-3">
                  <div>
                    <Label>Budget</Label>
                    <Select value={budget} onValueChange={(v) => setBudget(v)}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button onClick={onPlan} disabled={loadingPlan}>{loadingPlan ? "Crafting..." : "Generate Plan"}</Button>
                  <Button variant="secondary" onClick={saveCurrentPlan} disabled={!planRes || !!planRes?.error}><Save className="w-4 h-4 mr-1" /> Save</Button>
                  <Button variant="outline" onClick={exportPlan} disabled={!planRes || !!planRes?.error}><Download className="w-4 h-4 mr-1" /> Export JSON</Button>
                </div>
                {planRes && (
                  <div className="grid gap-3">
                    {planRes.error ? (
                      <div className="text-sm text-red-600">{planRes.error}</div>
                    ) : (
                      <div className="grid gap-3">
        {(planRes.week ?? planRes.days ?? []).map((d: PlanDay, i: number) => (
                          <div key={i} className="p-4 rounded-lg bg-muted/60 border border-border/60">
                            <div className="text-sm font-semibold mb-2 flex items-center gap-2"><Goal className="w-4 h-4" /> {d.day ?? `Day ${i+1}`}</div>
                            <ul className="text-sm list-disc pl-5">
          {d.meals?.map((m: Meal, j: number) => (
                                <li key={j}>
                                  <span className="font-medium">{m.name}</span>
                                  {m.calories ? ` - ${m.calories} kcal` : ''}
                                  {m.description ? ` — ${m.description}` : ''}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>


          {/* Grocery */}
          <TabsContent value="grocery" className="mt-4">
            <Card className="shadow-card hover:shadow-medical">
              <CardHeader>
                <CardTitle>Smart Grocery List</CardTitle>
                <CardDescription>Auto-collected from your current plan. Add or check off items.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3">
                <div className="flex flex-wrap gap-2">
                  <Input placeholder="Add item" value={newGrocery} onChange={(e) => setNewGrocery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addGrocery()} />
                  <Button onClick={addGrocery} variant="secondary"><Plus className="w-4 h-4 mr-1" /> Add</Button>
                  <Button onClick={clearChecked} variant="outline">Clear checked</Button>
                </div>
                <div className="grid gap-2">
                  {grocery.length === 0 && <div className="text-sm text-muted-foreground">Generate a plan to see ingredients here.</div>}
                  {grocery.map((g, i) => (
                    <label key={i} className="flex items-center gap-2 text-sm">
                      <input type="checkbox" checked={!!checked[g]} onChange={() => toggleGrocery(g)} />
                      <span className={checked[g] ? 'line-through text-muted-foreground' : ''}>{g}</span>
                    </label>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Saved Plans */}
          <TabsContent value="saved" className="mt-4">
            <Card className="shadow-card hover:shadow-medical">
              <CardHeader>
                <CardTitle>Saved Plans</CardTitle>
                <CardDescription>Your last 10 saved plans.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3">
                {savedPlans.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No saved plans yet. Generate and click Save.</div>
                ) : (
                  <div className="grid gap-3">
                    {savedPlans.map((p) => (
                      <div key={p.id} className="p-3 rounded-lg bg-muted/60 flex items-start justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium">{new Date(p.createdAt).toLocaleString()} — {p.meta.goal} {p.meta.pref ? `(${p.meta.pref})` : ""}</div>
                          <div className="text-xs text-muted-foreground">{p.meta.days} days • {p.meta.mealsPerDay} meals/day{p.meta.calTarget ? ` • ${p.meta.calTarget} kcal` : ""}</div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button variant="ghost" size="icon" aria-label="Delete" onClick={() => deleteSaved(p.id)}>
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      <Footer />
    </div>
  );
};

export default MealPlannerPage;
