/**
 * mlService.ts
 *
 * Replaces the existing /services/mlService.ts in your Next.js project.
 * Calls the FoodBridge Python ML API (deployed on Railway) instead of
 * using hardcoded math formulas.
 *
 * HOW TO USE:
 *   1. Deploy the Python API to Railway (see foodbridge-ml-api/README.md)
 *   2. Add ML_SERVICE_URL=https://your-railway-url.up.railway.app to Vercel env vars
 *   3. Replace your existing services/mlService.ts with this file
 */

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────

export type UrgencyClass = "Critical" | "High" | "Moderate" | "Low";

export interface MLPredictionInput {
  expiry_gap_hours:    number;
  food_category:       string;
  quantity_kg:         number;
  storage_temp_c:      number;
  donor_type?:         string;
  preparation_method?: string;
  distance_to_ngo_km?: number;
  donation_frequency?: number;
  time_of_day?:        number;
  ambient_humidity?:   number;
  packaging_integrity?:number;
  seasonal_indicator?: string;
}

export interface MLPredictionResult {
  urgency_class:  UrgencyClass;
  urgency_score:  number;   // 0–100
  confidence:     number;
  class_probs:    Record<UrgencyClass, number>;
}

// ── Map food type string → category name ─────────────────────────
function mapFoodCategory(foodType: string): string {
  const ft = foodType.toLowerCase();
  if (ft.includes("cooked") || ft.includes("meal") || ft.includes("prepared"))
    return "Cooked Meals";
  if (ft.includes("produce") || ft.includes("vegetable") || ft.includes("fruit"))
    return "Fresh Produce";
  if (ft.includes("dairy") || ft.includes("milk") || ft.includes("cheese") || ft.includes("egg"))
    return "Dairy Products";
  if (ft.includes("bakery") || ft.includes("bread") || ft.includes("cake"))
    return "Bakery Items";
  if (ft.includes("canned") || ft.includes("dry") || ft.includes("tin"))
    return "Canned/Dry Goods";
  if (ft.includes("beverage") || ft.includes("drink") || ft.includes("juice"))
    return "Beverages";
  if (ft.includes("frozen"))
    return "Frozen Foods";
  if (ft.includes("grain") || ft.includes("rice") || ft.includes("pulse") || ft.includes("lentil"))
    return "Grains & Pulses";
  if (ft.includes("snack"))
    return "Snacks";
  if (ft.includes("condiment") || ft.includes("sauce") || ft.includes("spice"))
    return "Condiments";
  if (ft.includes("salad"))
    return "Prepared Salads";
  return "Packaged Goods";
}

// ── Detect current season ────────────────────────────────────────
function getCurrentSeason(): string {
  const month = new Date().getMonth(); // 0–11
  if ([2, 3, 4].includes(month))  return "spring";
  if ([5, 6, 7].includes(month))  return "summer";
  if ([6, 7, 8].includes(month))  return "monsoon";  // India overlap
  if ([9, 10, 11].includes(month))return "winter";
  return "winter";
}

// ── Fallback: rule-based urgency (used if API is unreachable) ────
function fallbackUrgency(expiryHours: number, foodType: string): MLPredictionResult {
  const cat = mapFoodCategory(foodType);
  let urgency_class: UrgencyClass;

  if (expiryHours < 2)       urgency_class = "Critical";
  else if (expiryHours < 6)  urgency_class = "High";
  else if (expiryHours < 24) urgency_class = "Moderate";
  else                        urgency_class = "Low";

  // Perishable bump
  if (["Cooked Meals","Fresh Produce","Dairy Products"].includes(cat)) {
    if (urgency_class === "Moderate") urgency_class = "High";
    if (urgency_class === "Low" && expiryHours < 36) urgency_class = "Moderate";
  }

  const scoreMap: Record<UrgencyClass, number> = {
    Critical: 95, High: 70, Moderate: 40, Low: 15
  };

  return {
    urgency_class,
    urgency_score: scoreMap[urgency_class],
    confidence: 0.75,
    class_probs: { Critical: 0, High: 0, Moderate: 0, Low: 0, [urgency_class]: 1 } as Record<UrgencyClass, number>,
  };
}

// ── Main exported function ───────────────────────────────────────

/**
 * getUrgencyPrediction
 *
 * Calls the Railway ML API to classify donation urgency.
 * Falls back to rule-based logic if the API is unreachable.
 */
export async function getUrgencyPrediction(
  input: MLPredictionInput
): Promise<MLPredictionResult> {
  try {
    const payload = {
      expiry_gap_hours:    input.expiry_gap_hours,
      food_category:       mapFoodCategory(input.food_category),
      quantity_kg:         input.quantity_kg,
      storage_temp_c:      input.storage_temp_c ?? 6.0,
      donor_type:          input.donor_type          ?? "restaurant",
      preparation_method:  input.preparation_method  ?? "cooked",
      distance_to_ngo_km:  input.distance_to_ngo_km  ?? 5.0,
      donation_frequency:  input.donation_frequency  ?? 3,
      time_of_day:         input.time_of_day          ?? new Date().getHours(),
      ambient_humidity:    input.ambient_humidity     ?? 65.0,
      packaging_integrity: input.packaging_integrity  ?? 0.9,
      seasonal_indicator:  input.seasonal_indicator   ?? getCurrentSeason(),
    };

    const res = await fetch(`${ML_SERVICE_URL}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
      // 3 second timeout — fall back if API is slow
      signal:  AbortSignal.timeout(3000),
    });

    if (!res.ok) throw new Error(`ML API returned ${res.status}`);

    const data = await res.json();
    return {
      urgency_class: data.urgency_class as UrgencyClass,
      urgency_score: data.urgency_score,
      confidence:    data.confidence,
      class_probs:   data.class_probs,
    };

  } catch (err) {
    console.warn("[mlService] ML API unreachable, using fallback:", err);
    return fallbackUrgency(input.expiry_gap_hours, input.food_category);
  }
}

/**
 * getUrgencyScore (legacy compatibility)
 *
 * Keeps the original function signature used by recalculate-priority/route.ts
 * so existing code keeps working without changes.
 */
export async function getUrgencyScore(
  distance:        number,
  quantity:        number,
  expiryHours:     number,
  foodCategory:    number,  // legacy: 0=cooked, 1=raw, 2=packaged
  _biodegradability: number // legacy: unused now
): Promise<number> {
  const categoryMap: Record<number, string> = {
    0: "Cooked Meals",
    1: "Fresh Produce",
    2: "Packaged Goods",
  };

  const result = await getUrgencyPrediction({
    expiry_gap_hours:   expiryHours,
    food_category:      categoryMap[foodCategory] ?? "Packaged Goods",
    quantity_kg:        quantity,
    storage_temp_c:     6.0,
    distance_to_ngo_km: distance,
  });

  return result.urgency_score;
}
