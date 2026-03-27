"""
FoodBridge ML API — FastAPI server
Serves urgency classification predictions for the FoodBridge platform.
Deploy on Railway.app (see README inside this folder).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ── App setup ────────────────────────────────────────────────────
app = FastAPI(
    title="FoodBridge ML API",
    description="Urgency classification for surplus food donations",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model + encoders once at startup ────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return joblib.load(os.path.join(BASE, "models", name))

rf        = load("priority_model.pkl")
le_cat    = load("le_cat.pkl")
le_donor  = load("le_donor.pkl")
le_prep   = load("le_prep.pkl")
le_season = load("le_season.pkl")
le_target = load("le_target.pkl")

# Known label sets (for safe encoding of unseen values)
KNOWN_CATS    = list(le_cat.classes_)
KNOWN_DONORS  = list(le_donor.classes_)
KNOWN_PREPS   = list(le_prep.classes_)
KNOWN_SEASONS = list(le_season.classes_)

def safe_encode(encoder, value, fallback_index=0):
    """Encode a value; fall back to index 0 if unseen."""
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return fallback_index


# ── Request / Response schemas ───────────────────────────────────
class DonationFeatures(BaseModel):
    expiry_gap_hours:    float = Field(..., ge=0,    description="Hours until expiry")
    food_category:       str   = Field(...,           description="e.g. 'Cooked Meals'")
    quantity_kg:         float = Field(..., ge=0,    description="Quantity in kg")
    storage_temp_c:      float = Field(...,           description="Storage temperature °C")
    donor_type:          str   = Field("restaurant", description="restaurant / supermarket / household / event / corporate")
    preparation_method:  str   = Field("cooked",     description="cooked / raw / packaged / frozen / processed")
    distance_to_ngo_km:  float = Field(5.0,  ge=0,  description="Distance to nearest NGO in km")
    donation_frequency:  int   = Field(5,    ge=1,  description="Donations per month by this donor")
    time_of_day:         float = Field(12.0, ge=0, le=24, description="Hour of day (0–24)")
    ambient_humidity:    float = Field(60.0,         description="Ambient humidity %")
    packaging_integrity: float = Field(0.9,  ge=0, le=1, description="0 = damaged, 1 = perfect")
    seasonal_indicator:  str   = Field("summer",     description="summer / winter / monsoon / spring")

class PredictionResponse(BaseModel):
    urgency_class:  str
    urgency_score:  float   # 0–100 numeric score derived from class
    confidence:     float   # model's max class probability
    class_probs:    dict    # probabilities for all 4 classes
    feature_vector: list    # encoded vector (for debugging)


# ── Urgency class → numeric score mapping ────────────────────────
URGENCY_SCORES = {"Critical": 95, "High": 70, "Moderate": 40, "Low": 15}


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "FoodBridge ML API v2.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "RandomForest-12feat-v2"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: DonationFeatures):
    try:
        # Encode categorical inputs
        cat_enc    = safe_encode(le_cat,    data.food_category)
        donor_enc  = safe_encode(le_donor,  data.donor_type)
        prep_enc   = safe_encode(le_prep,   data.preparation_method)
        season_enc = safe_encode(le_season, data.seasonal_indicator)

        feature_vector = [
            data.expiry_gap_hours,
            cat_enc,
            data.quantity_kg,
            data.storage_temp_c,
            donor_enc,
            prep_enc,
            data.distance_to_ngo_km,
            data.donation_frequency,
            data.time_of_day,
            data.ambient_humidity,
            data.packaging_integrity,
            season_enc,
        ]

        X = np.array(feature_vector).reshape(1, -1)
        pred_enc   = rf.predict(X)[0]
        proba      = rf.predict_proba(X)[0]

        urgency_class = le_target.inverse_transform([pred_enc])[0]
        confidence    = float(np.max(proba))
        urgency_score = float(URGENCY_SCORES.get(urgency_class, 50))

        class_probs = {
            le_target.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return PredictionResponse(
            urgency_class  = urgency_class,
            urgency_score  = urgency_score,
            confidence     = round(confidence, 4),
            class_probs    = class_probs,
            feature_vector = feature_vector,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
def get_categories():
    """Returns valid values for categorical inputs."""
    return {
        "food_categories":    KNOWN_CATS,
        "donor_types":        KNOWN_DONORS,
        "preparation_methods":KNOWN_PREPS,
        "seasonal_indicators":KNOWN_SEASONS,
    }
