# FoodBridge ML API — Deployment Guide

## Folder Structure

```
foodbridge-ml-api/
├── main.py                  ← FastAPI server
├── train.py                 ← Run locally to retrain the model
├── requirements.txt         ← Python dependencies
├── Procfile                 ← Railway startup command
├── models/                  ← Put your .pkl files here
│   ├── priority_model.pkl
│   ├── le_cat.pkl
│   ├── le_donor.pkl
│   ├── le_prep.pkl
│   ├── le_season.pkl
│   └── le_target.pkl
└── README.md
```

---

## Step A — Retrain the model locally (do this first)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt xlrd
   ```

2. Put these two files in the same folder as `train.py`:
   - `FoodKeeper-Data.xls`
   - `food_expiry_tracker.csv`

3. Run training:
   ```bash
   python train.py
   ```

4. This creates a `models/` folder with all `.pkl` files. ✅

---

## Step B — Test the API locally

```bash
uvicorn main:app --reload --port 8000
```

Open browser → `http://localhost:8000/docs`

You'll see interactive Swagger UI. Test the `/predict` endpoint there.

---

## Step C — Deploy to Railway.app

### 1. Create a Railway account
Go to → https://railway.app → Sign up with GitHub (free)

### 2. Create a new project
- Click **"New Project"**
- Choose **"Deploy from GitHub repo"**
- OR choose **"Empty project"** → then use Railway CLI

### 3. Upload via GitHub (recommended)

Create a new GitHub repo called `foodbridge-ml-api`:
```bash
git init
git add .
git commit -m "FoodBridge ML API v2"
git remote add origin https://github.com/YOUR_USERNAME/foodbridge-ml-api.git
git push -u origin main
```

Then in Railway → **"New Project" → "Deploy from GitHub"** → select `foodbridge-ml-api`

### 4. Set the start command
Railway auto-detects the `Procfile`. No extra config needed.

### 5. Get your public URL
After deployment, Railway gives you a URL like:
```
https://foodbridge-ml-api-production.up.railway.app
```

Test it:
```
https://your-url.up.railway.app/health
```

You should see: `{"status":"healthy","model":"RandomForest-12feat-v2"}`

---

## Step D — Connect to your Next.js website

Add this to your Vercel environment variables:
```
ML_SERVICE_URL=https://your-url.up.railway.app
```

The Next.js mlService.ts file (provided separately) will call:
```
POST https://your-url.up.railway.app/predict
```

---

## API Reference

### POST /predict

**Request body:**
```json
{
  "expiry_gap_hours": 3.5,
  "food_category": "Cooked Meals",
  "quantity_kg": 10.0,
  "storage_temp_c": 6.0,
  "donor_type": "restaurant",
  "preparation_method": "cooked",
  "distance_to_ngo_km": 2.5,
  "donation_frequency": 5,
  "time_of_day": 14.0,
  "ambient_humidity": 65.0,
  "packaging_integrity": 0.9,
  "seasonal_indicator": "summer"
}
```

**Response:**
```json
{
  "urgency_class": "High",
  "urgency_score": 70.0,
  "confidence": 0.94,
  "class_probs": {
    "Critical": 0.02,
    "High": 0.94,
    "Low": 0.01,
    "Moderate": 0.03
  },
  "feature_vector": [3.5, 2, 10.0, 6.0, 3, 0, 2.5, 5, 14.0, 65.0, 0.9, 3]
}
```

### GET /categories
Returns all valid values for categorical inputs.

### GET /health
Returns server status.
