"""
train.py — Run this locally to regenerate the model from real data.

Usage:
    python train.py --foodkeeper FoodKeeper-Data.xls --expiry food_expiry_tracker.csv

Outputs (saved to ./models/):
    priority_model.pkl
    le_cat.pkl, le_donor.pkl, le_prep.pkl, le_season.pkl, le_target.pkl
    foodbridge_real_dataset.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

np.random.seed(42)

# ── CLI args ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--foodkeeper", default="FoodKeeper-Data.xls")
parser.add_argument("--expiry",     default="food_expiry_tracker.csv")
parser.add_argument("--out",        default="models")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── 1. Load FoodKeeper ───────────────────────────────────────────
print("📂 Loading FoodKeeper data...")
fk  = pd.read_excel(args.foodkeeper, sheet_name='Product',  engine='xlrd')
cat = pd.read_excel(args.foodkeeper, sheet_name='Category', engine='xlrd')

multiplier = {'Days': 24, 'Weeks': 168, 'Months': 720, 'Years': 8760}

def to_hours(val, metric):
    if pd.isna(val) or pd.isna(metric): return np.nan
    return val * multiplier.get(str(metric).strip(), np.nan)

fk['shelf_hours'] = fk.apply(
    lambda r: to_hours(r['DOP_Refrigerate_Max'], r['DOP_Refrigerate_Metric'])
           or to_hours(r['DOP_Pantry_Max'],      r['DOP_Pantry_Metric'])
           or to_hours(r['DOP_Freeze_Max'],       r['DOP_Freeze_Metric']),
    axis=1
)
fk = fk.merge(cat[['ID','Category_Name']], left_on='Category_ID', right_on='ID', how='left')

CAT_MAP = {
    'Produce':'Fresh Produce', 'Deli & Prepared Foods':'Cooked Meals',
    'Meat':'Cooked Meals', 'Poultry':'Cooked Meals', 'Seafood':'Fresh Produce',
    'Dairy Products & Eggs':'Dairy Products', 'Baked Goods':'Bakery Items',
    'Shelf Stable Foods':'Canned/Dry Goods',
    'Condiments, Sauces & Canned Goods':'Condiments',
    'Beverages':'Beverages', 'Grains, Beans & Pasta':'Grains & Pulses',
    'Food Purchased Frozen':'Frozen Foods', 'Baby Food':'Packaged Goods',
    'Vegetarian Proteins':'Packaged Goods',
}
fk['food_category'] = fk['Category_Name'].map(CAT_MAP).fillna('Packaged Goods')
fk_clean = fk[fk['shelf_hours'].notna()][['food_category','shelf_hours']].copy()
fk_clean['shelf_hours'] = fk_clean['shelf_hours'].clip(upper=2000)
print(f"   FoodKeeper usable rows: {len(fk_clean)}")

# ── 2. Build donation records from FoodKeeper ────────────────────
print("🔨 Building donation records from FoodKeeper shelf-life data...")
DONOR_TYPES  = ['restaurant','supermarket','household','event','corporate']
PREP_METHODS = ['cooked','raw','packaged','frozen','processed']
SEASONS      = ['summer','winter','monsoon','spring']
AUGMENT_PER_ROW = 80

rows = []
for _, fk_row in fk_clean.iterrows():
    max_shelf = fk_row['shelf_hours']
    cat_name  = fk_row['food_category']
    for _ in range(AUGMENT_PER_ROW):
        expiry_gap = np.random.uniform(0.5, min(max_shelf, 72))
        if cat_name in ['Cooked Meals', 'Fresh Produce', 'Dairy Products']:
            storage_temp = np.random.uniform(2, 8)
        elif cat_name == 'Frozen Foods':
            storage_temp = np.random.uniform(-20, -5)
        else:
            storage_temp = np.random.uniform(15, 25)

        rows.append({
            'expiry_gap_hours':    round(expiry_gap, 2),
            'food_category':       cat_name,
            'quantity_kg':         round(np.random.uniform(0.5, 50), 1),
            'storage_temp_c':      round(storage_temp, 1),
            'donor_type':          np.random.choice(DONOR_TYPES),
            'preparation_method':  np.random.choice(PREP_METHODS),
            'distance_to_ngo_km':  round(np.random.uniform(0.5, 30), 1),
            'donation_frequency':  int(np.random.randint(1, 20)),
            'time_of_day':         round(np.random.uniform(0, 23.9), 1),
            'ambient_humidity':    round(np.random.uniform(30, 90), 1),
            'packaging_integrity': round(np.random.uniform(0.3, 1.0), 2),
            'seasonal_indicator':  np.random.choice(SEASONS),
        })

df = pd.DataFrame(rows)

# ── 3. Fold in food_expiry_tracker.csv ───────────────────────────
print("📥 Folding in food_expiry_tracker.csv...")
et = pd.read_csv(args.expiry)

ITEM_COLS = ['item_beverage','item_dairy','item_fruit','item_grain',
             'item_meat','item_snack','item_vegetable']

def map_et_category(row):
    mapping = {
        'item_beverage':'Beverages', 'item_dairy':'Dairy Products',
        'item_fruit':'Fresh Produce', 'item_grain':'Grains & Pulses',
        'item_meat':'Cooked Meals', 'item_snack':'Snacks',
        'item_vegetable':'Fresh Produce'
    }
    for col in ITEM_COLS:
        if row.get(col) == True or row.get(col) == 1:
            return mapping[col]
    return 'Packaged Goods'

et['food_category']       = et.apply(map_et_category, axis=1)
et['expiry_gap_hours']    = (et['days_until_expiry'] * 24).clip(upper=72).round(2)
et['quantity_kg']         = (et['quantity'] * 50).round(1)
et['storage_temp_c']      = et.apply(
    lambda r: np.random.uniform(2, 8)   if r.get('storage_fridge')
    else (np.random.uniform(-20, -5)    if r.get('storage_freezer')
    else np.random.uniform(15, 25)), axis=1).round(1)
et['donor_type']          = np.random.choice(DONOR_TYPES, len(et))
et['preparation_method']  = np.random.choice(PREP_METHODS, len(et))
et['distance_to_ngo_km']  = np.random.uniform(0.5, 30, len(et)).round(1)
et['donation_frequency']  = np.random.randint(1, 20, len(et))
et['time_of_day']         = np.random.uniform(0, 23.9, len(et)).round(1)
et['ambient_humidity']    = np.random.uniform(30, 90, len(et)).round(1)
et['packaging_integrity'] = np.random.uniform(0.3, 1.0, len(et)).round(2)
et['seasonal_indicator']  = np.random.choice(SEASONS, len(et))

df = pd.concat([df, et[list(df.columns)]], ignore_index=True)
print(f"   Total records: {len(df)}")

# ── 4. Urgency labeling (transparent, rule-based) ─────────────────
def assign_urgency(row):
    h    = row['expiry_gap_hours']
    cat  = row['food_category']
    temp = row['storage_temp_c']

    # Primary rule: time to expiry
    if h < 2:    label = 'Critical'
    elif h < 6:  label = 'High'
    elif h < 24: label = 'Moderate'
    else:         label = 'Low'

    # Bump up perishable categories
    if cat in ['Cooked Meals', 'Fresh Produce', 'Dairy Products']:
        if label == 'Moderate':              label = 'High'
        if label == 'Low' and h < 36:        label = 'Moderate'

    # Bump up unsafe storage temperature
    if temp > 20 and cat in ['Cooked Meals', 'Dairy Products']:
        if label == 'High':     label = 'Critical'
        if label == 'Moderate': label = 'High'

    return label

df['urgency_class'] = df.apply(assign_urgency, axis=1)
print("\n📊 Urgency class distribution:")
print(df['urgency_class'].value_counts())

# ── 5. Encode features ───────────────────────────────────────────
le_cat    = LabelEncoder().fit(df['food_category'])
le_donor  = LabelEncoder().fit(df['donor_type'])
le_prep   = LabelEncoder().fit(df['preparation_method'])
le_season = LabelEncoder().fit(df['seasonal_indicator'])
le_target = LabelEncoder().fit(df['urgency_class'])

df['food_category_enc'] = le_cat.transform(df['food_category'])
df['donor_type_enc']    = le_donor.transform(df['donor_type'])
df['prep_method_enc']   = le_prep.transform(df['preparation_method'])
df['seasonal_enc']      = le_season.transform(df['seasonal_indicator'])
df['urgency_enc']       = le_target.transform(df['urgency_class'])

FEATURES = [
    'expiry_gap_hours', 'food_category_enc', 'quantity_kg', 'storage_temp_c',
    'donor_type_enc', 'prep_method_enc', 'distance_to_ngo_km', 'donation_frequency',
    'time_of_day', 'ambient_humidity', 'packaging_integrity', 'seasonal_enc'
]

X, y = df[FEATURES], df['urgency_enc']

# ── 6. Train / test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"\n🔀 After SMOTE — Train: {len(X_train_sm)}, Test: {len(X_test)}")

# ── 7. Train Random Forest ───────────────────────────────────────
print("\n🚀 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')

print(f"\n✅ Accuracy : {acc:.4f}")
print(f"✅ F1 (wtd) : {f1:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# 5-fold CV
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=2)
print(f"🔁 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 8. Save ──────────────────────────────────────────────────────
joblib.dump(rf,        os.path.join(args.out, 'priority_model.pkl'))
joblib.dump(le_cat,    os.path.join(args.out, 'le_cat.pkl'))
joblib.dump(le_donor,  os.path.join(args.out, 'le_donor.pkl'))
joblib.dump(le_prep,   os.path.join(args.out, 'le_prep.pkl'))
joblib.dump(le_season, os.path.join(args.out, 'le_season.pkl'))
joblib.dump(le_target, os.path.join(args.out, 'le_target.pkl'))
df.to_csv(os.path.join(args.out, 'foodbridge_real_dataset.csv'), index=False)

print(f"\n💾 Saved to ./{args.out}/")
print("   priority_model.pkl, le_*.pkl, foodbridge_real_dataset.csv")
