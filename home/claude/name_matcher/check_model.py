import os, sys, pickle
import pandas as pd
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("=" * 62)
print("  MODEL TRAINING AUDIT")
print("=" * 62)
print()

files = {
    "models/xgboost_model.joblib": "Trained model",
    "models/features.csv":         "Feature matrix (from YOUR data)",
    "models/cleaned_data.csv":     "Cleaned training data",
    "models/registry.pkl":         "Name registry",
}
missing = False
for fpath, label in files.items():
    if os.path.exists(fpath):
        dt   = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d  %H:%M:%S")
        size = os.path.getsize(fpath)
        print(f"  OK  {label:<40} {dt}  ({size:,} bytes)")
    else:
        print(f"  !! MISSING: {fpath}")
        missing = True

if missing:
    print()
    print("  Run:  python day1_explore_and_features.py")
    print("  Then: python day2_train_model.py")
    sys.exit(1)

print()
df = pd.read_csv("models/cleaned_data.csv")
print(f"Training rows: {len(df):,}")
label_col   = next((c for c in df.columns if "match" in c.lower()), None)
payer_col   = next((c for c in df.columns if "payer" in c.lower()), None)
account_col = next((c for c in df.columns if "account" in c.lower() or "holder" in c.lower()), None)
if label_col:
    m = int(df[label_col].sum())
    print(f"True matches:    {m} ({m/len(df):.1%})")
    print(f"True mismatches: {len(df)-m} ({(len(df)-m)/len(df):.1%})")
print()
print("Sample rows from YOUR training data:")
if payer_col and account_col:
    for _, row in df.sample(min(5,len(df)), random_state=7).iterrows():
        flag = "Match" if row.get(label_col,False) else "No Match"
        print(f"  {str(row[payer_col])[:30]:<32} vs {str(row[account_col])[:30]:<32} {flag}")
print()
if os.path.exists("models/class_info.pkl"):
    info = pickle.load(open("models/class_info.pkl","rb"))
    auc = info.get("baseline_auc","?")
    if isinstance(auc, float): print(f"Baseline AUC: {auc:.4f}")
print()
reg = pickle.load(open("models/registry.pkl","rb"))
print(f"Registry: {len(reg)} account holders, {sum(len(v) for v in reg.values())} aliases")
print()
print("=" * 62)
print("  If the sample rows above are from YOUR CSV, the model is")
print("  trained on your data.")
print()
print("  To retrain:")
print("  1. python day1_explore_and_features.py")
print("  2. python day2_train_model.py")
print("  3. streamlit run day3_demo_app.py")
print("=" * 62)
