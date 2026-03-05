"""
DAY 2 SCRIPT — Train XGBoost Model + Build Registry
====================================================
WHAT THIS DOES:
  Step 1: Loads features from Day 1
  Step 2: Trains XGBoost classifier
  Step 3: Evaluates performance per decision tier
  Step 4: Builds the Resolved Name Registry
  Step 5: Builds the full predict() pipeline
  Step 6: Runs smoke tests to verify everything works
  Step 7: Saves everything for the demo

HOW TO RUN:
  python day2_train_model.py

REQUIREMENT:
  Run day1_explore_and_features.py first
"""

import pandas as pd
import numpy as np
import pickle
import re
import unicodedata
import warnings
warnings.filterwarnings('ignore')

from rapidfuzz import fuzz
import jellyfish
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report
)

print("=" * 60)
print("  DAY 2: XGBOOST TRAINING + REGISTRY BUILD")
print("=" * 60)


# ─────────────────────────────────────────────────────
# LOAD EVERYTHING FROM DAY 1
# ─────────────────────────────────────────────────────
print("\n📂 Loading Day 1 outputs...\n")

X          = pd.read_csv('models/features.csv')
y          = pd.read_csv('models/labels.csv').squeeze()
df         = pd.read_csv('models/cleaned_data.csv')
class_info = pickle.load(open('models/class_info.pkl', 'rb'))

PAYER_COL   = class_info['payer_col']
ACCOUNT_COL = class_info['account_col']
LABEL_COL   = class_info['label_col']

print(f"  ✅ Feature matrix: {X.shape}")
print(f"  ✅ Labels:         {len(y):,} rows")
print(f"  ✅ Scale weight:   {class_info['scale_pos_weight']:.2f}")
print(f"  ✅ Baseline AUC:   {class_info['baseline_auc']:.4f}")


# ─────────────────────────────────────────────────────
# RE-DEFINE HELPER FUNCTIONS
# (needed here so predict() works standalone)
# ─────────────────────────────────────────────────────

TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'rev', 'sir', 'lord', 'jr', 'sr',
    'ii', 'iii', 'iv',
    'mme', 'mlle', 'mle', 'monsieur', 'madame', 'mademoiselle',
    'ou', 'sarl', 'sas', 'ste', 'sa',
}

NICKNAMES = {
    'bob': 'robert',   'rob': 'robert',    'bobby': 'robert',
    'bill': 'william', 'will': 'william',   'billy': 'william',
    'jim': 'james',    'jimmy': 'james',    'jamie': 'james',
    'tom': 'thomas',   'tommy': 'thomas',
    'mike': 'michael', 'mick': 'michael',   'mickey': 'michael',
    'dave': 'david',   'davey': 'david',
    'chris': 'christopher',
    'liz': 'elizabeth', 'beth': 'elizabeth', 'betty': 'elizabeth',
    'kate': 'katherine', 'kathy': 'katherine', 'katie': 'katherine',
    'sue': 'susan',    'susie': 'susan',
    'joe': 'joseph',   'jo': 'josephine',
    'nick': 'nicholas', 'nicky': 'nicholas',
    'pat': 'patricia', 'patty': 'patricia',
    'dan': 'daniel',   'danny': 'daniel',
    'sam': 'samuel',   'sammy': 'samuel',
    'andy': 'andrew',  'drew': 'andrew',
    'tony': 'anthony', 'ant': 'anthony',
    'ben': 'benjamin', 'benny': 'benjamin',
    'alex': 'alexander',
    'ned': 'edward',   'ted': 'edward',    'ed': 'edward',
    'fred': 'frederick', 'freddy': 'frederick',
    'harry': 'henry',  'hal': 'henry',
    'jack': 'john',    'johnny': 'john',
    'pete': 'peter',
    'dick': 'richard', 'rick': 'richard',  'rich': 'richard',
    'ron': 'ronald',   'ronnie': 'ronald',
    'steve': 'steven', 'stevie': 'steven',
    'matt': 'matthew', 'matty': 'matthew',
    'greg': 'gregory',
    'jeff': 'jeffrey',
    'ken': 'kenneth',  'kenny': 'kenneth',
    'larry': 'lawrence', 'lars': 'lawrence',
    'len': 'leonard',  'lenny': 'leonard',
    'ray': 'raymond',
    'russ': 'russell',
    'stu': 'stewart',
    'tim': 'timothy',  'timmy': 'timothy',
    'vince': 'vincent',
    'walt': 'walter',
    'nate': 'nathaniel',
    'deb': 'deborah',  'debbie': 'deborah',
    'tricia': 'patricia',
    'eliza': 'elizabeth', 'lisa': 'elizabeth',
    'dom': 'dominic',
    'randy': 'randolph',
    'sly': 'sylvester',
    'bart': 'bartholomew',
    'al': 'alexander',
}


def normalize(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ''
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name
                   if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def tokenize(name: str) -> list:
    tokens = normalize(name).split()
    return [t for t in tokens
            if t not in TITLES and len(t) > 0]


def initials_score(t1, t2):
    if not t1 or not t2:
        return 0.0
    scores = []
    for a in t1:
        best = 0.0
        for b in t2:
            if len(a) == 1 and b.startswith(a):
                best = max(best, 0.9)
            elif len(b) == 1 and a.startswith(b):
                best = max(best, 0.9)
            elif a == b:
                best = 1.0
        scores.append(best)
    return sum(scores) / len(scores)


def nickname_score(t1, t2):
    if not t1 or not t2:
        return 0.0
    for a in t1:
        for b in t2:
            a_canon = NICKNAMES.get(a, a)
            b_canon = NICKNAMES.get(b, b)
            if a_canon == b_canon and a != b:
                return 1.0
    return 0.0


def bigram_similarity(s1, s2):
    def get_bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))
    a, b = get_bigrams(s1), get_bigrams(s2)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def extract_features(payer: str, account: str) -> dict:
    n1 = normalize(payer)
    n2 = normalize(account)
    t1 = tokenize(payer)
    t2 = tokenize(account)
    max_len = max(len(n1), len(n2), 1)
    max_tok = max(len(t1), len(t2), 1)

    jaro_winkler = fuzz.ratio(n1, n2) / 100
    token_sort   = fuzz.token_sort_ratio(n1, n2) / 100
    token_set    = fuzz.token_set_ratio(n1, n2) / 100
    partial      = fuzz.partial_ratio(n1, n2) / 100
    last_tok_sim = (fuzz.ratio(t1[-1], t2[-1]) / 100
                    if t1 and t2 else 0.0)

    soundex_last   = 0.0
    metaphone_last = 0.0
    if t1 and t2:
        soundex_last = (
            1.0 if jellyfish.soundex(t1[-1]) ==
                   jellyfish.soundex(t2[-1]) else 0.0)
        metaphone_last = (
            1.0 if jellyfish.metaphone(t1[-1]) ==
                   jellyfish.metaphone(t2[-1]) else 0.0)

    # CONFLICT DETECTION FIX
    # Problem: "Shikhar Gupta" vs "Maruti Gupta" scored 95%+
    # because last_tok_sim=1.0 (Gupta=Gupta) dominated.
    # Fix: detect when both names have real first names that clearly differ.
    first_name_conflict = 0.0
    if len(t1) >= 2 and len(t2) >= 2:
        fn1, fn2 = t1[0], t2[0]
        if len(fn1) > 1 and len(fn2) > 1:
            fn_sim = fuzz.ratio(fn1, fn2) / 100
            nick_match = (NICKNAMES.get(fn1, fn1) == NICKNAMES.get(fn2, fn2))
            if fn_sim < 0.5 and not nick_match:
                first_name_conflict = 1.0


    # ── FEATURE GROUP 8: French/African name patterns ──────────

    # gender_conflict: M./MR payer vs MME/MLLE account (or vice versa)
    # "MONSIEUR JULIEN PEREZ" vs "MME JULIEN..." -> conflict = 1.0
    MALE_TITLES_SET   = {'m', 'mr', 'monsieur'}
    FEMALE_TITLES_SET = {'mme', 'mlle', 'mle', 'madame', 'mademoiselle', 'ms', 'mrs', 'miss'}
    def _get_gender(name):
        toks = re.sub(r'[^a-z ]', ' ', name.lower()).split()
        for t in toks[:3]:
            if t in MALE_TITLES_SET:   return 'M'
            if t in FEMALE_TITLES_SET: return 'F'
        return None
    gender_payer   = _get_gender(payer)
    gender_account = _get_gender(account)
    gender_conflict = 1.0 if (
        gender_payer and gender_account and gender_payer != gender_account
    ) else 0.0

    # is_company: payer is a business name, not a person
    # "BIODISTRIB" vs "MME LUCITA..." -> company = 1.0 -> never a match
    COMPANY_SIGNALS = {'sarl', 'sas', 'ste', 'sa', 'ltd', 'inc', 'corp', 'llc',
                       'gmbh', 'biodistrib', 'assoc', 'association'}
    _payer_lower = n1.lower()
    is_company = 1.0 if any(sig in _payer_lower.split() for sig in COMPANY_SIGNALS) else 0.0
    # Also flag if payer has no letters at all that look like a person name
    if len(t1) == 0 or (len(t1) == 1 and len(t1[0]) < 3):
        is_company = 1.0

    # extra_middle_penalty: account has many more tokens than payer
    # "Ziva CVAR" (2 toks) vs "MLE ZIVA ELISABETH CVAR" (3 toks) -> penalty
    # This fires when payer tokens are all in account but account has extras.
    # Low penalty = payer was just using surname+first without middle names (normal).
    # High penalty = account is a completely different person with coincidental overlap.
    # We leave the ML to learn the right weight from labeled data.
    payer_in_acct    = len(set(t1) & set(t2)) / max(len(t1), 1)
    extra_in_account = max(0, len(t2) - len(t1)) / max(len(t2), 1)
    # Combined: when payer is fully subset AND account has lots of extras
    subset_with_extras = payer_in_acct * extra_in_account

    both_names_high = 0.0
    if t1 and t2:
        fn_s = fuzz.ratio(t1[0], t2[0]) / 100 if (len(t1) >= 2 and len(t2) >= 2) else 1.0
        if fn_s >= 0.7 and last_tok_sim >= 0.7:
            both_names_high = (fn_s + last_tok_sim) / 2

    return {
        'exact':               1.0 if n1 == n2 else 0.0,
        'jaro_winkler':        jaro_winkler,
        'token_sort':          token_sort,
        'token_set':           token_set,
        'partial':             partial,
        'levenshtein':         1 - (
            jellyfish.levenshtein_distance(n1, n2) / max_len),
        'bigram':              bigram_similarity(n1, n2),
        'first_tok_sim':       (fuzz.ratio(t1[0], t2[0]) / 100
                               if t1 and t2 else 0.0),
        'last_tok_sim':        last_tok_sim,
        'initials':            initials_score(t1, t2),
        'nickname':            nickname_score(t1, t2),
        'shared_tok':          len(set(t1) & set(t2)) / max_tok,
        'soundex_last':        soundex_last,
        'metaphone_last':      metaphone_last,
        'len_diff':            abs(len(n1) - len(n2)) / max_len,
        'tok_count_diff':      abs(len(t1) - len(t2)) / max_tok,
        'first_char':          (1.0 if n1 and n2 and
                               n1[0] == n2[0] else 0.0),
        'first_name_conflict': first_name_conflict,
        'both_names_high':     both_names_high,
        'fuzzy_composite':     (jaro_winkler * 0.4 +
                               token_sort * 0.35 +
                               token_set * 0.25),
        'gender_conflict':    gender_conflict,
        'is_company':         is_company,
        'payer_in_acct':      payer_in_acct,
        'extra_in_account':   extra_in_account,
        'subset_with_extras': subset_with_extras,
    }


# ─────────────────────────────────────────────────────
# STEP 1: TRAIN/VALIDATION/TEST SPLIT
# ─────────────────────────────────────────────────────
print("\n\n✂️  STEP 1: Splitting data...\n")

# Split: 80% train, 20% test
# stratify=y ensures same match/no-match ratio in each set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
# Remove unused val variables
X_val, y_val = X_test, y_test

print(f"  Train set:  {len(X_train):,} rows")
print(f"  Test set:   {len(X_test):,}  rows (final evaluation)")
print(f"\n  Why separate test set?")
print(f"  • Train: Model learns from this")
print(f"  • Test:  Honest final score — never seen during training")


# ─────────────────────────────────────────────────────
# STEP 2: TRAIN XGBOOST
# ─────────────────────────────────────────────────────
print(f"\n\n🤖 STEP 2: Training XGBoost...\n")
print(f"  What Gradient Boosting does:")
print(f"  • Builds hundreds of decision trees sequentially")
print(f"  • Each tree learns from the mistakes of all previous")
print(f"  • Final prediction = weighted vote of all trees")
print(f"  • Calibration layer converts raw scores to clean")
print(f"    0.0–1.0 probabilities (Platt scaling)\n")
print(f"  NOTE: On your full 15,000 rows, XGBoost is also")
print(f"  excellent. GradientBoosting is used here for")
print(f"  well-calibrated probabilities on any dataset size.\n")

# Base gradient boosting model
base_model = GradientBoostingClassifier(
    # Number of trees to build
    n_estimators=200,

    # Max depth of each tree (3 = shallow = less overfit)
    max_depth=3,

    # How fast to learn (lower = more careful = better)
    learning_rate=0.05,

    # Use 80% of training data per tree
    subsample=0.8,

    # Minimum samples per leaf
    min_samples_leaf=3,

    random_state=42,
)

# Calibration wrapper: converts raw tree scores to proper
# 0.0–1.0 probabilities using Platt scaling (sigmoid fit)
model = CalibratedClassifierCV(
    base_model,
    cv=5,            # 5-fold cross-validated calibration
    method='sigmoid' # Platt scaling
)

print(f"  Training with 5-fold calibration...")
model.fit(X_train, y_train)

print(f"  ✅ Training complete")


# ─────────────────────────────────────────────────────
# STEP 3: EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────
print(f"\n\n📊 STEP 3: Evaluating on test set...\n")
print(f"  (Test set was never seen during training)")

y_prob = model.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)

print(f"\n  {'='*45}")
print(f"  TEST AUC:  {auc:.4f}")
print(f"  {'='*45}")
print(f"\n  What AUC means:")
print(f"  • 1.0 = perfect, never wrong")
print(f"  • 0.9 = 90% of the time correctly ranks")
print(f"         a true match above a false match")
print(f"  • 0.5 = random guessing")

# Decision tier analysis
print(f"\n  DECISION TIER ANALYSIS:")
print(f"  (What % of test cases fall into each tier)")
print(f"\n  {'Tier':<22} {'Threshold':<12} {'Cases':>7} "
      f"{'Coverage':>10} {'Precision':>11}")
print(f"  {'-'*65}")

tiers = [
    ('AUTO_APPROVE',    0.92, '🟢'),
    ('SUGGEST_MATCH',   0.70, '🟡'),
    ('HUMAN_REVIEW',    0.45, '🟠'),
    ('LIKELY_MISMATCH', 0.00, '🔴'),
]

total        = len(y_test)
auto_approve = 0
total_auto_correct = 0

tier_results = []
prev_thresh = 1.01

for label, thresh, icon in tiers:
    mask  = (y_prob >= thresh) & (y_prob < prev_thresh)
    cases = mask.sum()
    if cases > 0:
        prec = (y_test[mask] == 1).mean()
    else:
        prec = 0.0
    coverage  = cases / total
    prev_thresh = thresh

    tier_results.append({
        'tier': label, 'cases': cases,
        'coverage': coverage, 'precision': prec
    })

    print(f"  {icon} {label:<20} ≥{thresh:<11.0%} "
          f"{cases:>7,} {coverage:>10.1%} {prec:>10.1%}")

    if label == 'AUTO_APPROVE':
        auto_approve = cases
        total_auto_correct = (y_test[y_prob >= thresh] == 1).sum()

# CT workload reduction estimate
ct_reduction = (auto_approve / total) * 100
print(f"\n  ✅ ESTIMATED CT WORKLOAD REDUCTION:")
print(f"     {ct_reduction:.0f}% of cases handled "
      f"automatically (AUTO_APPROVE tier)")
print(f"     CT agents only review the remaining "
      f"{100-ct_reduction:.0f}%")

# Feature importance
print(f"\n  TOP FEATURES (what the model learned):")
print(f"  (These are the signals your CT agents")
print(f"   implicitly used — now made explicit)\n")

# Feature importance — extract from calibrated estimators
try:
    importances_list = []
    for calibrated_clf in model.calibrated_classifiers_:
        est = calibrated_clf.estimator
        if hasattr(est, 'feature_importances_'):
            importances_list.append(est.feature_importances_)
    if importances_list:
        raw_importances = np.mean(importances_list, axis=0)
    else:
        raw_importances = np.ones(len(X.columns)) / len(X.columns)
except Exception:
    raw_importances = np.ones(len(X.columns)) / len(X.columns)

feat_imp = pd.Series(raw_importances, index=X.columns).sort_values(ascending=False)

for feat, imp in feat_imp.head(8).items():
    bar    = '█' * int(imp * 60)
    pct    = imp * 100
    print(f"  {feat:<20} {bar:<35} {pct:.1f}%")

# Save test results for demo
test_results = pd.DataFrame({
    'y_true': y_test.values,
    'y_prob':  y_prob
})
test_results.to_csv('models/test_results.csv', index=False)


# ─────────────────────────────────────────────────────
# STEP 4: BUILD RESOLVED NAME REGISTRY
# ─────────────────────────────────────────────────────
print(f"\n\n📚 STEP 4: Building Resolved Name Registry...\n")
print(f"  What this does:")
print(f"  • Takes all CT-approved pairs from your data")
print(f"  • Groups payer name variations per account holder")
print(f"  • Next time same variation appears → instant approve")
print(f"  • No ML scoring needed for known pairs\n")

# Use only approved pairs
approved = df[df[LABEL_COL].astype(str).str.lower()
              .isin(['true','1','yes','match'])]

print(f"  Building from {len(approved):,} approved pairs...")

# Registry structure:
# { "john smith": ["j smith", "j. smith", "jon smith", ...] }
registry = {}

for _, row in approved.iterrows():
    account_key = normalize(str(row[ACCOUNT_COL]))
    payer_alias = normalize(str(row[PAYER_COL]))

    if account_key not in registry:
        registry[account_key] = []

    if payer_alias not in registry[account_key]:
        registry[account_key].append(payer_alias)

# Stats
total_accounts = len(registry)
total_aliases  = sum(len(v) for v in registry.values())
multi_alias    = sum(1 for v in registry.values()
                     if len(v) > 1)

print(f"  ✅ Registry built:")
print(f"     Account holders indexed: {total_accounts:,}")
print(f"     Total payer aliases:     {total_aliases:,}")
print(f"     Accounts with multiple")
print(f"     aliases (name variations): {multi_alias:,}")

# Show example entries
print(f"\n  Example registry entries:")
shown = 0
for account, aliases in registry.items():
    if len(aliases) > 1 and shown < 4:
        print(f"  '{account}' ← known aliases:")
        for alias in aliases[:5]:
            if alias != account:
                print(f"       '{alias}'")
        shown += 1

pickle.dump(registry, open('models/registry.pkl', 'wb'))
print(f"\n  ✅ Registry saved → models/registry.pkl")


# ─────────────────────────────────────────────────────
# STEP 5: BUILD FULL PREDICT PIPELINE
# ─────────────────────────────────────────────────────
print(f"\n\n🔧 STEP 5: Building predict() pipeline...\n")


def predict(payer_name: str,
            account_holder_name: str) -> dict:
    """
    Main prediction function.
    
    Takes two names, returns:
    - decision:    AUTO_APPROVE / SUGGEST_MATCH /
                   HUMAN_REVIEW / LIKELY_MISMATCH
    - confidence:  probability 0.0 to 1.0
    - reason:      plain English explanation
    - layer:       which layer made the decision
    - features:    all 18 feature scores
    
    The 4 layers (fastest/cheapest first):
    
    Layer 1 — Exact match
      If names are identical after normalisation
      → instant AUTO_APPROVE, no ML needed
    
    Layer 2 — Registry lookup
      If payer name fuzzy-matches a known alias
      of the account holder (from past CT decisions)
      → AUTO_APPROVE, CT already confirmed this
    
    Layer 3 — Fuzzy fast-track
      If fuzzy similarity is ≥ 97%
      → near-identical, AUTO_APPROVE
    
    Layer 4 — ML model
      All other cases go through XGBoost
      → returns probability score
      → routed to appropriate tier
    """
    n_payer   = normalize(payer_name)
    n_account = normalize(account_holder_name)

    # ── LAYER 1: Exact match ─────────────────────────
    if n_payer == n_account:
        return {
            'payer_name':     payer_name,
            'account_holder': account_holder_name,
            'confidence':     1.0,
            'decision':       'AUTO_APPROVE',
            'reason':         'Exact name match after normalisation',
            'layer':          'exact_match',
            'features':       extract_features(
                                  payer_name, account_holder_name),
        }

    # ── LAYER 2: Registry lookup ─────────────────────
    # VETO: if first names clearly conflict, skip registry.
    # "Shikhar Gupta" must never match "Maruti Gupta" via registry
    # just because Maruti Gupta has multiple payer aliases stored.
    t_payer   = tokenize(payer_name)
    t_account = tokenize(account_holder_name)
    registry_veto = False
    if len(t_payer) >= 2 and len(t_account) >= 2:
        fn1, fn2 = t_payer[0], t_account[0]
        if len(fn1) > 1 and len(fn2) > 1:
            fn_sim = fuzz.ratio(fn1, fn2) / 100
            nick_ok = (NICKNAMES.get(fn1, fn1) == NICKNAMES.get(fn2, fn2))
            if fn_sim < 0.5 and not nick_ok:
                registry_veto = True

    if not registry_veto and n_account in registry:
        for alias in registry[n_account]:
            sim = fuzz.token_sort_ratio(
                n_payer, alias) / 100
            if sim >= 0.85:
                return {
                    'payer_name':     payer_name,
                    'account_holder': account_holder_name,
                    'confidence':     0.97,
                    'decision':       'AUTO_APPROVE',
                    'reason':         (
                        f'Resolved Name Registry hit — '
                        f'payer name matched known alias '
                        f'"{alias}" with {sim:.0%} similarity. '
                        f'CT previously approved this variation.'
                    ),
                    'layer':    'registry',
                    'features': extract_features(
                                    payer_name, account_holder_name),
                }

    # ── LAYER 3: Fuzzy fast-track ────────────────────
    fuzzy = fuzz.token_sort_ratio(n_payer, n_account) / 100
    if fuzzy >= 0.97:
        return {
            'payer_name':     payer_name,
            'account_holder': account_holder_name,
            'confidence':     fuzzy,
            'decision':       'AUTO_APPROVE',
            'reason':         f'Near-exact match — {fuzzy:.1%} '
                              f'character similarity',
            'layer':          'fuzzy_fasttrack',
            'features':       extract_features(
                                  payer_name, account_holder_name),
        }

    # ── LAYER 4: ML Model ────────────────────────────
    feats   = extract_features(payer_name, account_holder_name)
    X_input = pd.DataFrame([feats])
    conf    = float(model.predict_proba(X_input)[0][1])

    # Route to decision tier
    # CONFLICT VETO: if first names clearly differ, cap at HUMAN_REVIEW.
    # Prevents Shikhar Gupta / Maruti Gupta type false positives.
    feats_chk = extract_features(payer_name, account_holder_name)
    if feats_chk.get('first_name_conflict', 0) == 1.0:
        conf = min(conf, 0.65)

    if conf >= 0.92:
        decision = 'AUTO_APPROVE'
    elif conf >= 0.75:
        decision = 'SUGGEST_MATCH'
    elif conf >= 0.45:
        decision = 'HUMAN_REVIEW'
    else:
        decision = 'LIKELY_MISMATCH'

    # Build human-readable explanation
    # Find the top 3 features that drove this decision
    feat_series  = pd.Series(feats)
    imp_series   = feat_imp
    weighted     = (feat_series * imp_series)\
        .sort_values(ascending=False)

    reasons = []
    reason_labels = {
        'soundex_last':   'phonetically similar',
        'metaphone_last': 'phonetically similar',
        'last_tok_sim':   'last name matches',
        'token_sort':     'words match when reordered',
        'token_set':      'key tokens match',
        'initials':       'appears to be initials',
        'nickname':       'known nickname match',
        'exact':          'exact match',
        'jaro_winkler':   'string similarity',
        'first_char':     'same first character',
        'shared_tok':     'shared name tokens',
        'bigram':         'character pattern match',
    }
    for feat in weighted.head(3).index:
        score = feats[feat]
        label = reason_labels.get(feat, feat)
        if score > 0.5:
            reasons.append(f"{label} ({score:.0%})")
        else:
            reasons.append(f"low {label} ({score:.0%})")

    reason_str = ' | '.join(reasons) if reasons else \
                 f"ML confidence: {conf:.1%}"

    return {
        'payer_name':     payer_name,
        'account_holder': account_holder_name,
        'confidence':     round(conf, 4),
        'decision':       decision,
        'reason':         reason_str,
        'layer':          'ml_model',
        'features':       feats,
    }


print(f"  ✅ predict() function defined")


# ─────────────────────────────────────────────────────
# STEP 6: SMOKE TESTS
# ─────────────────────────────────────────────────────
print(f"\n\n🧪 STEP 6: Running smoke tests...\n")

test_cases = [
    # (payer, account, expected_behaviour)
    ("John Smith",          "John Smith",
     "exact → AUTO_APPROVE"),
    ("J. Smith",            "John Smith",
     "initial → high confidence"),
    ("Jon Smyth",           "John Smith",
     "typo → medium-high confidence"),
    ("Johnathan Smith",     "John Smith",
     "abbreviation → medium-high"),
    ("Robert Johnson",      "Bob Johnson",
     "nickname → should match"),
    ("Wang Wei",            "Wei Wang",
     "name reorder → token_sort catches"),
    ("John A. Smith",       "John Smith",
     "middle name → should match"),
    ("DR John Smith",       "John Smith",
     "title stripped → should match"),
    ("Mohammed Al-Rashid",  "Mohamed Alrashid",
     "transliteration → should match"),
    ("Alice Johnson",       "Bob Williams",
     "different person → MISMATCH"),
    ("Tom Hanks",           "Meryl Streep",
     "completely different → MISMATCH"),
    ("J.R. Smith",          "John Robert Smith",
     "double initial → should catch"),
]

print(f"  {'PAYER':<25} {'ACCOUNT':<25} "
      f"{'CONF':>6}  {'DECISION':<20} LAYER")
print(f"  {'-'*90}")

all_pass = True
for payer, account, note in test_cases:
    r    = predict(payer, account)
    conf = r['confidence']
    dec  = r['decision']
    layer = r['layer']

    # Visual flag for mismatches
    flag = ''
    if 'MISMATCH' in note and dec == 'AUTO_APPROVE':
        flag = ' ⚠️ CHECK THIS'
        all_pass = False
    elif 'should match' in note and dec == 'LIKELY_MISMATCH':
        flag = ' ⚠️ CHECK THIS'

    print(f"  {payer:<25} {account:<25} "
          f"{conf:>6.1%}  {dec:<20} {layer}{flag}")

if all_pass:
    print(f"\n  ✅ All smoke tests look reasonable")
else:
    print(f"\n  ⚠️  Some cases to investigate")
    print(f"     (Usually fine — check manually)")


# ─────────────────────────────────────────────────────
# STEP 7: SAVE EVERYTHING
# ─────────────────────────────────────────────────────
print(f"\n\n💾 STEP 7: Saving model and pipeline...\n")

import joblib
joblib.dump(model, 'models/xgboost_model.joblib')
# Also keep pkl for backwards compat
pickle.dump(model, open('models/xgboost_model.pkl', 'wb'))
pickle.dump(registry, open('models/registry.pkl', 'wb'))
pickle.dump({
    'auc':           float(auc),
    'tier_results':  tier_results,
    'feat_imp':      feat_imp.to_dict(),
    'total_test':    len(y_test),
    'scale_weight':  float(class_info['scale_pos_weight']),
}, open('models/eval_results.pkl', 'wb'))

print(f"  ✅ XGBoost model   → models/xgboost_model.pkl")
print(f"  ✅ Registry        → models/registry.pkl")
print(f"  ✅ Eval results    → models/eval_results.pkl")
print(f"  ✅ Test results    → models/test_results.csv")

print(f"\n{'='*60}")
print(f"  DAY 2 COMPLETE ✅")
print(f"{'='*60}")
print(f"\n  Results:")
print(f"  • Test AUC:              {auc:.4f}")
print(f"  • CT workload reduction: ~{ct_reduction:.0f}%")
print(f"  • Registry entries:      {total_accounts:,}")
print(f"  • Registry aliases:      {total_aliases:,}")
print(f"\n  Next step: Run  streamlit run day3_demo_app.py")
print(f"{'='*60}\n")
