"""
DAY 1 SCRIPT — Data Exploration + Feature Extraction
=====================================================
WHAT THIS DOES:
  Step 1: Loads your CSV and checks for problems
  Step 2: Extracts 18 similarity features per name pair
  Step 3: Trains a quick baseline model
  Step 4: Saves features to disk for Day 2

HOW TO RUN:
  python day1_explore_and_features.py

WHAT YOU NEED:
  Your CSV file with these columns:
    - payer_name
    - account_holder_name
    - ct_match  (True/False)

  Put your CSV in the data/ folder.
  Change DATA_FILE below to match your filename.
"""

# ─────────────────────────────────────────────────────
# CONFIGURATION — Change these to match your setup
# ─────────────────────────────────────────────────────
DATA_FILE    = "data/sample_ct_data.csv"   # ← your CSV file
PAYER_COL    = "payer_name"                # ← column name for payer
ACCOUNT_COL  = "account_holder_name"       # ← column name for account holder
LABEL_COL    = "ct_match"                  # ← column name for True/False label
# ─────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import re
import unicodedata
import pickle
import warnings
warnings.filterwarnings('ignore')

from rapidfuzz import fuzz
import jellyfish
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("  DAY 1: DATA EXPLORATION + FEATURE EXTRACTION")
print("=" * 60)


# ─────────────────────────────────────────────────────
# STEP 1: LOAD AND CHECK YOUR DATA
# ─────────────────────────────────────────────────────
print("\n📂 STEP 1: Loading your data...\n")

df = pd.read_csv(DATA_FILE)

# Convert ct_match to proper boolean
# Handles: True/False strings, 1/0, yes/no
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()
df[LABEL_COL] = df[LABEL_COL].map({
    'true': True, '1': True, 'yes': True, 'match': True,
    'false': False, '0': False, 'no': False, 'no match': False,
    'nomatch': False
})

print(f"  ✅ Rows loaded:     {len(df):,}")
print(f"  ✅ Columns found:   {df.columns.tolist()}")

# Check for missing values
nulls = df[[PAYER_COL, ACCOUNT_COL, LABEL_COL]].isnull().sum()
if nulls.sum() > 0:
    print(f"\n  ⚠️  Missing values found:")
    print(f"     {nulls[nulls > 0].to_dict()}")
    print(f"  → Removing rows with missing values...")
    df = df.dropna(subset=[PAYER_COL, ACCOUNT_COL, LABEL_COL])
    print(f"  ✅ Rows after cleaning: {len(df):,}")
else:
    print(f"  ✅ No missing values — clean data!")

# Class balance — very important
match_count    = df[LABEL_COL].sum()
no_match_count = (~df[LABEL_COL]).sum()
match_rate     = match_count / len(df)

print(f"\n  📊 CLASS BALANCE:")
print(f"     True  (match):    {match_count:,}  ({match_rate:.1%})")
print(f"     False (no match): {no_match_count:,}  ({1-match_rate:.1%})")

if match_rate < 0.15:
    print(f"\n  ⚠️  WARNING: Very few matches ({match_rate:.1%})")
    print(f"     XGBoost will use scale_pos_weight to handle this")
elif match_rate > 0.85:
    print(f"\n  ⚠️  WARNING: Very few rejections ({1-match_rate:.1%})")
    print(f"     Check if your data is representative of real CT cases")
else:
    print(f"  ✅ Class balance looks good")

# Show sample rows
print(f"\n  👀 SAMPLE TRUE MATCHES (5 random):")
matches = df[df[LABEL_COL] == True]
for _, row in matches.sample(min(5, len(matches))).iterrows():
    print(f"     '{row[PAYER_COL]}' → '{row[ACCOUNT_COL]}'")

print(f"\n  👀 SAMPLE REJECTIONS (5 random):")
rejects = df[df[LABEL_COL] == False]
for _, row in rejects.sample(min(5, len(rejects))).iterrows():
    print(f"     '{row[PAYER_COL]}' → '{row[ACCOUNT_COL]}'")


# ─────────────────────────────────────────────────────
# STEP 2: FEATURE EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────────
print("\n\n🔧 STEP 2: Building feature extractor...\n")

# Titles to strip from names
# These add noise — "Mr John Smith" should equal "John Smith"
TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof',
    'rev', 'sir', 'lord', 'jr', 'sr', 'ii', 'iii', 'iv',
    'monsieur', 'madame', 'mme', 'mlle', 'mademoiselle',
}

# Nickname mappings
# "Bob" → "Robert", "Bill" → "William", etc.
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
    'tricia': 'patricia', 'trishia': 'patricia',
    'eliza': 'elizabeth', 'lisa': 'elizabeth',
    'dom': 'dominic',
    'rand': 'randolph', 'randy': 'randolph',
    'sly': 'sylvester',
    'bart': 'bartholomew',
    'al': 'alexander',
    'jp': 'jeanpierre', 'jeanpierre': 'jeanpierre',
    'jl': 'jeanluc',    'jeanluc': 'jeanluc',
    'jm': 'jeanmarc',   'jeanmarc': 'jeanmarc',
    'jf': 'jeanfrancois', 'jc': 'jeanchristophe',
    'jb': 'jeanbaptiste', 'jn': 'jeannoel',
    'pierrot': 'pierre', 'jeannot': 'jean',
    'mado': 'madeleine', 'loulou': 'louise',
}


def normalize(name: str) -> str:
    """
    Clean a name for comparison.
    
    What it does:
    1. Removes accents: é → e, ü → u
    2. Lowercases: JOHN → john
    3. Removes punctuation: J. Smith → j smith
    4. Collapses spaces: John  Smith → john smith
    
    Example:
        "Mr. JOHN Smith Jr." → "mr john smith jr"
        (titles removed separately in tokenize)
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    # Remove accents (é, ü, ñ etc.)
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name
                   if not unicodedata.combining(c))
    # Lowercase
    name = name.lower()
    # Remove everything except letters, numbers, spaces
    name = re.sub(r'[^a-z0-9 ]', ' ', name)
    # French/Dutch particle normalisation
    # Merge particles so edit-distance features score higher:
    # "de la Tour" -> "dela Tour" ~ "Delatour"
    # "van den Berg" -> "vanden Berg" ~ "Vandenberg"
    name = re.sub('de la', 'dela', name)
    name = re.sub('van den', 'vanden', name)
    name = re.sub('van der', 'vander', name)
    # Collapse multiple spaces into one
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def tokenize(name: str) -> list:
    """
    Split a name into individual words,
    removing titles like Mr, Dr, Jr.
    
    Example:
        "Dr. John A. Smith Jr." → ["john", "a", "smith"]
    """
    tokens = normalize(name).split()
    return [t for t in tokens
            if t not in TITLES and len(t) > 0]


def initials_score(t1: list, t2: list) -> float:
    """
    Checks if a single letter could be an initial
    of a full name token.
    
    Example:
        "J. Smith" → tokens: ["j", "smith"]
        "John Smith" → tokens: ["john", "smith"]
        
        "j" matches "john" because john starts with j
        "smith" matches "smith" exactly
        Score = (0.9 + 1.0) / 2 = 0.95
    """
    if not t1 or not t2:
        return 0.0
    scores = []
    for a in t1:
        best = 0.0
        for b in t2:
            if len(a) == 1 and b.startswith(a):
                # a is an initial of b
                best = max(best, 0.9)
            elif len(b) == 1 and a.startswith(b):
                # b is an initial of a
                best = max(best, 0.9)
            elif a == b:
                # Exact token match
                best = 1.0
        scores.append(best)
    return sum(scores) / len(scores)


def nickname_score(t1: list, t2: list) -> float:
    """
    Checks if any token is a known nickname of another.
    
    Example:
        "Bob Johnson" → tokens: ["bob", "johnson"]
        "Robert Johnson" → tokens: ["robert", "johnson"]
        
        NICKNAMES["bob"] = "robert"
        "bob" and "robert" map to same canonical name
        → returns 1.0 (nickname match found)
    """
    if not t1 or not t2:
        return 0.0
    for a in t1:
        for b in t2:
            # Map both to canonical form
            a_canon = NICKNAMES.get(a, a)
            b_canon = NICKNAMES.get(b, b)
            # If they map to same canonical AND
            # they're not already the same word
            if a_canon == b_canon and a != b:
                return 1.0
    return 0.0


def bigram_similarity(s1: str, s2: str) -> float:
    """
    Compares character pairs (bigrams) between strings.
    
    Example:
        "john" → bigrams: {jo, oh, hn}
        "jon"  → bigrams: {jo, on}
        Shared: {jo} = 1
        Total unique: {jo, oh, hn, on} = 4
        Score = 2 * 1 / (3 + 2) = 0.4
    
    Good for catching typos and partial matches.
    """
    def get_bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))

    a = get_bigrams(s1)
    b = get_bigrams(s2)

    if not a or not b:
        return 0.0

    # Dice coefficient formula
    return 2 * len(a & b) / (len(a) + len(b))


def extract_features(payer: str, account: str) -> dict:
    """
    The main function — converts two name strings
    into 18 numbers that the ML model can understand.
    
    Each number is between 0 (completely different)
    and 1 (identical).
    
    Args:
        payer:   name from the payment sender
        account: name on the receiving account
    
    Returns:
        dict of 18 feature scores
    """
    # Clean both names first
    n1 = normalize(payer)
    n2 = normalize(account)
    t1 = tokenize(payer)
    t2 = tokenize(account)

    # Safe denominators — avoid division by zero
    max_len = max(len(n1), len(n2), 1)
    max_tok = max(len(t1), len(t2), 1)

    # ── FEATURE GROUP 1: Whole name similarity ─────
    # These compare the full name string

    exact = 1.0 if n1 == n2 else 0.0
    # Are they exactly the same after normalisation?
    # "JOHN SMITH" vs "john smith" → 1.0 (after normalise)

    jaro_winkler = fuzz.ratio(n1, n2) / 100
    # Overall string similarity, 0-100 scaled to 0-1
    # Jaro-Winkler gives bonus for matching prefix
    # "John Smith" vs "John Smyth" → 0.93

    token_sort = fuzz.token_sort_ratio(n1, n2) / 100
    # Sorts tokens alphabetically before comparing
    # "Wang Wei" vs "Wei Wang" → 1.0 (both become "wang wei")

    token_set = fuzz.token_set_ratio(n1, n2) / 100
    # Finds the best matching subset of tokens
    # "John A Smith" vs "John Smith" → 0.95
    # (handles middle names/initials gracefully)

    partial = fuzz.partial_ratio(n1, n2) / 100
    # Finds best matching substring
    # "Jon" vs "Jonathan" → 1.0 (jon is inside jonathan)

    # ── FEATURE GROUP 2: Edit distance ─────────────
    # How many character changes to get from one to other?

    lev_dist = jellyfish.levenshtein_distance(n1, n2)
    # "Jon" → "John": 1 insertion = distance 1
    # "Smith" → "Smyth": 1 substitution = distance 1
    levenshtein = 1 - (lev_dist / max_len)
    # Normalised: 0 = completely different, 1 = identical

    # ── FEATURE GROUP 3: Character n-grams ─────────

    bigram = bigram_similarity(n1, n2)
    # Character bigram overlap (see function above)

    # ── FEATURE GROUP 4: Token-level ───────────────
    # Compare individual name parts

    first_tok_sim = (
        fuzz.ratio(t1[0], t2[0]) / 100
        if t1 and t2 else 0.0
    )
    # How similar are the first tokens?
    # "John" vs "Jon" → 0.86

    last_tok_sim = (
        fuzz.ratio(t1[-1], t2[-1]) / 100
        if t1 and t2 else 0.0
    )
    # How similar are the last tokens (surnames)?
    # "Smith" vs "Smith" → 1.0
    # Last name is a STRONG signal for identity

    initials = initials_score(t1, t2)
    # Does "J." match "John"? (see function above)

    nickname = nickname_score(t1, t2)
    # Does "Bob" match "Robert"? (see function above)

    shared_tok = len(set(t1) & set(t2)) / max_tok
    # What fraction of tokens are shared?
    # ["john", "smith"] & ["john", "smith"] = 2/2 = 1.0
    # ["j", "smith"] & ["john", "smith"] = 1/2 = 0.5

    # ── FEATURE GROUP 5: Phonetic ──────────────────
    # Do the names SOUND the same?

    soundex_last = 0.0
    metaphone_last = 0.0
    if t1 and t2:
        # Compare last names phonetically
        soundex_last = (
            1.0 if jellyfish.soundex(t1[-1]) ==
                   jellyfish.soundex(t2[-1])
            else 0.0
        )
        # Soundex: converts name to code based on sound
        # "Smith" → S530, "Smyth" → S530 → match!
        # "Mohamed" → M530, "Mohammed" → M530 → match!

        metaphone_last = (
            1.0 if jellyfish.metaphone(t1[-1]) ==
                   jellyfish.metaphone(t2[-1])
            else 0.0
        )
        # Metaphone: more sophisticated phonetic algorithm

    # ── FEATURE GROUP 6: Structural ────────────────
    # Shape and structure of the names

    len_diff = abs(len(n1) - len(n2)) / max_len
    # "Jo" vs "Josephine" → big length difference → high score
    # (high len_diff = MORE different)

    tok_count_diff = abs(len(t1) - len(t2)) / max_tok
    # "John Smith" (2 tokens) vs "John A Smith" (3 tokens)
    # → 1/3 = 0.33 difference

    first_char = (
        1.0 if (n1 and n2 and n1[0] == n2[0])
        else 0.0
    )
    # Do both names start with the same letter?
    # Weak signal but cheap to compute

    # -- CONFLICT DETECTION (THE FIX for same-surname false positives) --
    # Problem: "Shikhar Gupta" vs "Maruti Gupta"
    #   last_tok_sim=1.0 (Gupta=Gupta) inflated scores.
    #   We need to know the FIRST NAME clearly differs.
    #
    # first_name_conflict = 1.0 when:
    #   Both names have real first names (2+ tokens, not initials)
    #   AND first names are clearly different (similarity < 0.5)
    #   AND they are not a known nickname pair
    # CONFLICT DETECTION -- checks ALL name tokens, not just first.
    # Fixes: "Jean Pierre Dubois" vs "Jean Parker Dubois" was falsely matching
    # because only t1[0] vs t2[0] (jean==jean) was being compared.
    # Now checks every non-surname, non-initial token with phonetics.
    def _has_match(tok, candidates):
        for other in candidates:
            if len(tok) == 1 or len(other) == 1:
                if tok[0] == other[0]: return True
                continue
            sim    = fuzz.ratio(tok, other) / 100
            sdx_ok = jellyfish.soundex(tok)  == jellyfish.soundex(other)
            mtp_ok = jellyfish.metaphone(tok) == jellyfish.metaphone(other)
            nck_ok = NICKNAMES.get(tok, tok)  == NICKNAMES.get(other, other)
            if sim >= 0.6 or sdx_ok or mtp_ok or nck_ok:
                return True
        return False
    first_name_conflict = 0.0
    if len(t1) >= 2 and len(t2) >= 2:
        ns1 = [t for t in t1[:-1] if len(t) > 1]
        ns2 = [t for t in t2[:-1] if len(t) > 1]
        if ns1 and ns2:
            for tok in ns1:
                if not _has_match(tok, ns2):
                    first_name_conflict = 1.0
                    break

    # both_names_high: 1.0 only when BOTH first AND last match well
    # "John Smith" vs "John Smith" -> 1.0
    # "Shikhar Gupta" vs "Maruti Gupta" -> 0.0  (first name fails)
    both_names_high = 0.0
    if t1 and t2:
        fn_s = fuzz.ratio(t1[0], t2[0]) / 100 if (len(t1) >= 2 and len(t2) >= 2) else 1.0
        if fn_s >= 0.7 and last_tok_sim >= 0.7:
            both_names_high = (fn_s + last_tok_sim) / 2

    return {
        'exact':               exact,
        'jaro_winkler':        jaro_winkler,
        'token_sort':          token_sort,
        'token_set':           token_set,
        'partial':             partial,
        'levenshtein':         levenshtein,
        'bigram':              bigram,
        'first_tok_sim':       first_tok_sim,
        'last_tok_sim':        last_tok_sim,
        'initials':            initials,
        'nickname':            nickname,
        'shared_tok':          shared_tok,
        'soundex_last':        soundex_last,
        'metaphone_last':      metaphone_last,
        'len_diff':            len_diff,
        'tok_count_diff':      tok_count_diff,
        'first_char':          first_char,
        # NEW: conflict detection
        'first_name_conflict': first_name_conflict,
        'both_names_high':     both_names_high,
        # FIXED composite: last_tok_sim removed
        # Surname alone must never drive auto-approval
        'fuzzy_composite': (
            jaro_winkler * 0.4 +
            token_sort   * 0.35 +
            token_set    * 0.25
        ),
    }



# ─────────────────────────────────────────────────────
# STEP 3: APPLY FEATURES TO ALL ROWS
# ─────────────────────────────────────────────────────
print("  ✅ Feature functions defined")
print("\n  📊 Extracting features from all rows...")
print("     (This may take 1-2 minutes for 15,000 rows)")

features_list = []
for i, row in df.iterrows():
    feats = extract_features(
        str(row[PAYER_COL]),
        str(row[ACCOUNT_COL])
    )
    features_list.append(feats)

    # Progress indicator every 1000 rows
    if (i + 1) % 1000 == 0:
        print(f"     Processed {i+1:,} rows...")

X = pd.DataFrame(features_list)
y = df[LABEL_COL].astype(int)

print(f"\n  ✅ Feature matrix shape: {X.shape}")
print(f"     ({X.shape[0]:,} rows × {X.shape[1]} features)")

# Show feature statistics
print(f"\n  📊 FEATURE SUMMARY (averages by class):")
print(f"  {'Feature':<20} {'TRUE (match)':<15} {'FALSE (no match)':<18} {'Difference'}")
print(f"  {'-'*65}")
X_copy = X.copy()
X_copy['label'] = y
for feat in X.columns:
    true_avg  = X_copy[X_copy['label']==1][feat].mean()
    false_avg = X_copy[X_copy['label']==0][feat].mean()
    diff      = abs(true_avg - false_avg)
    # Highlight the most discriminating features
    marker = " ⭐" if diff > 0.3 else ""
    print(f"  {feat:<20} {true_avg:<15.3f} "
          f"{false_avg:<18.3f} {diff:.3f}{marker}")


# ─────────────────────────────────────────────────────
# STEP 4: BASELINE MODEL
# ─────────────────────────────────────────────────────
print(f"\n\n📈 STEP 4: Training baseline model...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=42)
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

scores = cross_val_score(lr, X, y, cv=cv,
                         scoring='roc_auc')
baseline_auc = scores.mean()

print(f"  ✅ Logistic Regression (5-fold CV)")
print(f"     AUC:  {baseline_auc:.4f}  "
      f"(± {scores.std():.4f})")
print(f"     Each fold: "
      f"{[f'{s:.3f}' for s in scores]}")

print(f"\n  INTERPRETATION:")
if baseline_auc >= 0.95:
    print(f"  ✅ EXCELLENT — Data is very clean and "
          f"features are highly discriminating")
    print(f"     XGBoost should push this to 0.97+")
elif baseline_auc >= 0.90:
    print(f"  ✅ VERY GOOD — Strong signal in your data")
    print(f"     XGBoost should push this to 0.93-0.96")
elif baseline_auc >= 0.85:
    print(f"  ✅ GOOD — Solid baseline")
    print(f"     XGBoost should push this to 0.90-0.94")
elif baseline_auc >= 0.75:
    print(f"  ⚠️  MODERATE — Some noise in data")
    print(f"     Check for mislabelled rows")
else:
    print(f"  ❌ LOW — Data quality issue")
    print(f"     Before Day 2, manually check 20 rows "
          f"where features are high but label is False")


# ─────────────────────────────────────────────────────
# STEP 5: SAVE EVERYTHING FOR DAY 2
# ─────────────────────────────────────────────────────
print(f"\n\n💾 STEP 5: Saving for Day 2...\n")

# Save the feature matrix
X.to_csv('models/features.csv', index=False)
print(f"  ✅ Features saved → models/features.csv")

# Save the labels
y.to_csv('models/labels.csv', index=False)
print(f"  ✅ Labels saved   → models/labels.csv")

# Save the cleaned dataframe
df.to_csv('models/cleaned_data.csv', index=False)
print(f"  ✅ Clean data saved → models/cleaned_data.csv")

# Save the feature extractor functions
# (we do this by saving the column names)
pickle.dump(X.columns.tolist(),
            open('models/feature_names.pkl', 'wb'))
print(f"  ✅ Feature names saved → models/feature_names.pkl")

# Save class balance info
class_info = {
    'match_rate':    float(match_rate),
    'scale_pos_weight': float((1 - match_rate) / match_rate),
    'total_rows':    len(df),
    'baseline_auc':  float(baseline_auc),
    'payer_col':     PAYER_COL,
    'account_col':   ACCOUNT_COL,
    'label_col':     LABEL_COL,
}
pickle.dump(class_info,
            open('models/class_info.pkl', 'wb'))
print(f"  ✅ Class info saved → models/class_info.pkl")

print(f"\n{'='*60}")
print(f"  DAY 1 COMPLETE ✅")
print(f"{'='*60}")
print(f"\n  Summary:")
print(f"  • Rows in dataset:    {len(df):,}")
print(f"  • Match rate:         {match_rate:.1%}")
print(f"  • Features created:   {X.shape[1]}")
print(f"  • Baseline AUC:       {baseline_auc:.4f}")
print(f"\n  Next step: Run  python day2_train_model.py")
print(f"{'='*60}\n")
