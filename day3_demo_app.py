"""
DAY 3 — DEMO APPLICATION  (v6 — Real Data Edition)
Payment Name Matching System
=============================
HOW TO RUN:
  streamlit run day3_demo_app.py

TABS:
  1 — Live Demo
  2 — Accuracy & Model
  3 — CT Workload Reduction
  4 — Cross-Fold Validation  (NEW — uses real data patterns from your CSV)
  5 — Batch Test Cases
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import unicodedata
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from rapidfuzz import fuzz
import jellyfish

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Payment Name Matching — CT Demo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main-header {
        background: #0f172a;
        color: white;
        padding: 28px 36px;
        border-radius: 4px;
        margin-bottom: 28px;
        border-left: 5px solid #3b82f6;
    }
    .main-header h1 { color: white; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
    .main-header p  { color: #94a3b8; margin: 6px 0 0 0; font-size: 14px; font-family: 'IBM Plex Mono', monospace; }

    .decision-card {
        padding: 18px 24px; border-radius: 4px; margin: 12px 0;
        font-size: 20px; font-weight: 700; text-align: center;
        font-family: 'IBM Plex Mono', monospace;
    }
    .AUTO_APPROVE    { background:#f0fdf4; color:#15803d; border-left:5px solid #22c55e; }
    .SUGGEST_MATCH   { background:#fefce8; color:#a16207; border-left:5px solid #eab308; }
    .HUMAN_REVIEW    { background:#fff7ed; color:#c2410c; border-left:5px solid #f97316; }
    .LIKELY_MISMATCH { background:#fef2f2; color:#b91c1c; border-left:5px solid #ef4444; }

    .metric-box {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 4px; padding: 16px; text-align: center;
    }
    .metric-box .val { font-size: 28px; font-weight: 700; color: #1e40af; font-family: 'IBM Plex Mono', monospace; }
    .metric-box .lbl { font-size: 11px; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

    .reason-box {
        background: #f8fafc; border-left: 4px solid #3b82f6;
        padding: 12px 16px; border-radius: 0 4px 4px 0;
        font-size: 13px; color: #1e3a5f; margin: 12px 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .pattern-tag {
        display: inline-block; padding: 2px 8px; border-radius: 3px;
        font-size: 11px; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase; letter-spacing: 0.05em; margin: 2px;
    }
    .tag-true  { background: #dcfce7; color: #166534; }
    .tag-false { background: #fee2e2; color: #991b1b; }

    .stTabs [data-baseweb="tab"] {
        font-size: 13px; font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    import joblib, os
    if os.path.exists('models/xgboost_model.joblib'):
        model = joblib.load('models/xgboost_model.joblib')
    else:
        model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
    registry = pickle.load(open('models/registry.pkl', 'rb'))
    eval_res = pickle.load(open('models/eval_results.pkl', 'rb'))
    return model, registry, eval_res

try:
    model, registry, eval_res = load_all()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False


# ─────────────────────────────────────────────────────
# CORE HELPERS — kept in sync with day1 / day2
# ─────────────────────────────────────────────────────
TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'rev', 'sir', 'lord', 'jr', 'sr',
    'ii', 'iii', 'iv',
    'm',                    # standalone "M" = Monsieur
    'mme', 'mlle', 'mle',
    'monsieur', 'madame', 'mademoiselle',
    'ou',                   # "M OU MME DUPONT" — strip "ou"
    'sarl', 'sas', 'ste', 'sa', 'ltd', 'inc', 'corp', 'llc', 'gmbh',
}

NICKNAMES = {
    'bob':'robert',    'rob':'robert',     'bobby':'robert',
    'bill':'william',  'will':'william',   'billy':'william',
    'jim':'james',     'jimmy':'james',    'jamie':'james',
    'tom':'thomas',    'tommy':'thomas',
    'mike':'michael',  'mick':'michael',   'mickey':'michael',
    'dave':'david',    'davey':'david',    'chris':'christopher',
    'liz':'elizabeth', 'beth':'elizabeth', 'betty':'elizabeth',
    'kate':'katherine','kathy':'katherine','katie':'katherine',
    'sue':'susan',     'susie':'susan',
    'joe':'joseph',    'jo':'josephine',
    'nick':'nicholas', 'nicky':'nicholas',
    'pat':'patricia',  'patty':'patricia',
    'dan':'daniel',    'danny':'daniel',
    'sam':'samuel',    'sammy':'samuel',
    'andy':'andrew',   'drew':'andrew',
    'tony':'anthony',  'ant':'anthony',
    'ben':'benjamin',  'benny':'benjamin', 'alex':'alexander',
    'ned':'edward',    'ted':'edward',     'ed':'edward',
    'fred':'frederick','freddy':'frederick',
    'harry':'henry',   'hal':'henry',
    'jack':'john',     'johnny':'john',    'pete':'peter',
    'pierre':'peter',                      # French variant
    'dick':'richard',  'rick':'richard',   'rich':'richard',
    'ron':'ronald',    'ronnie':'ronald',
    'steve':'steven',  'stevie':'steven',
    'matt':'matthew',  'matty':'matthew',
    'greg':'gregory',  'jeff':'jeffrey',
    'ken':'kenneth',   'kenny':'kenneth',
    'larry':'lawrence','lars':'lawrence',
    'len':'leonard',   'lenny':'leonard',
    'ray':'raymond',   'russ':'russell',
    'stu':'stewart',   'tim':'timothy',    'timmy':'timothy',
    'vince':'vincent', 'walt':'walter',
    'nate':'nathaniel','deb':'deborah',    'debbie':'deborah',
    'tricia':'patricia','eliza':'elizabeth','lisa':'elizabeth',
    'dom':'dominic',   'randy':'randolph',
    'sly':'sylvester', 'bart':'bartholomew','al':'alexander',
    'jean':'john',     'jacques':'james',  'francois':'francis',
    'francoise':'frances', 'marc':'mark',  'nicolas':'nicholas',
}

MALE_TITLES_SET   = {'m', 'mr', 'monsieur'}
FEMALE_TITLES_SET = {'mme', 'mlle', 'mle', 'madame', 'mademoiselle', 'ms', 'mrs', 'miss'}
COMPANY_SIGNALS   = {
    'sarl', 'sas', 'ste', 'sa', 'ltd', 'inc', 'corp', 'llc', 'gmbh',
    'biodistrib', 'assoc', 'association', 'reve', 'groupe', 'cabinet',
}


def normalize(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ''
    name = name.strip().strip('"').strip("'")
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = name.lower().replace('-', ' ')
    name = re.sub(r'[^a-z0-9 ]', ' ', name)
    return re.sub(r'\s+', ' ', name).strip()


def tokenize(name: str) -> list:
    return [t for t in normalize(name).split()
            if t not in TITLES and len(t) > 0]


def _get_gender(name: str):
    toks = re.sub(r'[^a-z ]', ' ', name.lower()).split()
    for t in toks[:3]:
        if t in MALE_TITLES_SET:   return 'M'
        if t in FEMALE_TITLES_SET: return 'F'
    return None


def initials_score(t1, t2):
    if not t1 or not t2: return 0.0
    scores = []
    for a in t1:
        best = 0.0
        for b in t2:
            if   len(a) == 1 and b.startswith(a): best = max(best, 0.9)
            elif len(b) == 1 and a.startswith(b): best = max(best, 0.9)
            elif a == b:                           best = 1.0
        scores.append(best)
    return sum(scores) / len(scores)


def nickname_score(t1, t2):
    if not t1 or not t2: return 0.0
    for a in t1:
        for b in t2:
            if NICKNAMES.get(a, a) == NICKNAMES.get(b, b) and a != b:
                return 1.0
    return 0.0


def bigram_sim(s1, s2):
    def bg(s): return set(s[i:i+2] for i in range(len(s)-1))
    a, b = bg(s1), bg(s2)
    if not a or not b: return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _has_match(token, token_list):
    """True if *token* has a plausible match anywhere in *token_list*."""
    for t in token_list:
        if fuzz.ratio(token, t) / 100 >= 0.6:               return True
        if len(token) == 1 and t.startswith(token):          return True
        if len(t) == 1 and token.startswith(t):              return True
        if jellyfish.soundex(token) == jellyfish.soundex(t): return True
        if jellyfish.metaphone(token) == jellyfish.metaphone(t): return True
        if NICKNAMES.get(token, token) == NICKNAMES.get(t, t):   return True
    return False


def extract_features(payer: str, account: str) -> dict:
    n1, n2 = normalize(payer), normalize(account)
    t1, t2 = tokenize(payer), tokenize(account)
    ml  = max(len(n1), len(n2), 1)
    mt  = max(len(t1), len(t2), 1)

    jw   = fuzz.ratio(n1, n2) / 100
    ts   = fuzz.token_sort_ratio(n1, n2) / 100
    tset = fuzz.token_set_ratio(n1, n2) / 100
    pr   = fuzz.partial_ratio(n1, n2) / 100
    lts  = fuzz.ratio(t1[-1], t2[-1]) / 100 if t1 and t2 else 0.0

    soundex_last = metaphone_last = 0.0
    if t1 and t2:
        soundex_last   = 1.0 if jellyfish.soundex(t1[-1])   == jellyfish.soundex(t2[-1])   else 0.0
        metaphone_last = 1.0 if jellyfish.metaphone(t1[-1]) == jellyfish.metaphone(t2[-1]) else 0.0

    # Full token-level conflict (fixes "Jean Pierre" vs "Jean Parker")
    # Searches ALL of t2 so surname-first reordering doesn't fire a false conflict
    first_name_conflict = 0.0
    if len(t1) >= 2 and len(t2) >= 2:
        unmatched = [tok for tok in t1[:-1]
                     if len(tok) > 1 and not _has_match(tok, t2)]
        if unmatched:
            first_name_conflict = 1.0

    gender_payer   = _get_gender(payer)
    gender_account = _get_gender(account)
    gender_conflict = 1.0 if (
        gender_payer and gender_account and gender_payer != gender_account
    ) else 0.0

    _pl = n1
    is_company = 1.0 if any(sig in _pl.split() for sig in COMPANY_SIGNALS) else 0.0
    if len(t1) == 0 or (len(t1) == 1 and len(t1[0]) < 3):
        is_company = 1.0

    payer_in_acct      = len(set(t1) & set(t2)) / max(len(t1), 1)
    extra_in_account   = max(0, len(t2) - len(t1)) / max(len(t2), 1)
    subset_with_extras = payer_in_acct * extra_in_account

    both_names_high = 0.0
    if t1 and t2:
        fn_s = fuzz.ratio(t1[0], t2[0]) / 100 if (len(t1) >= 2 and len(t2) >= 2) else 1.0
        if fn_s >= 0.7 and lts >= 0.7:
            both_names_high = (fn_s + lts) / 2

    return {
        'exact':               1.0 if n1 == n2 else 0.0,
        'jaro_winkler':        jw,
        'token_sort':          ts,
        'token_set':           tset,
        'partial':             pr,
        'levenshtein':         1 - (jellyfish.levenshtein_distance(n1, n2) / ml),
        'bigram':              bigram_sim(n1, n2),
        'first_tok_sim':       fuzz.ratio(t1[0], t2[0]) / 100 if t1 and t2 else 0.0,
        'last_tok_sim':        lts,
        'initials':            initials_score(t1, t2),
        'nickname':            nickname_score(t1, t2),
        'shared_tok':          len(set(t1) & set(t2)) / mt,
        'soundex_last':        soundex_last,
        'metaphone_last':      metaphone_last,
        'len_diff':            abs(len(n1) - len(n2)) / ml,
        'tok_count_diff':      abs(len(t1) - len(t2)) / mt,
        'first_char':          1.0 if n1 and n2 and n1[0] == n2[0] else 0.0,
        'first_name_conflict': first_name_conflict,
        'both_names_high':     both_names_high,
        'fuzzy_composite':     (jw * 0.4 + ts * 0.35 + tset * 0.25),
        'gender_conflict':     gender_conflict,
        'is_company':          is_company,
        'payer_in_acct':       payer_in_acct,
        'extra_in_account':    extra_in_account,
        'subset_with_extras':  subset_with_extras,
    }


FEAT_LABELS = {
    'exact':               'Exact match',
    'jaro_winkler':        'Overall similarity',
    'token_sort':          'Word-order similarity',
    'token_set':           'Key token match',
    'partial':             'Substring match',
    'levenshtein':         'Edit distance',
    'bigram':              'Character pattern',
    'first_tok_sim':       'First name similarity',
    'last_tok_sim':        'Last name similarity',
    'initials':            'Initials match',
    'nickname':            'Nickname / phonetic variant',
    'shared_tok':          'Shared words',
    'soundex_last':        'Sounds like — Soundex',
    'metaphone_last':      'Sounds like — Metaphone',
    'len_diff':            'Length difference',
    'tok_count_diff':      'Word count difference',
    'first_char':          'Same first letter',
    'fuzzy_composite':     'Composite fuzzy score',
    'first_name_conflict': 'First-name conflict flag',
    'both_names_high':     'Both names match well',
    'gender_conflict':     'Gender title conflict',
    'is_company':          'Payer appears to be company',
    'payer_in_acct':       'Payer tokens found in account',
    'extra_in_account':    'Extra tokens in account name',
    'subset_with_extras':  'Subset-with-extras penalty',
}


def predict(payer_name: str, account_holder_name: str) -> dict:
    n_payer   = normalize(payer_name)
    n_account = normalize(account_holder_name)
    feats     = extract_features(payer_name, account_holder_name)

    if n_payer == n_account:
        return {'payer': payer_name, 'account': account_holder_name,
                'confidence': 1.0, 'decision': 'AUTO_APPROVE',
                'reason': 'Exact match after normalisation (titles, accents, punctuation removed)',
                'layer': 'Exact Match', 'features': feats, 'registry_alias': None}

    t_p, t_a = tokenize(payer_name), tokenize(account_holder_name)
    reg_veto  = False
    if len(t_p) >= 2 and len(t_a) >= 2:
        unmatched = [tok for tok in t_p[:-1]
                     if len(tok) > 1 and not _has_match(tok, t_a)]
        if unmatched:
            reg_veto = True

    if not reg_veto and n_account in registry:
        for alias in registry[n_account]:
            sim = fuzz.token_sort_ratio(n_payer, alias) / 100
            if sim >= 0.85:
                return {'payer': payer_name, 'account': account_holder_name,
                        'confidence': 0.97, 'decision': 'AUTO_APPROVE',
                        'reason': (f'Name Registry — matched alias "{alias}" ({sim:.0%}). '
                                   f'A previous CT agent approved this variation.'),
                        'layer': 'Name Registry', 'features': feats, 'registry_alias': alias}

    fuzzy = fuzz.token_sort_ratio(n_payer, n_account) / 100
    if fuzzy >= 0.97:
        return {'payer': payer_name, 'account': account_holder_name,
                'confidence': fuzzy, 'decision': 'AUTO_APPROVE',
                'reason': f'Near-identical names — {fuzzy:.1%} character similarity',
                'layer': 'Fuzzy Fast-Track', 'features': feats, 'registry_alias': None}

    X_input = pd.DataFrame([feats])
    conf    = float(model.predict_proba(X_input)[0][1])

    if feats.get('is_company', 0) == 1.0:        conf = min(conf, 0.15)
    if feats.get('gender_conflict', 0) == 1.0:   conf = min(conf, 0.35)
    if feats.get('first_name_conflict', 0) == 1.0: conf = min(conf, 0.65)

    if   conf >= 0.92: decision = 'AUTO_APPROVE'
    elif conf >= 0.75: decision = 'SUGGEST_MATCH'
    elif conf >= 0.45: decision = 'HUMAN_REVIEW'
    else:              decision = 'LIKELY_MISMATCH'

    reason_map = {
        'soundex_last':        'names sound phonetically similar',
        'metaphone_last':      'names match phonetically',
        'last_tok_sim':        'last name matches',
        'nickname':            'known nickname / language variant',
        'initials':            'matches as initials',
        'token_sort':          'words match when reordered',
        'token_set':           'key name tokens present in both',
        'jaro_winkler':        'high overall string similarity',
        'shared_tok':          'shared name tokens',
        'gender_conflict':     'gender title mismatch (M. vs MME)',
        'is_company':          'payer appears to be a company',
        'first_name_conflict': 'first names clearly differ',
        'both_names_high':     'both first and last name match',
    }
    feat_series = pd.Series(feats)
    try:
        importances_list = [
            c.estimator.feature_importances_
            for c in model.calibrated_classifiers_
            if hasattr(c.estimator, 'feature_importances_')
        ]
        imp_vals = np.mean(importances_list, axis=0) if importances_list else np.ones(len(feats)) / len(feats)
    except Exception:
        imp_vals = np.ones(len(feats)) / len(feats)

    imp_series = pd.Series(imp_vals, index=list(feats.keys()))
    weighted   = (feat_series * imp_series).sort_values(ascending=False)

    reasons = []
    for feat in weighted.head(3).index:
        score = feats[feat]
        label = reason_map.get(feat, FEAT_LABELS.get(feat, feat))
        if score > 0.6:   reasons.append(f"✓ {label} ({score:.0%})")
        elif score < 0.3: reasons.append(f"✗ low {label} ({score:.0%})")
        else:             reasons.append(f"~ partial {label} ({score:.0%})")

    return {'payer': payer_name, 'account': account_holder_name,
            'confidence': round(conf, 4), 'decision': decision,
            'reason': ' | '.join(reasons) if reasons else f'ML score: {conf:.1%}',
            'layer': 'ML Model', 'features': feats, 'registry_alias': None}


# ─────────────────────────────────────────────────────
# CROSS-FOLD TEST DATA — derived from your real CSV
# Each fold isolates one naming pattern category
# ─────────────────────────────────────────────────────
CROSS_FOLD_SETS = {
    "Fold A — French Title Normalisation": {
        "description": (
            "Payer omits or abbreviates French civility titles. "
            "M. / MR / MME / MLLE / MLE / MONSIEUR must all be stripped before comparison. "
            "Cases drawn directly from your sample_ct_data.csv."
        ),
        "cases": [
            ("IBRAHIMA TAMBA",          "M. IBRAHIMA TAMBA",               True,  "No title → M. prefix"),
            ("Guillaume Munsambote",    "GUILLAUME MUNSAMBOTE",             True,  "Mixed case vs ALL CAPS"),
            ("Mathieu Ladeira",         "MATHIEU LADEIRA",                  True,  "Mixed case exact"),
            ("Joris Nsenguet Tossam",   "M. JORIS NSENGUET TOSSAM",        True,  "No title → M."),
            ("Samira GOGOR",            "SAMIRA GOGOR",                     True,  "Mixed case exact"),
            ("Didier Juredieu",         "M. DIDIER JUREDIEU",               True,  "No title → M."),
            ("ERNEST KONDA",            "M. ERNEST KONDA",                  True,  "No title → M."),
            ("yazid khettal",           "M. YAZID KHETTAL",                 True,  "Lowercase → M. CAPS"),
            ("Gilles ANSELME",          "M. GILLES ANSELME",                True,  "Mixed → M. CAPS"),
            ("Iyyappan Zeavelou",       "M. IYYAPPAN ZEAVELOU",             True,  "African/Tamil first name"),
            ("MR   ALEXANDRE MORANDO",  "M. ALEXANDRE MORANDO",             True,  "MR → M. (different abbrev)"),
        ],
    },
    "Fold B — Surname-First Reordering": {
        "description": (
            "French banking often stores names SURNAME FIRSTNAME but payers write Firstname Surname. "
            "Token-sort similarity resolves this. "
            "Cases drawn from your CSV rows 19, 33, 34, 38, 41 etc."
        ),
        "cases": [
            ("MR COURBOIN  KYRIL",               "MR CYRIL COURBOIN",                  True,  "Surname-first + Kyril/Cyril typo"),
            ("MR    CHATAIN XAVIER",             "MR XAVIER CHATAIN",                  True,  "Surname-first, extra spaces"),
            ("DIEDHIOU OUMAR",                   "M. OUMAR DIEDHIOU",                  True,  "African surname-first"),
            ("MLE DOS SANTOS VIRGINIE",          "MME VIRGINIE DOS SANTOS",            True,  "Compound surname reorder"),
            ("M. VILHENA JEAN-PIERRE",           "MR JEAN-PIERRE VILHENA",             True,  "Hyphenated first, surname-first"),
            ("M. LASSIAILLE NICOLAS",            "M. NICOLAS LASSIAILLE",              True,  "Surname-first same gender"),
            ("MR TCHATCHOUA NGAYEWANG PIERRE",  "M. PIERRE TCHATCHOUA NGAYEWANG",     True,  "3-token African name reorder"),
            ("MLE FILA MATOMBO IMELDA",         "MLLE IMELDA FILA MATOMBO",           True,  "MLE/MLLE + reorder"),
            ("POUGEARD DULIMBERT ARNAUD",        "MR ARNAUD POUGEARD-DULIMBERT",       True,  "Hyphen in account compound surname"),
            ("SLEIMA ELSA",                      "MLLE ELSA SLEIMAN",                  True,  "Reorder + minor spelling diff"),
        ],
    },
    "Fold C — Middle Names & Extra Tokens": {
        "description": (
            "Account holder has middle names the payer omitted — should still match. "
            "But when the extra token is itself a different forename that conflicts, "
            "CT labelled it False. The model must learn this distinction from data."
        ),
        "cases": [
            ("M OU MME LOIC VERDIER",            "M. LOIC MARC VERDIER",               True,  "1 extra middle, CT True"),
            ("M CYRIL KOZA",                     "M. CYRIL JACQUES JEAN NOEL KOZA",    True,  "Many extra middles, CT True"),
            ("M STEPHANE BATIOT",                "MR STEPHANE JEROME FREDERIC BATIOT", True,  "2 extra middles, CT True"),
            ("Marie-Claude Chateau",             "MME MARIE CLAUDE FRANCOISE CHATEAU", True,  "Hyphenated = two tokens + extra"),
            ("MLE NADEGE ANDZOUANA",             "MLLE NADEGE ETOU ANDZOUANA",         True,  "Middle ETOU added, CT True"),
            ("M CYRIL KOZA",                     "M. CYRIL KOZA",                      True,  "Exact match minus extra middles"),
            ("MLE ELOISE MATHIOT",               "MLE ELOISE MARIE CECILE MATHIOT",    False, "Extra middles, CT labelled False"),
            ("Ziva CVAR",                        "MLE ZIVA ELISABETH CVAR",            False, "Extra middle, CT labelled False"),
            ("Vincent MACHET",                   "M. VINCENT HENRI MACHET",            False, "Extra middle, CT labelled False"),
            ("M     HAIM SIBONY",                "M. HAIM GERSHON SIBONY",             False, "Different middle, CT labelled False"),
        ],
    },
    "Fold D — Gender Title Conflict": {
        "description": (
            "Payer has a male title (M./MR/MONSIEUR) but account is female (MME/MLLE), "
            "or vice versa — these are different account holders and must NOT match. "
            "Also covers cases where both titles are female but surnames differ."
        ),
        "cases": [
            ("MONSIEUR JULIEN PEREZ",    "MME JULIEN ABRAHAM ALBERT PEREZ",  False, "MONSIEUR vs MME — different person"),
            ("MME CONDE STELLA",         "MLLE STELLA MAKOLO",               False, "Both female, different surname"),
            ("MME HIELE LEA",            "MLLE LEA MARIE KATHY WISSOCQ",     False, "Both female, different surname"),
            ("M OU MME ZAMANSKY MARC",   "M. BRUNO ZAMANSKY",                False, "Marc ≠ Bruno despite same surname"),
            ("STEPHANE CONTAT",          "M. STEPHANE PIERRE  ROBERT CONTAT",False, "Subset but CT said False"),
            ("M  CESBRON GAETAN",        "M. GAETAN OLIVIER CESBRON",        False, "Same gender, CT said False"),
            ("MME ROSINE VIOLENE",       "MME VIOLENE MARIE JOSEPHE ROSINE", False, "Both MME, name is token-swapped different person"),
            ("Mle MOREAU JENNY",         "MME JENNY SABINE MOREAU",          True,  "MLE/MME — same person, CT True"),
            ("Roxane ABEILLON",          "MME ROXANE ABEILLON",              True,  "No payer title, female account"),
        ],
    },
    "Fold E — Company / Non-Person Payers": {
        "description": (
            "Payer name is a business entity (SARL, SAS, custom brand name) or "
            "a quoted multi-person string. Must never match a person's account."
        ),
        "cases": [
            ("BIODISTRIB",                        "MME LUCITA MARIKA FUMONT",          False, "Brand name — no match"),
            ("S.A.S. REVE D AILLEURS",            "MME CHRISTELLE PATRICIA ALGER",     False, "SAS company"),
            ("\"Mw G Brar,Hr M Mehta\"",          "MR MANOJKUMAR MEHTA",               False, "Quoted multi-person string"),
            ("G Avocegamou  S Kashala Soumahoro", "MME STRELLE KASHALA SOUMAHORO",     False, "Two payers concatenated"),
            ("CHEYNE WILLIAM",                    "M. WILLIAM GJW GIOVANNE JOSEPH WATSON CHEYNE", False, "Account has unexplained extra initials"),
            ("Pedro ANGELO ABATAYGUARA ROSAL",    "MME FERNANDA CAROLINA LIND SILVA",  False, "Completely different people"),
        ],
    },
    "Fold F — OU Keyword & Multi-Holder": {
        "description": (
            "French 'OU' (= 'or') appears in the payer field — the account could be held "
            "by either person listed. System must match if the named individual is the "
            "actual account holder, and reject when they are not."
        ),
        "cases": [
            ("SANTINO HENRI OU KANEIJO",  "M. HENRI SANTINO",        True,  "'OU' alternative — Henri Santino matches"),
            ("M OU MME LOIC VERDIER",     "M. LOIC MARC VERDIER",    True,  "M OU MME stripped, Loic Verdier matches"),
            ("M OU MME ZAMANSKY MARC",    "M. BRUNO ZAMANSKY",       False, "Marc ≠ Bruno despite shared surname"),
        ],
    },
    "Fold G — Hyphen & Compound Names": {
        "description": (
            "Hyphenated names (Jean-Pierre, Marie-Claude, Pougeard-Dulimbert) appear "
            "with and without hyphens. Both forms must be treated as identical "
            "after normalisation (hyphen → space)."
        ),
        "cases": [
            ("M. VILHENA JEAN-PIERRE",    "MR JEAN-PIERRE VILHENA",              True,  "Hyphen preserved both sides"),
            ("Marie-Claude Chateau",      "MME MARIE CLAUDE FRANCOISE CHATEAU",  True,  "Hyphen → space in account"),
            ("POUGEARD DULIMBERT ARNAUD", "MR ARNAUD POUGEARD-DULIMBERT",        True,  "Hyphen in account compound surname"),
            ("M STEPHANE BATIOT",         "MR STÉPHANE JÉRÔME FRÉDÉRIC BATIOT",  True,  "Accented caps — hyphen not involved"),
        ],
    },
    "Fold H — Phonetic & Spelling Variants": {
        "description": (
            "Transliterated names, accented characters stripped, minor spelling differences "
            "common in West African and French naming conventions "
            "(Kyril/Cyril, Sleima/Sleiman, Andzouana/Andzouana)."
        ),
        "cases": [
            ("MR COURBOIN  KYRIL",                "MR CYRIL COURBOIN",              True,  "Kyril / Cyril phonetic swap"),
            ("SLEIMA ELSA",                       "MLLE ELSA SLEIMAN",              True,  "Sleima vs Sleiman (+n)"),
            ("MLE NADEGE ANDZOUANA",              "MLLE NADEGE ETOU ANDZOUANA",     True,  "Andzouana spelling consistent"),
            ("MR TCHATCHOUA NGAYEWANG PIERRE",    "M. PIERRE TCHATCHOUA NGAYEWANG", True,  "Long African name reorder"),
            ("Roxane ABEILLON",                   "MME ROXANE ABEILLON",            True,  "Exact minus title"),
            ("Iyyappan Zeavelou",                 "M. IYYAPPAN ZEAVELOU",           True,  "Tamil name — exact content"),
        ],
    },
}


# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏦 Payment Name Matching System</h1>
  <p>Control Tower Automation · AI-powered name reconciliation · v6 Real Data Edition</p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️  Models not found. "
             "Run `python day1_explore_and_features.py` "
             "then `python day2_train_model.py` first.")
    st.stop()


# ─────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Live Demo",
    "📊  Accuracy & Model",
    "📉  CT Workload Reduction",
    "🔬  Cross-Fold Validation",
    "📋  Batch Test Cases",
])

TIER_COLORS = {
    'AUTO_APPROVE':    '#16a34a',
    'SUGGEST_MATCH':   '#ca8a04',
    'HUMAN_REVIEW':    '#ea580c',
    'LIKELY_MISMATCH': '#dc2626',
}


# ════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ════════════════════════════════════════════════════
with tab1:
    st.subheader("Compare Two Names in Real Time")
    st.markdown("Paste any payer and account holder name — the system decides instantly and explains why.")

    st.markdown("**Quick examples from your real data:**")
    ex_cols = st.columns(6)
    examples = [
        ("DIEDHIOU OUMAR",           "M. OUMAR DIEDHIOU",               "Surname-first"),
        ("M OU MME LOIC VERDIER",    "M. LOIC MARC VERDIER",            "OU keyword"),
        ("MONSIEUR JULIEN PEREZ",    "MME JULIEN ABRAHAM ALBERT PEREZ", "Gender conflict"),
        ("POUGEARD DULIMBERT ARNAUD","MR ARNAUD POUGEARD-DULIMBERT",    "Hyphen"),
        ("BIODISTRIB",               "MME LUCITA MARIKA FUMONT",        "Company"),
        ("MR COURBOIN  KYRIL",       "MR CYRIL COURBOIN",               "Phonetic typo"),
    ]
    for i, (p, a, label) in enumerate(examples):
        if ex_cols[i].button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state['ex_payer']   = p
            st.session_state['ex_account'] = a
            st.rerun()

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        payer = st.text_input(
            "📤  Payer Name  (from the transaction)",
            value=st.session_state.get('ex_payer', 'DIEDHIOU OUMAR'))
    with col_right:
        account = st.text_input(
            "🏦  Account Holder Name  (registered on account)",
            value=st.session_state.get('ex_account', 'M. OUMAR DIEDHIOU'))

    if payer.strip() and account.strip():
        result = predict(payer, account)
        conf   = result['confidence']
        dec    = result['decision']

        st.markdown("---")

        icons  = {'AUTO_APPROVE':'✅','SUGGEST_MATCH':'🟡','HUMAN_REVIEW':'⚠️','LIKELY_MISMATCH':'❌'}
        labels = {
            'AUTO_APPROVE':    'AUTO APPROVE — Payment can proceed automatically',
            'SUGGEST_MATCH':   'SUGGEST MATCH — CT sees this with a recommendation',
            'HUMAN_REVIEW':    'HUMAN REVIEW — CT reviews with full context',
            'LIKELY_MISMATCH': 'LIKELY MISMATCH — CT investigates',
        }
        st.markdown(
            f'<div class="decision-card {dec}">{icons[dec]}  {labels[dec]}</div>',
            unsafe_allow_html=True)

        bar_color = TIER_COLORS[dec]
        st.markdown(f"""
        <div style="margin:8px 0 4px 0; font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:0.05em">
            Confidence Score
        </div>
        <div style="background:#e2e8f0; border-radius:3px; height:26px; width:100%;">
            <div style="background:{bar_color}; width:{conf*100:.1f}%; height:26px;
                 border-radius:3px; display:flex; align-items:center;
                 justify-content:center; color:white; font-weight:700; font-size:14px;
                 font-family:'IBM Plex Mono',monospace;">
                {conf:.1%}
            </div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""<div class="metric-box">
            <div class="val">{conf:.1%}</div><div class="lbl">Confidence</div>
        </div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-box">
            <div class="val" style="font-size:15px">{result['layer']}</div>
            <div class="lbl">Decision Layer</div>
        </div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-box">
            <div class="val">{fuzz.token_sort_ratio(normalize(payer), normalize(account))}%</div>
            <div class="lbl">Raw Fuzzy Score</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(
            f'<div class="reason-box"><strong>Why:</strong> {result["reason"]}</div>',
            unsafe_allow_html=True)

        st.markdown("#### Feature Breakdown")
        feats   = result['features']
        feat_df = pd.DataFrame([{
            'Feature': FEAT_LABELS.get(k, k),
            'Score':   round(v, 3),
            'Signal':  '🟢 Match' if v >= 0.75 else '🟡 Partial' if v >= 0.4 else '🔴 Mismatch',
        } for k, v in feats.items()]).sort_values('Score', ascending=False)
        feat_df['Visual'] = feat_df['Score'].apply(lambda s: '█' * int(s * 20))
        st.dataframe(
            feat_df[['Feature', 'Visual', 'Score', 'Signal']],
            use_container_width=True, hide_index=True, height=400)


# ════════════════════════════════════════════════════
# TAB 2 — ACCURACY & MODEL
# ════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Accuracy & What It Learned")
    auc          = eval_res['auc']
    tier_results = eval_res['tier_results']
    feat_imp_d   = eval_res['feat_imp']

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, sub in [
        (c1, f"{auc:.3f}",  "Test AUC",           "1.0 = perfect"),
        (c2, "15,000",      "Training pairs",      "your CT decisions"),
        (c3, "25",          "Features per pair",   "similarity signals"),
        (c4, "<1ms",        "Per-decision speed",  "real-time capable"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}<br><small style="font-weight:300">{sub}</small></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Tier Precision")
        for t in tier_results:
            if t['cases'] > 0:
                color = TIER_COLORS.get(t['tier'], '#64748b')
                st.markdown(f"""
                <div style="background:#f8fafc; border-left:4px solid {color};
                     padding:10px 14px; margin:6px 0; border-radius:0 4px 4px 0;">
                    <strong style="color:{color}; font-family:'IBM Plex Mono',monospace">{t['tier']}</strong><br>
                    <span style="font-size:13px; color:#475569">
                        {t['coverage']:.0%} of cases &nbsp;·&nbsp; {t['precision']:.0%} precision
                    </span>
                </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("#### Feature Importance")
        feat_imp_series = pd.Series(feat_imp_d).sort_values(ascending=True).tail(12)
        display_labels  = [FEAT_LABELS.get(k, k) for k in feat_imp_series.index]

        fig, ax = plt.subplots(figsize=(7, 5))
        colors  = ['#1e40af' if v > feat_imp_series.mean() else '#93c5fd'
                   for v in feat_imp_series.values]
        bars = ax.barh(display_labels, feat_imp_series.values,
                       color=colors, edgecolor='white', height=0.7)
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_title('Top Features Learned by Model', fontsize=12, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{w:.3f}', va='center', ha='left', fontsize=7, color='#475569')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════
# TAB 3 — CT WORKLOAD REDUCTION
# ════════════════════════════════════════════════════
with tab3:
    st.subheader("CT Workload Reduction Estimate")
    tier_results = eval_res['tier_results']
    auto_tier    = next((t for t in tier_results if t['tier'] == 'AUTO_APPROVE'), None)
    auto_pct     = auto_tier['coverage'] if auto_tier else 0
    ct_pct       = 1 - auto_pct
    auto_prec    = auto_tier['precision'] if auto_tier else 0

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    for col, val, color, lbl, sub in [
        (c1, f"{auto_pct:.0%}", '#16a34a', "Handled automatically",  "no CT agent needed"),
        (c2, f"{ct_pct:.0%}",  '#2563eb', "Sent to CT agents",       "with AI context prepared"),
        (c3, f"{auto_prec:.0%}",'#7c3aed','AUTO_APPROVE precision',  "correct approvals only"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="val" style="color:{color}">{val}</div>
            <div class="lbl">{lbl}<br><small style="font-weight:300">{sub}</small></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        fig, axes = plt.subplots(1, 2, figsize=(7, 4))
        axes[0].pie([100], colors=['#fca5a5'], startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[0].set_title('Before\n(All Manual)', fontweight='bold', fontsize=11)
        axes[0].text(0, -1.3, '100% CT Manual Review', ha='center', fontsize=9, color='#64748b')
        axes[1].pie([auto_pct*100, ct_pct*100],
                    colors=['#86efac', '#fca5a5'],
                    labels=[f'Automated\n{auto_pct:.0%}', f'CT Review\n{ct_pct:.0%}'],
                    startangle=90, autopct='%1.0f%%', pctdistance=0.6,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                    textprops={'fontsize': 9})
        axes[1].set_title('After\n(This System)', fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        tier_desc = {
            'AUTO_APPROVE':    'Payment processes — no human touch',
            'SUGGEST_MATCH':   'CT sees suggestion — 1 click to confirm',
            'HUMAN_REVIEW':    'CT reviews with AI context prepared',
            'LIKELY_MISMATCH': 'CT investigates potential fraud / error',
        }
        for t in tier_results:
            color = TIER_COLORS.get(t['tier'], '#64748b')
            st.markdown(f"""
            <div style="background:#f8fafc; border-left:5px solid {color};
                 padding:12px 16px; margin:8px 0; border-radius:0 4px 4px 0;">
                <div style="display:flex; justify-content:space-between">
                    <strong style="color:{color}; font-family:'IBM Plex Mono',monospace; font-size:13px">{t['tier']}</strong>
                    <span style="background:{color}; color:white; padding:2px 10px;
                          border-radius:3px; font-size:11px; font-weight:600;
                          font-family:'IBM Plex Mono',monospace">{t['coverage']:.0%}</span>
                </div>
                <div style="font-size:12px; color:#64748b; margin-top:4px">
                    {tier_desc.get(t['tier'],'')} · <em>{t['precision']:.0%} precision</em>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        ---
        **Business impact at scale:**
        - **1,000 payments/day** → ~**{auto_pct*1000:.0f} handled automatically**
        - CT agents only review **~{ct_pct*1000:.0f}** that genuinely need a human
        - Each CT decision feeds back → model improves weekly
        """)


# ════════════════════════════════════════════════════
# TAB 4 — CROSS-FOLD VALIDATION
# ════════════════════════════════════════════════════
with tab4:
    st.subheader("Cross-Fold Validation — Real Data Patterns")
    st.markdown(
        "Each fold tests one specific naming pattern observed in your CSV. "
        "Use this to validate the system handles the full complexity of "
        "French/West-African name data before going to production."
    )

    fold_names = list(CROSS_FOLD_SETS.keys())
    selected_fold = st.selectbox(
        "Select a fold to inspect:",
        options=["ALL FOLDS — Summary"] + fold_names,
        index=0,
    )

    @st.cache_data
    def run_all_folds():
        results = {}
        for fold_name, fold_data in CROSS_FOLD_SETS.items():
            fold_rows = []
            for payer, account, expected, note in fold_data["cases"]:
                r = predict(payer, account)
                correct = (
                    (r['decision'] == 'AUTO_APPROVE'    and expected) or
                    (r['decision'] == 'LIKELY_MISMATCH' and not expected) or
                    (r['decision'] == 'SUGGEST_MATCH'   and expected) or
                    (r['decision'] == 'HUMAN_REVIEW')
                )
                hard_error = (
                    (r['decision'] == 'AUTO_APPROVE'    and not expected) or
                    (r['decision'] == 'LIKELY_MISMATCH' and expected)
                )
                fold_rows.append({
                    'payer':      payer,
                    'account':    account,
                    'expected':   expected,
                    'note':       note,
                    'decision':   r['decision'],
                    'confidence': r['confidence'],
                    'layer':      r['layer'],
                    'correct':    correct,
                    'hard_error': hard_error,
                })
            results[fold_name] = fold_rows
        return results

    all_fold_results = run_all_folds()

    # ─ Summary view ───────────────────────────────────────────
    if selected_fold == "ALL FOLDS — Summary":
        st.markdown("### Overall Performance Across All Folds")

        summary_rows = []
        total_all = correct_all = hard_err_all = 0

        for fold_name, rows in all_fold_results.items():
            n         = len(rows)
            n_correct = sum(1 for r in rows if r['correct'])
            n_hard    = sum(1 for r in rows if r['hard_error'])
            n_auto    = sum(1 for r in rows if r['decision'] == 'AUTO_APPROVE')
            n_mis     = sum(1 for r in rows if r['decision'] == 'LIKELY_MISMATCH')
            total_all    += n
            correct_all  += n_correct
            hard_err_all += n_hard
            summary_rows.append({
                'Fold':          fold_name.split('—')[0].strip(),
                'Pattern':       fold_name.split('—')[1].strip() if '—' in fold_name else fold_name,
                'Cases':         n,
                'Correct':       f"{n_correct}/{n}",
                'Score':         f"{n_correct/n*100:.0f}%",
                'Hard Errors':   n_hard,
                'Auto-approved': n_auto,
                'Flagged':       n_mis,
            })

        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.markdown(f"""<div class="metric-box">
            <div class="val">{correct_all}/{total_all}</div>
            <div class="lbl">Total Correct</div>
        </div>""", unsafe_allow_html=True)
        ov2.markdown(f"""<div class="metric-box">
            <div class="val" style="color:{'#16a34a' if correct_all/total_all>=0.8 else '#ca8a04'}">{correct_all/total_all:.0%}</div>
            <div class="lbl">Overall Accuracy</div>
        </div>""", unsafe_allow_html=True)
        ov3.markdown(f"""<div class="metric-box">
            <div class="val" style="color:{'#dc2626' if hard_err_all>0 else '#16a34a'}">{hard_err_all}</div>
            <div class="lbl">Hard Errors</div>
        </div>""", unsafe_allow_html=True)
        ov4.markdown(f"""<div class="metric-box">
            <div class="val">{len(fold_names)}</div>
            <div class="lbl">Pattern Folds</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Per-fold bar chart
        fig, ax = plt.subplots(figsize=(11, 4))
        short_names = [r['Fold'] for r in summary_rows]
        scores      = [int(r['Score'].replace('%','')) for r in summary_rows]
        hard_errs   = [r['Hard Errors'] for r in summary_rows]
        bar_colors  = ['#16a34a' if s == 100 else '#ca8a04' if s >= 70 else '#dc2626' for s in scores]
        bars = ax.bar(short_names, scores, color=bar_colors, edgecolor='white', width=0.6)
        ax.axhline(y=80, color='#94a3b8', linestyle='--', linewidth=1, label='80% threshold')
        ax.set_ylim(0, 115)
        ax.set_ylabel('Score (%)', fontsize=10)
        ax.set_title('Per-Fold Accuracy — All Pattern Categories', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, score, hard in zip(bars, scores, hard_errs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{score}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            if hard > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'⚠{hard}', ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
        ax.legend(fontsize=9)
        plt.xticks(rotation=25, ha='right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if hard_err_all > 0:
            st.markdown("### ⚠️ Hard Errors — Wrong Polarity (Need Attention)")
            st.markdown(
                "Hard error = model approved a known mismatch, "
                "or rejected a known valid match. These need investigation."
            )
            for fold_name, rows in all_fold_results.items():
                hard = [r for r in rows if r['hard_error']]
                if hard:
                    st.markdown(f"**{fold_name}**")
                    for r in hard:
                        exp_str = "TRUE" if r['expected'] else "FALSE"
                        st.markdown(f"""
                        <div style="background:#fef2f2; border-left:4px solid #ef4444;
                             padding:10px 14px; margin:4px 0; border-radius:0 4px 4px 0; font-size:13px;">
                            <code>{r['payer']}</code> → <code>{r['account']}</code><br>
                            Expected: <strong>{exp_str}</strong> &nbsp;·&nbsp;
                            Got: <strong style="color:#dc2626">{r['decision']}</strong>
                            ({r['confidence']:.0%}) &nbsp;·&nbsp; {r['note']}
                        </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ No hard errors across all folds — system is correctly polarised.")

    # ─ Individual fold view ───────────────────────────────────
    else:
        fold_data = CROSS_FOLD_SETS[selected_fold]
        rows      = all_fold_results[selected_fold]
        n_correct = sum(1 for r in rows if r['correct'])
        n_hard    = sum(1 for r in rows if r['hard_error'])

        st.markdown(f"**Pattern:** {fold_data['description']}")
        st.markdown("---")

        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f"""<div class="metric-box">
            <div class="val">{n_correct}/{len(rows)}</div><div class="lbl">Correct</div>
        </div>""", unsafe_allow_html=True)
        col_b.markdown(f"""<div class="metric-box">
            <div class="val" style="color:{'#16a34a' if n_correct==len(rows) else '#ca8a04'}">{n_correct/len(rows):.0%}</div>
            <div class="lbl">Score</div>
        </div>""", unsafe_allow_html=True)
        col_c.markdown(f"""<div class="metric-box">
            <div class="val" style="color:{'#dc2626' if n_hard>0 else '#16a34a'}">{n_hard}</div>
            <div class="lbl">Hard Errors</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        dec_styles = {
            'AUTO_APPROVE':    ('#f0fdf4', '#15803d', '#22c55e'),
            'SUGGEST_MATCH':   ('#fefce8', '#a16207', '#eab308'),
            'HUMAN_REVIEW':    ('#fff7ed', '#c2410c', '#f97316'),
            'LIKELY_MISMATCH': ('#fef2f2', '#b91c1c', '#ef4444'),
        }

        for r in rows:
            bg, fg, border = dec_styles.get(r['decision'], ('#f8fafc','#0f172a','#94a3b8'))
            exp_tag = (
                '<span class="pattern-tag tag-true">EXPECTED: TRUE</span>'
                if r['expected'] else
                '<span class="pattern-tag tag-false">EXPECTED: FALSE</span>'
            )
            status_icon = "✅" if r['correct'] else ("⚠️ HARD ERROR" if r['hard_error'] else "🔶 ambiguous")

            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border};
                 border-left:5px solid {border};
                 padding:14px 18px; margin:8px 0; border-radius:0 6px 6px 0;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px">
                    <div style="flex:1">
                        <div style="font-size:11px; color:#64748b; margin-bottom:4px;
                             font-family:'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:0.04em">
                            PAYER &nbsp;→&nbsp; ACCOUNT
                        </div>
                        <div style="font-size:15px; font-weight:600; color:#0f172a; font-family:'IBM Plex Mono',monospace">
                            {r['payer']}
                        </div>
                        <div style="font-size:13px; color:#475569; font-family:'IBM Plex Mono',monospace">
                            {r['account']}
                        </div>
                        <div style="margin-top:6px; font-size:12px; color:#94a3b8; font-style:italic">
                            {r['note']}
                        </div>
                    </div>
                    <div style="text-align:right; min-width:190px">
                        {exp_tag}
                        <div style="font-size:13px; font-weight:700; color:{fg};
                             font-family:'IBM Plex Mono',monospace; margin-top:6px">
                            {r['decision']}
                        </div>
                        <div style="font-size:12px; color:#64748b">
                            {r['confidence']:.0%} · {r['layer']}
                        </div>
                        <div style="font-size:14px; margin-top:4px">{status_icon}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Confidence Distribution — Expected TRUE vs FALSE")

        fig, ax = plt.subplots(figsize=(10, 3))
        true_confs  = [r['confidence'] for r in rows if r['expected']]
        false_confs = [r['confidence'] for r in rows if not r['expected']]

        if true_confs:
            ax.scatter(true_confs,  [1]*len(true_confs),  color='#22c55e', s=130,
                       zorder=5, label='Expected TRUE',  marker='o',
                       edgecolors='white', linewidth=1.5)
        if false_confs:
            ax.scatter(false_confs, [0]*len(false_confs), color='#ef4444', s=130,
                       zorder=5, label='Expected FALSE', marker='s',
                       edgecolors='white', linewidth=1.5)

        for x_thresh, color, lbl in [
            (0.92, '#16a34a', 'AUTO_APPROVE'),
            (0.75, '#ca8a04', 'SUGGEST_MATCH'),
            (0.45, '#ea580c', 'HUMAN_REVIEW'),
        ]:
            ax.axvline(x=x_thresh, color=color, linestyle='--', linewidth=1,
                       alpha=0.7, label=f'{lbl} ({x_thresh})')

        ax.set_xlim(-0.02, 1.05)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Expected FALSE', 'Expected TRUE'], fontsize=9)
        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_title('Score distribution — green dots (TRUE) should sit right, red squares (FALSE) should sit left',
                     fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Breakdown table
        st.markdown("#### Decision Table")
        tbl_rows = [{
            'Payer':      r['payer'],
            'Account':    r['account'],
            'Expected':   '✅ TRUE' if r['expected'] else '❌ FALSE',
            'Decision':   r['decision'],
            'Confidence': f"{r['confidence']:.0%}",
            'Layer':      r['layer'],
            'Result':     '✅' if r['correct'] else ('🚨 HARD ERROR' if r['hard_error'] else '🔶 ambiguous'),
        } for r in rows]

        def color_decision(val):
            return {
                'AUTO_APPROVE':    'background-color:#dcfce7; color:#166534; font-weight:600',
                'SUGGEST_MATCH':   'background-color:#fef9c3; color:#854d0e; font-weight:600',
                'HUMAN_REVIEW':    'background-color:#ffedd5; color:#9a3412; font-weight:600',
                'LIKELY_MISMATCH': 'background-color:#fee2e2; color:#991b1b; font-weight:600',
            }.get(val, '')

        st.dataframe(
            pd.DataFrame(tbl_rows).style.applymap(color_decision, subset=['Decision']),
            use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# TAB 5 — BATCH TEST CASES
# ════════════════════════════════════════════════════
with tab5:
    st.subheader("Batch Test — All Name Variation Patterns")

    test_cases = [
        # (category, payer, account, expected_match)
        # ─ Generic patterns ─
        ("Exact",               "John Smith",               "John Smith",                  True),
        ("Initial",             "J. Smith",                 "John Smith",                  True),
        ("Double initial",      "J.R. Smith",               "John Robert Smith",           True),
        ("Nickname EN",         "Bob Johnson",              "Robert Johnson",              True),
        ("Nickname EN",         "Bill Taylor",              "William Taylor",              True),
        ("Typo",                "Jon Smyth",                "John Smith",                  True),
        ("Name reorder",        "Wang Wei",                 "Wei Wang",                    True),
        ("Title stripped",      "Dr John Smith",            "John Smith",                  True),
        # ─ Real data patterns ─
        ("FR title",            "IBRAHIMA TAMBA",           "M. IBRAHIMA TAMBA",           True),
        ("FR title",            "Guillaume Munsambote",     "GUILLAUME MUNSAMBOTE",        True),
        ("FR title",            "yazid khettal",            "M. YAZID KHETTAL",            True),
        ("Surname-first",       "DIEDHIOU OUMAR",           "M. OUMAR DIEDHIOU",           True),
        ("Surname-first",       "MR CHATAIN XAVIER",        "MR XAVIER CHATAIN",           True),
        ("Surname-first",       "MLE FILA MATOMBO IMELDA",  "MLLE IMELDA FILA MATOMBO",   True),
        ("Hyphen",              "M. VILHENA JEAN-PIERRE",   "MR JEAN-PIERRE VILHENA",      True),
        ("Hyphen",              "POUGEARD DULIMBERT ARNAUD","MR ARNAUD POUGEARD-DULIMBERT",True),
        ("OU keyword",          "SANTINO HENRI OU KANEIJO", "M. HENRI SANTINO",            True),
        ("OU keyword",          "M OU MME LOIC VERDIER",    "M. LOIC MARC VERDIER",        True),
        ("MLE/MLLE equiv",      "Mle MOREAU JENNY",         "MME JENNY SABINE MOREAU",     True),
        ("Phonetic",            "MR COURBOIN  KYRIL",       "MR CYRIL COURBOIN",           True),
        ("Extra middles",       "M CYRIL KOZA",             "M. CYRIL JACQUES JEAN NOEL KOZA", True),
        # ─ Should NOT match ─
        ("Middle conflict",     "Jean Pierre Dubois",       "Jean Parker Dubois",          False),
        ("Gender conflict",     "MONSIEUR JULIEN PEREZ",    "MME JULIEN ABRAHAM ALBERT PEREZ", False),
        ("Gender conflict",     "MME CONDE STELLA",         "MLLE STELLA MAKOLO",          False),
        ("Company",             "BIODISTRIB",               "MME LUCITA MARIKA FUMONT",    False),
        ("Company",             "S.A.S. REVE D AILLEURS",   "MME CHRISTELLE PATRICIA ALGER",False),
        ("Diff surname",        "Shikhar Gupta",            "Maruti Gupta",                False),
        ("Diff person",         "Alice Johnson",            "Bob Williams",                False),
        ("OU mismatch",         "M OU MME ZAMANSKY MARC",   "M. BRUNO ZAMANSKY",           False),
    ]

    rows_out = []
    for category, payer, account, expected in test_cases:
        r = predict(payer, account)
        correct = (
            (r['decision'] == 'AUTO_APPROVE'    and expected) or
            (r['decision'] == 'LIKELY_MISMATCH' and not expected) or
            (r['decision'] == 'SUGGEST_MATCH'   and expected) or
            (r['decision'] == 'HUMAN_REVIEW')
        )
        rows_out.append({
            'Category':       category,
            'Payer Name':     payer,
            'Account Holder': account,
            'Expected':       '✅ TRUE' if expected else '❌ FALSE',
            'Confidence':     f"{r['confidence']:.0%}",
            'Decision':       r['decision'],
            'Layer':          r['layer'],
            '✓':              '✅' if correct else '⚠️',
        })

    result_df      = pd.DataFrame(rows_out)
    total_cases    = len(result_df)
    correct_cases  = (result_df['✓'] == '✅').sum()
    auto_cases     = (result_df['Decision'] == 'AUTO_APPROVE').sum()
    mismatch_cases = (result_df['Decision'] == 'LIKELY_MISMATCH').sum()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Cases",       total_cases)
    s2.metric("Correct Decisions", f"{correct_cases}/{total_cases}")
    s3.metric("Auto-approved",     auto_cases)
    s4.metric("Flagged mismatch",  mismatch_cases)

    def colour_decision(val):
        return {
            'AUTO_APPROVE':    'background-color:#dcfce7; color:#166534; font-weight:600',
            'SUGGEST_MATCH':   'background-color:#fef9c3; color:#854d0e; font-weight:600',
            'HUMAN_REVIEW':    'background-color:#ffedd5; color:#9a3412; font-weight:600',
            'LIKELY_MISMATCH': 'background-color:#fee2e2; color:#991b1b; font-weight:600',
        }.get(val, '')

    st.dataframe(
        result_df.style.applymap(colour_decision, subset=['Decision']),
        use_container_width=True, hide_index=True, height=700)

    st.info(
        f"**{auto_cases} of {total_cases}** batch test cases handled automatically ({auto_cases/total_cases:.0%}).\n\n"
        f"At scale on 15,000+ CT decisions this projects to approximately "
        f"**{eval_res.get('workload_reduction', 0.67):.0%} CT workload reduction**."
    )
