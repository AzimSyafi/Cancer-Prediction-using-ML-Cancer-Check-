"""
Cancer Check: Streamlit app for the FYP "Cancer Prediction Using Machine Learning".

Pages: Check a patient (step-by-step questions, result, reasons, what-if, PDF),
Upload CSV (cancer type detected from the columns), Model results, About.
Themes: Light, Dark and Frutiger Aero (see theme.py).

Run locally from the FYP2 folder, after 03_train.py has created models/ and results/:
    c:\\python313\\python.exe -m streamlit run app.py
"""

import os
import re
import json
import math
import html
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as pgo
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so theme.py and report.py are always found
import theme
import report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
RANDOM_STATE = 42          # must match 03_train.py so the test split is identical
MAX_UPLOAD_MB = 5
MAX_ROWS = 5000

st.set_page_config(page_title="Cancer Check", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------
CANCERS = {
    "lung": {"name": "Lung cancer", "short": "lung", "tab": "Lung",
             "desc": "Simple yes or no questions about habits and symptoms.", "time": "About 1 minute"},
    "breast": {"name": "Breast cancer", "short": "breast", "tab": "Breast",
               "desc": "Cell measurements from a lab report of a breast tissue sample.", "time": "For clinicians"},
    "cervical": {"name": "Cervical cancer: lifestyle only", "short": "cervical", "tab": "Cervical: lifestyle only",
                 "desc": "Age, lifestyle and health history. No test results needed, but results are weak.",
                 "time": "About 2 minutes"},
    "cervical_screening": {"name": "Cervical cancer: with screening", "short": "screening",
                           "tab": "Cervical: with screening",
                           "desc": "Lifestyle questions plus three screening test results from the clinic.",
                           "time": "After screening"},
}
MODELS = {"ann": "ANN", "svm": "SVM", "random_forest": "Random Forest"}
NAV = {"Check a patient": ":material/stethoscope:", "Upload CSV": ":material/upload_file:",
       "Model results": ":material/bar_chart:", "About": ":material/info:"}

LABELS = {
    "GENDER": "Gender", "AGE": "Age (years)", "SMOKING": "Smokes", "YELLOW_FINGERS": "Yellow fingers",
    "ANXIETY": "Anxiety", "PEER_PRESSURE": "Peer pressure", "CHRONIC_DISEASE": "Long-term illness",
    "FATIGUE": "Tiredness", "ALLERGY": "Allergies", "WHEEZING": "Wheezing",
    "ALCOHOL_CONSUMING": "Drinks alcohol", "COUGHING": "Coughing",
    "SHORTNESS_OF_BREATH": "Shortness of breath", "SWALLOWING_DIFFICULTY": "Difficulty swallowing",
    "CHEST_PAIN": "Chest pain",
    "Age": "Age (years)", "Number_of_sexual_partners": "Number of sexual partners",
    "First_sexual_intercourse": "Age at first sexual intercourse", "Num_of_pregnancies": "Number of pregnancies",
    "Smokes": "Smokes", "Smokes_(years)": "Years smoking", "Smokes_(packs/year)": "Packs smoked per year",
    "Hormonal_Contraceptives": "Uses hormonal contraceptives",
    "Hormonal_Contraceptives_(years)": "Years on hormonal contraceptives",
    "IUD": "Has an IUD", "IUD_(years)": "Years with an IUD",
    "STDs": "Has had an STD", "STDs_(number)": "Number of STDs",
    "STDs:condylomatosis": "Genital warts (condylomatosis)",
    "STDs:cervical_condylomatosis": "Cervical condylomatosis",
    "STDs:vaginal_condylomatosis": "Vaginal condylomatosis",
    "STDs:vulvo-perineal_condylomatosis": "Vulvo-perineal condylomatosis",
    "STDs:syphilis": "Syphilis", "STDs:pelvic_inflammatory_disease": "Pelvic inflammatory disease",
    "STDs:genital_herpes": "Genital herpes", "STDs:molluscum_contagiosum": "Molluscum contagiosum",
    "STDs:AIDS": "AIDS", "STDs:HIV": "HIV", "STDs:Hepatitis_B": "Hepatitis B", "STDs:HPV": "HPV infection",
    "STDs:_Number_of_diagnosis": "Number of STD diagnoses",
    "STDs_not_answered": "Skipped the STD questions",
    "Hinselmann": "Hinselmann test abnormal", "Schiller": "Schiller test abnormal",
    "Citology": "Pap smear (cytology) abnormal",
}
HINTS = {
    "SMOKING": "Currently smokes cigarettes", "ALCOHOL_CONSUMING": "Drinks alcohol regularly",
    "PEER_PRESSURE": "Feels pushed by friends to smoke or drink",
    "CHRONIC_DISEASE": "Has a chronic disease such as diabetes", "ALLERGY": "Has any allergy",
    "ANXIETY": "Often feels anxious or worried", "FATIGUE": "Often feels very tired",
    "COUGHING": "A cough that keeps coming back", "WHEEZING": "A whistling sound when breathing",
    "SHORTNESS_OF_BREATH": "Gets out of breath easily", "CHEST_PAIN": "Pain or tightness in the chest",
    "SWALLOWING_DIFFICULTY": "Food feels stuck when swallowing", "YELLOW_FINGERS": "Yellow stains on the fingers",
    "Hinselmann": "Colposcopy screening test", "Schiller": "Iodine stain test of the cervix",
    "Citology": "Pap smear cell test", "STDs_not_answered": "Choose Yes if the patient did not answer the STD questions",
}
GUIDANCE = {
    "lung": {
        "high": [("See a doctor soon", "Especially if a cough, chest pain or breathing problem has lasted more than three weeks."),
                 ("Ask about a chest scan", "A doctor may suggest a chest X-ray or a low-dose CT scan to look more closely.")],
        "low": [("Keep an eye on symptoms", "These answers show few warning signs. See a doctor if a symptom appears or does not go away."),
                ("Ask about screening", "People who smoke or smoked heavily can ask a doctor whether lung screening suits them.")],
    },
    "breast": {
        "high": [("Specialist review", "These cell measurements look more like the malignant samples in the data."),
                 ("Discuss the sample", "The result should be reviewed by a pathologist and discussed with a breast specialist.")],
        "low": [("Likely benign pattern", "These cell measurements look more like the benign samples in the data."),
                ("Follow-up checks", "Follow the specialist's advice on follow-up checks.")],
    },
    "cervical": {
        "high": [("Book cervical screening", "Book a Pap smear or HPV test if it is due or overdue."),
                 ("Treat this as a prompt", "These models are weak, so this is a reason to get screened, not an answer.")],
        "low": [("Keep up with screening", "Risk factors alone cannot rule out cervical cancer, so routine screening still matters."),
                ("Prevention", "HPV vaccination and regular screening are the best ways to prevent cervical cancer.")],
    },
    "cervical_screening": {
        "high": [("Follow up abnormal results", "A doctor may recommend a colposcopy or a biopsy."),
                 ("Do not wait for symptoms", "Early follow-up makes treatment far more effective.")],
        "low": [("Results look reassuring", "They look like those of patients whose biopsy was negative."),
                ("Routine screening", "Continue screening at the interval your doctor recommends.")],
    },
}
TIPS = {
    "SMOKING": ("Stop smoking", "It is the most effective way to lower lung cancer risk, at any age.", "Because: smokes"),
    "ALCOHOL_CONSUMING": ("Cut down on alcohol", "It lowers cancer risk and helps overall health.", "Because: drinks alcohol"),
    "Smokes": ("Stop smoking", "Smoking is linked to cervical cancer, so quitting lowers the risk.", "Because: smokes"),
    "STDs:HPV": ("Follow up on HPV", "HPV causes almost all cervical cancers, so follow-up screening matters.", "Because: HPV"),
}
LEVELS = {3: ("high", "High concern", "All three models think this patient is likely to have {c}."),
          2: ("mid", "Some concern", "Two of the three models think this patient is likely to have {c}."),
          1: ("mid", "Low to moderate concern", "Only one of the three models flags this patient, so the models disagree."),
          0: ("low", "Low concern", "None of the three models think this patient is likely to have {c}.")}
LABEL_COLUMNS = {"lung": ["lungcancer", "target"], "breast": ["diagnosis", "target"],
                 "cervical": ["biopsy", "target"], "cervical_screening": ["biopsy", "target"]}
ID_COLUMNS = ["patientid", "patient", "id", "patientno", "patientnumber"]

ICON = {
    "lung": '<path d="M3 8h11a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h7"></path>',
    "breast": '<path d="M12 3a4 4 0 0 1 4 4c0 3-4 6-4 6s-4-3-4-6a4 4 0 0 1 4-4z"></path><path d="M10 13l-4 8M14 13l4 8"></path>',
    "cervical": '<circle cx="12" cy="8" r="4"></circle><path d="M5 21a7 7 0 0 1 14 0"></path>',
    "cervical_screening": '<circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.5-4.5M11 8v6M8 11h6"></path>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"></path>',
    "info": '<circle cx="12" cy="12" r="9"></circle><path d="M12 11v6M12 7.5v.5"></path>',
    "warn": '<path d="M12 3l10 18H2z"></path><path d="M12 10v5M12 18v.5"></path>',
    "check": '<path d="M20 6L9 17l-5-5"></path>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V7a4 4 0 0 1 8 0v4"></path>',
    "bulb": '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.8.8 1 1.5 1 2.5h6c0-1 .2-1.7 1-2.5A6 6 0 0 0 12 3z"></path>',
    "shield": '<path d="M12 3l8 4v5c0 5-3.5 8-8 9-4.5-1-8-4-8-9V7z"></path>',
    "pulse": '<path d="M3 12h4l3-8 4 16 3-8h4"></path>',
}

ss = st.session_state
for key, default in {"view": "home", "cancer": None, "step": 0, "answers": {}, "actual": {},
                     "nav": "Check a patient"}.items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def H(markup):
    """Render HTML. Lines are joined so Markdown never treats indents as code blocks."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


_card_count = [0]


def card():
    """A bordered container with a stable key, so theme.py can style it as a card."""
    _card_count[0] += 1
    return st.container(border=True, key=f"card_{_card_count[0]}")


def svg(name, size=20, colour="currentColor", width=2):
    return (f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{colour}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


def box(kind, text, icon="info"):
    H(f'<div class="cc-box {kind}">{svg(icon, 18)}<span>{text}</span></div>')


def hero(title, text=""):
    H(f'<div class="cc-hero"><h1>{title}</h1>{f"<p>{text}</p>" if text else ""}</div>')


def label(name):
    if name in LABELS:
        return LABELS[name]
    for suffix, word in [("_mean", "average"), ("_se", "variation"), ("_worst", "largest")]:
        if name.endswith(suffix):
            return f"{name[:-len(suffix)].replace('_', ' ').capitalize()} ({word})"
    return " ".join(name.replace("_", " ").split()).capitalize()


def bin_opts(name):
    return ["Female", "Male"] if name == "GENDER" else ["No", "Yes"]


def is_whole(f):
    return all(float(v).is_integer() for v in (f["min"], f["max"], f["median"]))


def fmt(f, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not given"
    if f["type"] == "binary":
        return bin_opts(f["name"])[int(round(v))]
    return f"{v:g}" if is_whole(f) else f"{v:.3g}"


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def level_of(votes):
    return LEVELS[int(votes)]


# ---------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------
@st.cache_data
def load_meta(cancer):
    with open(os.path.join(MODEL_DIR, f"{cancer}_meta.json")) as f:
        return json.load(f)


@st.cache_resource
def load_models(cancer):
    return {label_: joblib.load(os.path.join(MODEL_DIR, f"{cancer}_{key}.joblib"))
            for key, label_ in MODELS.items()}


@st.cache_data
def load_results():
    return (pd.read_csv(os.path.join(RESULTS_DIR, "test_results.csv")),
            pd.read_csv(os.path.join(RESULTS_DIR, "cv_results.csv")))


@st.cache_data
def load_clean(cancer):
    return pd.read_csv(os.path.join(CLEAN_DIR, f"{cancer}_clean.csv"))


@st.cache_data
def load_test(cancer):
    """The exact held-out test split from 03_train.py. The models never saw these patients."""
    df = load_clean(cancer)
    X, y = df.drop(columns="target"), df["target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return X_test.assign(target=y_test)


def available():
    return [c for c in CANCERS if os.path.exists(os.path.join(MODEL_DIR, f"{c}_meta.json"))]


def predict_all(models, X):
    return {n: (m.predict(X).astype(int), m.predict_proba(X)[:, 1]) for n, m in models.items()}


def model_scores(cancer):
    test_df, cv_df = load_results()
    return test_df[test_df.Cancer == cancer], cv_df[cv_df.Cancer == cancer]


@st.cache_data
def load_nested():
    """Nested CV summary from 05_extended_eval.py, or None if it has not been run."""
    path = os.path.join(RESULTS_DIR, "extended", "nested_cv_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def reliability(cancer):
    """Badge from the best nested-CV MCC of the three app models.

    MCC is used instead of F1 because F1 ignores correct "no cancer" answers:
    on lung, a model that always says "cancer" already scores F1 0.93.
    Falls back to CV F1 if the extended results are missing.
    """
    nested = load_nested()
    if nested is not None:
        rows = nested[(nested.Dataset == cancer) & nested.Model.isin(["ANN", "SVM", "Random Forest"])]
        if len(rows):
            best = float(rows["MCC mean"].max())
            if best >= 0.85:
                return "green", f"Very reliable · MCC {best:.2f}", best
            if best >= 0.6:
                return "green", f"Reliable · MCC {best:.2f}", best
            if best >= 0.3:
                return "blue", f"Moderate · MCC {best:.2f}", best
            return "amber", f"Experimental · MCC {best:.2f}", best
    _, cvc = model_scores(cancer)
    best = cvc["f1_mean"].max() if len(cvc) else 0
    if best >= 0.95:
        return "green", f"Very reliable · F1 {best:.2f}", best
    if best >= 0.85:
        return "green", f"Reliable · F1 {best:.2f}", best
    if best >= 0.6:
        return "blue", f"Moderate · F1 {best:.2f}", best
    return "amber", f"Experimental · F1 {best:.2f}", best


# ---------------------------------------------------------------------
# Question groups
# ---------------------------------------------------------------------
def build_groups(cancer, feats):
    names = [f["name"] for f in feats]
    if cancer == "lung":
        spec = [("About the patient", "Basic details used by all three models.", ["GENDER", "AGE"]),
                ("Lifestyle", "Habits that can affect lung health.", ["SMOKING", "ALCOHOL_CONSUMING", "PEER_PRESSURE"]),
                ("General health", "Long-term conditions and how the patient feels day to day.",
                 ["CHRONIC_DISEASE", "ALLERGY", "ANXIETY", "FATIGUE"]),
                ("Symptoms", "Signs the patient has noticed recently.",
                 ["COUGHING", "WHEEZING", "SHORTNESS_OF_BREATH", "CHEST_PAIN", "SWALLOWING_DIFFICULTY", "YELLOW_FINGERS"])]
    elif cancer == "breast":
        spec = [("Average values", "The mean of each measurement across the cells in the sample.",
                 [n for n in names if n.endswith("_mean")]),
                ("Variation", "How much each measurement varies between cells (standard error).",
                 [n for n in names if n.endswith("_se")]),
                ("Largest values", "The largest (worst) value found in the sample.",
                 [n for n in names if n.endswith("_worst")])]
    else:
        spec = [("About the patient", "Age and pregnancy history.",
                 ["Age", "Number_of_sexual_partners", "First_sexual_intercourse", "Num_of_pregnancies"]),
                ("Smoking", "Smoking habits, now or in the past.", ["Smokes", "Smokes_(years)", "Smokes_(packs/year)"]),
                ("Contraception", "Hormonal contraceptives and IUD use.",
                 ["Hormonal_Contraceptives", "Hormonal_Contraceptives_(years)", "IUD", "IUD_(years)"]),
                ("Sexual health", "Answer only what the patient is comfortable sharing.",
                 [n for n in names if n.startswith("STDs")])]
        if cancer == "cervical_screening":
            spec.append(("Screening tests", "Results from the clinic. Yes means the result was abnormal.",
                         ["Hinselmann", "Schiller", "Citology"]))
    by_name = {f["name"]: f for f in feats}
    used, groups = set(), []
    for title, hint, cols in spec:
        group = [by_name[c] for c in cols if c in by_name and c not in used]
        used.update(f["name"] for f in group)
        if group:
            groups.append((title, hint, group))
    rest = [f for f in feats if f["name"] not in used]
    if rest:
        groups.append(("Other details", "", rest))
    return groups


def bounds(f):
    lo, hi = float(f["min"]), float(f["max"])
    span = max(hi - lo, 1e-6)
    if f["name"].lower() == "age":
        return 1, 120, 1
    if is_whole(f):
        return (0 if lo >= 0 else int(lo - span)), int(hi + max(10, span)), 1
    low = 0.0 if lo >= 0 else lo - 2 * span
    return low, hi + 2 * span, float(f"{span / 100:.2g}")


# ---------------------------------------------------------------------
# Answers state
# ---------------------------------------------------------------------
def fresh_answers(meta):
    out = {}
    for f in meta["features"]:
        if f["type"] == "binary":
            out[f["name"]] = None
        else:
            out[f["name"]] = int(round(f["median"])) if is_whole(f) else float(f["median"])
    return out


def clear_widgets(prefix):
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith(prefix)]:
        del ss[k]


def start_check(cancer):
    ss.cancer, ss.view, ss.step = cancer, "questions", 0
    if cancer not in ss.answers:
        ss.answers[cancer] = fresh_answers(load_meta(cancer))


def load_test_patient(cancer):
    meta = load_meta(cancer)
    row = load_test(cancer).sample(1).iloc[0]
    ans = {}
    for f in meta["features"]:
        v = row[f["name"]]
        if pd.isna(v):
            v = f["median"]
        if f["type"] == "binary":
            ans[f["name"]] = int(round(v))
        else:
            lo, hi, _ = bounds(f)
            v = min(max(float(v), lo), hi)
            ans[f["name"]] = int(round(v)) if is_whole(f) else float(v)
    ss.answers[cancer] = ans
    ss.actual[cancer] = int(row["target"])
    ss.step = len(build_groups(cancer, meta["features"])) - 1
    clear_widgets(f"w_{cancer}_")
    clear_widgets(f"wi_{cancer}_")


def reset_answers(cancer):
    ss.answers[cancer] = fresh_answers(load_meta(cancer))
    ss.actual.pop(cancer, None)
    ss.step = 0
    clear_widgets(f"w_{cancer}_")


def sync(cancer, f, key):
    v = ss.get(key)
    if f["type"] == "binary":
        ss.answers[cancer][f["name"]] = None if v is None else bin_opts(f["name"]).index(v)
    else:
        ss.answers[cancer][f["name"]] = v
    ss.actual.pop(cancer, None)
    clear_widgets(f"wi_{cancer}_")


def patient_frame(cancer, meta, answers=None):
    answers = answers or ss.answers[cancer]
    row = {}
    for f in meta["features"]:
        v = answers.get(f["name"])
        if v is None:
            v = round(f["median"]) if f["name"] == "GENDER" else (0 if f["type"] == "binary" else f["median"])
        row[f["name"]] = float(v)
    return pd.DataFrame([row])


def go(view):
    ss.view = view


def go_page(page):
    ss.nav = page


def jump_step(cancer, labels):
    v = ss.get(f"stepnav_{cancer}")
    if v in labels:
        ss.step = labels.index(v)


# ---------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------
def explain(models, X, meta):
    """Swap each input for a typical (median) value and measure how much the
    models' average confidence changes. Positive = that input raised concern."""
    feats = meta["features"]
    def avg(df):
        return np.mean([m.predict_proba(df)[:, 1] for m in models.values()], axis=0)
    base = avg(X)[0]
    batch = pd.concat([X] * len(feats), ignore_index=True)
    for i, f in enumerate(feats):
        batch.loc[i, f["name"]] = f["median"]
    impact = (base - avg(batch)) * 100
    rows = [{"f": f, "label": label(f["name"]), "value": fmt(f, X.iloc[0][f["name"]]), "impact": imp}
            for f, imp in zip(feats, impact)]
    rows = [r for r in rows if abs(r["impact"]) >= 0.5]
    return sorted(rows, key=lambda r: -abs(r["impact"]))[:6]


def why_html(drivers):
    top = max(abs(d["impact"]) for d in drivers)
    out = []
    for d in drivers:
        w = abs(d["impact"]) / top * 100
        pts = max(1, round(abs(d["impact"])))
        word = f"{pts} point{'s' if pts != 1 else ''}"
        if d["impact"] > 0:
            track = f'<div class="neg"></div><div class="pos"><div class="up" style="width:{w:.0f}%"></div></div>'
            val = f'<span class="v up-t">+{word}</span>'
        else:
            track = f'<div class="neg"><div class="down" style="width:{w:.0f}%"></div></div><div class="pos"></div>'
            val = f'<span class="v down-t">−{word}</span>'
        out.append(f'<div class="cc-why"><span><b>{html.escape(d["label"])}:</b> {html.escape(d["value"])}</span>'
                   f'<div class="track">{track}</div>{val}</div>')
    ups = [d["label"].lower() for d in drivers if d["impact"] > 0][:2]
    downs = [d["label"].lower() for d in drivers if d["impact"] < 0][:1]
    summary = []
    if ups:
        summary.append(f"<b>{' and '.join(ups)}</b> pushed the result up the most.")
    if downs:
        summary.append(f"{downs[0].capitalize()} lowered it.")
    return "".join(out), " ".join(summary)


def gauge(pct, level):
    frac = min(max(pct, 0.01), 0.99)
    angle = math.pi * (1 - frac)
    x, y = 110 + 90 * math.cos(angle), 110 - 90 * math.sin(angle)
    colour = {"high": "var(--red)", "mid": "#D97706", "low": "var(--green)"}[level]
    return (f'<svg role="img" aria-label="Average model confidence {pct:.0%}" width="250" height="148" viewBox="0 0 220 130">'
            f'<path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" style="stroke:var(--track)" stroke-width="18" stroke-linecap="round"></path>'
            f'<path d="M 20 110 A 90 90 0 0 1 {x:.1f} {y:.1f}" fill="none" style="stroke:{colour}" stroke-width="18" stroke-linecap="round"></path>'
            f'<text x="110" y="96" text-anchor="middle" style="fill:var(--ink);font-family:var(--head)" font-size="34" font-weight="700">{pct:.0%}</text>'
            f'<text x="110" y="116" text-anchor="middle" style="fill:var(--muted)" font-size="11">average confidence</text>'
            f'<text x="20" y="128" text-anchor="middle" style="fill:var(--muted)" font-size="10">Low</text>'
            f'<text x="200" y="128" text-anchor="middle" style="fill:var(--muted)" font-size="10">High</text></svg>')


# ---------------------------------------------------------------------
# Page: Check a patient
# ---------------------------------------------------------------------
def steps_html(active):
    names = ["Choose a cancer", "Answer questions", "See the result"]
    parts = []
    for i, n in enumerate(names):
        if i:
            parts.append('<div class="cc-line"></div>')
        parts.append(f'<div class="cc-step{" on" if i == active else ""}"><b>{i + 1}</b>{n}</div>')
    return f'<div class="cc-steps">{"".join(parts)}</div>'


def page_home():
    H(steps_html(0))
    hero("Which cancer would you like to check?",
         "Three AI models (ANN, SVM and Random Forest) look at the answers and each give their opinion. "
         "You will also see why they decided and what to do next.")
    st.write("")
    cancers = available()
    cols = st.columns(2, gap="medium")
    for i, c in enumerate(cancers):
        info = CANCERS[c]
        meta = load_meta(c)
        n = len(meta["features"])
        colour, badge, _ = reliability(c)
        needs = f"{n} lab values" if c == "breast" else f"{n} questions"
        with cols[i % 2]:
            with card():
                H(f'<div class="cc-card-head"><div class="cc-icon">{svg(c, 22)}</div>'
                  f'<span class="cc-badge {colour}">{badge}</span></div>'
                  f'<p class="cc-title">{info["name"]}</p><p class="cc-text">{info["desc"]}</p>'
                  f'<div class="cc-meta"><span>{needs}</span><span>{info["time"]}</span></div>')
                st.button(f"Start {info['short']} check", key=f"start_{c}", type="primary",
                          on_click=start_check, args=(c,), width="stretch")
    with card():
        a, b = st.columns([3, 1], vertical_alignment="center")
        with a:
            H(f'<div style="display:flex;gap:14px;align-items:center"><div class="cc-icon">{svg("upload", 22)}</div>'
              f'<div><p class="cc-title" style="font-size:16px;margin:0">Have many patients? Upload a CSV file</p>'
              f'<p class="cc-text">Check a whole list at once, then download the results.</p></div></div>')
        with b:
            st.button("Upload CSV", on_click=go_page, args=("Upload CSV",), width="stretch")
    box("plain", "The badge on each card shows MCC, a score of how well the models tell apart patients with and without cancer on patients they did not train on. 0 means guessing and 1.00 is perfect.")


def render_input(cancer, f):
    key = f"w_{cancer}_{f['name']}"
    value = ss.answers[cancer].get(f["name"])
    left, right = st.columns([1.35, 1], vertical_alignment="center")
    hint = HINTS.get(f["name"], "")
    if f["type"] != "binary" and f["name"] != "GENDER":
        hint = hint or f"Usual range in the data: {fmt(f, f['min'])} to {fmt(f, f['max'])}"
    with left:
        H(f'<div class="cc-q"><div class="l">{label(f["name"])}</div><div class="s">{html.escape(hint)}</div></div>')
    with right:
        if f["type"] == "binary":
            opts = bin_opts(f["name"])
            if key not in ss:
                ss[key] = None if value is None else opts[int(value)]
            st.segmented_control(label(f["name"]), opts, key=key, on_change=sync, args=(cancer, f, key),
                                 label_visibility="collapsed", width="stretch")
        else:
            lo, hi, step = bounds(f)
            if key not in ss:
                v = min(max(value, lo), hi)
                ss[key] = int(v) if is_whole(f) else float(v)
            fmt_str = None if is_whole(f) else ("%.4f" if abs(f["max"]) < 10 else "%.2f")
            st.number_input(label(f["name"]), min_value=lo, max_value=hi, step=step, key=key, format=fmt_str,
                            on_change=sync, args=(cancer, f, key), label_visibility="collapsed")
            v = ss.answers[cancer].get(f["name"])
            if v is not None and (v < f["min"] or v > f["max"]):
                box("warn", f"{fmt(f, v)} is outside the patients the models learned from "
                            f"({fmt(f, f['min'])} to {fmt(f, f['max'])}), so the result may be less reliable.", "warn")


def page_questions(cancer):
    meta = load_meta(cancer)
    groups = build_groups(cancer, meta["features"])
    ss.step = min(ss.step, len(groups) - 1)
    H(steps_html(1))
    top_l, top_r = st.columns([3, 1.3], vertical_alignment="bottom")
    with top_l:
        st.button("Back to cancer list", icon=":material/arrow_back:", type="tertiary", on_click=go, args=("home",))
        hero(f"{CANCERS[cancer]['name']} check")
    with top_r:
        st.button("Load a test patient", icon=":material/casino:", on_click=load_test_patient, args=(cancer,),
                  width="stretch", help="Loads a real patient from the test set, which the models never saw.")

    labels_ = [f"{i + 1}. {g[0]}" for i, g in enumerate(groups)]
    ss[f"stepnav_{cancer}"] = labels_[ss.step]
    st.segmented_control("Step", labels_, key=f"stepnav_{cancer}", on_change=jump_step,
                         args=(cancer, labels_), label_visibility="collapsed")
    pct = (ss.step + 1) / len(groups) * 100
    H(f'<div class="cc-progress"><div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div>'
      f'<span>Step {ss.step + 1} of {len(groups)}</span></div>')

    if cancer == "cervical":
        box("warn", "These models are weak (no better than guessing in testing), because lifestyle answers alone "
                    "predict cervical cancer poorly. Treat the result as experimental.", "warn")
    if ss.actual.get(cancer) is not None:
        box("info", "A test patient is loaded. Change any answer and the real result will no longer apply.", "info")

    title, hint, feats = groups[ss.step]
    main, side = st.columns([2.1, 1], gap="medium")
    with main:
        with card():
            H(f'<p class="cc-title">{title}</p><p class="cc-text">{hint}</p>')
            for f in feats:
                render_input(cancer, f)
            st.write("")
            b1, _, b2 = st.columns([1, 1.5, 1.3])
            with b1:
                if ss.step > 0:
                    st.button("Back", key="q_back", width="stretch",
                              on_click=lambda: ss.update(step=ss.step - 1))
            with b2:
                if ss.step < len(groups) - 1:
                    st.button("Next", key="q_next", type="primary", width="stretch",
                              on_click=lambda: ss.update(step=ss.step + 1))
                else:
                    st.button("See the result", key="q_result", type="primary", width="stretch",
                              on_click=go, args=("result",))
    with side:
        with card():
            rows = []
            for i, (gt, _, gf) in enumerate(groups):
                bins = [f for f in gf if f["type"] == "binary"]
                if bins:
                    done = sum(ss.answers[cancer].get(f["name"]) is not None for f in bins)
                    status = f"{done} of {len(bins)} answered"
                else:
                    status = "Filled in"
                rows.append(f'<div class="cc-sum"><span>{gt}</span><span>{status}</span></div>')
            H(f'<p class="cc-title" style="font-size:15px">Answers so far</p>{"".join(rows)}')
        box("info", "Not sure about a question? Leave it. Unanswered yes or no questions count as No.", "bulb")
        st.button("Clear all answers", icon=":material/restart_alt:", type="tertiary",
                  on_click=reset_answers, args=(cancer,))


def page_result(cancer):
    meta, models = load_meta(cancer), load_models(cancer)
    test_df, _ = model_scores(cancer)
    X = patient_frame(cancer, meta)
    preds = predict_all(models, X)
    votes = int(sum(p[0][0] for p in preds.values()))
    avg = float(np.mean([p[1][0] for p in preds.values()]))
    level, title, text = level_of(votes)
    text = text.format(c=CANCERS[cancer]["name"].split(":")[0].lower())
    guidance_level = "high" if votes >= 2 else "low"
    actual = ss.actual.get(cancer)
    drivers = explain(models, X, meta)

    answers = ss.answers[cancer]
    who = []
    if actual is not None:
        who.append("Test patient")
    if "GENDER" in answers and answers["GENDER"] is not None:
        who.append(bin_opts("GENDER")[answers["GENDER"]])
    for k in ("AGE", "Age"):
        if k in answers:
            who.append(f"age {fmt({'min': 0, 'max': 0, 'median': 0, 'type': 'numeric', 'name': k}, answers[k])}")
    patient_line = " · ".join(who) or "Patient details as entered"

    H(steps_html(2))
    top_l, top_r = st.columns([2.2, 2], vertical_alignment="bottom")
    with top_l:
        st.button("Change the answers", icon=":material/arrow_back:", type="tertiary", on_click=go, args=("questions",))
        hero(f"{CANCERS[cancer]['name']} result", patient_line)
    with top_r:
        r1, r2 = st.columns(2)
        pdf = report.patient_pdf(
            CANCERS[cancer]["name"], patient_line, level, title, text, avg,
            [[n, "Cancer likely" if p[0][0] else "Cancer unlikely", f"{p[1][0]:.0%}"] for n, p in preds.items()],
            [f"{d['label']}: {d['value']} {'raised' if d['impact'] > 0 else 'lowered'} confidence by "
             f"{max(1, round(abs(d['impact'])))} points" for d in drivers[:4]],
            [(label(f["name"]), fmt(f, X.iloc[0][f["name"]])) for f in meta["features"]],
            [f"{a}: {b}" for a, b in GUIDANCE[cancer][guidance_level]])
        r1.download_button("Summary (PDF)", pdf, file_name=f"{cancer}_summary.pdf", mime="application/pdf",
                           icon=":material/download:", width="stretch")
        r2.button("Check another", type="primary", width="stretch", on_click=go, args=("home",))

    # Verdict
    with card():
        real = ""
        if actual is not None:
            correct = sum(int(p[0][0] == actual) for p in preds.values())
            real = (f'<div class="cc-box plain">{svg("check", 18)}<span>Real result for this test patient: '
                    f'<b>{"cancer" if actual else "no cancer"}</b>. {correct} of 3 models were correct.</span></div>')
        badge = {"high": "red", "mid": "amber", "low": "green"}[level]
        H(f'<div class="cc-hero-result"><div>{gauge(avg, level)}</div><div class="txt">'
          f'<span class="cc-badge {badge}" style="align-self:flex-start;font-size:13px;padding:6px 12px">'
          f'{svg("warn" if level != "low" else "check", 15)}{title}</span>'
          f'<h2>{text}</h2><p class="cc-text">Confidence shows how sure the models are. It is not the '
          f'real-world chance of having cancer.</p>{real}</div></div>')

    # Each model
    H('<h3 style="margin:18px 0 0 0">What each model says</h3>')
    cards = []
    colours = theme.TOKENS[current_theme()]["models"]
    for n, (p, prob) in preds.items():
        row = test_df[test_df.Model == n]
        f1 = f"Test score {row['F1'].iloc[0]:.2f}" if len(row) else ""
        bar_col = "var(--red)" if p[0] else "var(--green)"
        cards.append(f'<div class="cc-mcard"><div class="top">{n}<span>{f1}</span></div>'
                     f'<div class="res {"yes" if p[0] else "no"}">{"Cancer likely" if p[0] else "Cancer unlikely"}</div>'
                     f'<div class="cc-mbar"><div style="width:{prob[0] * 100:.0f}%;background:{bar_col}"></div></div>'
                     f'<div class="conf">{prob[0]:.0%} confident</div></div>')
    H(f'<div class="cc-grid3">{"".join(cards)}</div>')

    # Why
    with card():
        H('<p class="cc-title">Why the models think this</p><p class="cc-text">Compared with a typical patient '
          'in the data. Bars to the right raised concern, bars to the left lowered it.</p>')
        if drivers:
            rows, summary = why_html(drivers)
            H(rows)
            if summary:
                box("plain", f"In short: {summary}", "bulb")
        else:
            box("plain", "This patient looks very similar to a typical patient in the data, so no single answer stands out.")

    # What if
    with card():
        H('<p class="cc-title">What if something changes?</p><p class="cc-text">Change an answer and the models '
          'check again straight away. Nothing is saved.</p>')
        items = [d["f"] for d in drivers[:5]] or meta["features"][:4]
        left, right = st.columns([1.6, 1], gap="medium")
        changed = dict(answers)
        with left:
            for f in items:
                key = f"wi_{cancer}_{f['name']}"
                cur = X.iloc[0][f["name"]]
                a, b = st.columns([1.2, 1], vertical_alignment="center")
                a.markdown(f"**{label(f['name'])}**  \n<span style='font-size:12px;color:var(--muted)'>Patient's answer: "
                           f"{fmt(f, cur)}</span>", unsafe_allow_html=True)
                with b:
                    if f["type"] == "binary":
                        opts = bin_opts(f["name"])
                        if key not in ss:
                            ss[key] = opts[int(round(cur))]
                        v = st.segmented_control(label(f["name"]), opts, key=key, label_visibility="collapsed",
                                                 width="stretch")
                        changed[f["name"]] = int(round(cur)) if v is None else opts.index(v)
                    else:
                        lo, hi, step = bounds(f)
                        if key not in ss:
                            ss[key] = int(cur) if is_whole(f) else float(cur)
                        v = st.number_input(label(f["name"]), min_value=lo, max_value=hi, step=step, key=key,
                                            label_visibility="collapsed",
                                            format=None if is_whole(f) else "%.4f")
                        changed[f["name"]] = v
        X2 = patient_frame(cancer, meta, changed)
        preds2 = predict_all(models, X2)
        votes2 = int(sum(p[0][0] for p in preds2.values()))
        avg2 = float(np.mean([p[1][0] for p in preds2.values()]))
        lvl2, title2, _ = level_of(votes2)
        delta = round((avg2 - avg) * 100)
        dtext = "no change" if delta == 0 else f"{'+' if delta > 0 else '−'}{abs(delta)} points"
        badge2 = {"high": "red", "mid": "amber", "low": "green"}[lvl2]
        with right:
            H(f'<div class="cc-panel {lvl2}"><span class="small"><b>New average confidence</b></span>'
              f'<span class="big">{avg2:.0%}</span><span class="small">Was {avg:.0%} · {dtext}</span>'
              f'<span class="small">{votes2} of 3 models flag cancer</span>'
              f'<span class="cc-badge {badge2}" style="align-self:flex-start">{title2}</span></div>')
            st.button("Reset to patient's answers", width="stretch",
                      on_click=clear_widgets, args=(f"wi_{cancer}_",))

    # Next steps
    H('<h3 style="margin:18px 0 0 0">What to do next</h3>')
    steps = [(t, d, "") for t, d in GUIDANCE[cancer][guidance_level]]
    for feat, (t, d, why) in TIPS.items():
        if feat in answers and (answers.get(feat) or 0) >= 1 and all(t != s[0] for s in steps):
            steps.append((t, d, why))
    cards = []
    for i, (t, d, why) in enumerate(steps, 1):
        tag = f'<span class="cc-badge green">{why}</span>' if why else ""
        cards.append(f'<div class="cc-mcard cc-next"><div style="display:flex;justify-content:space-between;'
                     f'align-items:center"><span class="cc-num">{i}</span>{tag}</div><h3>{t}</h3><p>{d}</p></div>')
    H(f'<div class="cc-grid3">{"".join(cards)}</div>')
    H('<p class="cc-foot">This is a student research tool. It cannot diagnose cancer. '
      'Always follow advice from a qualified doctor.</p>')


# ---------------------------------------------------------------------
# Page: Upload CSV
# ---------------------------------------------------------------------
@st.cache_data
def template_csv(cancer):
    meta = load_meta(cancer)
    feats = [f for f in meta["features"] if f["name"] != "STDs_not_answered"]
    sample = load_test(cancer).head(5)
    rows = []
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        row = {"patient_id": f"P{i:03d}"}
        for f in feats:
            v = r[f["name"]]
            if pd.isna(v):
                row[f["name"]] = ""
            elif f["type"] == "binary":
                row[f["name"]] = bin_opts(f["name"])[int(round(v))]
            else:
                row[f["name"]] = v
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def detect(df):
    cols = {norm(c) for c in df.columns}
    scores = {}
    for c in available():
        feats = [f["name"] for f in load_meta(c)["features"] if f["name"] != "STDs_not_answered"]
        matched = sum(norm(f) in cols for f in feats)
        scores[c] = (matched / max(len(feats), 1), matched, len(feats))
    best = max(scores, key=lambda c: (scores[c][0], scores[c][1]))
    return best, scores


def parse_binary(series, name):
    text = series.astype(str).str.strip().str.lower()
    present = series.notna() & (text != "") & (text != "nan")
    if name == "GENDER":
        values = text.map({"m": 1, "male": 1, "1": 1, "1.0": 1, "f": 0, "female": 0, "0": 0, "0.0": 0})
    else:
        nums = pd.to_numeric(series, errors="coerce")
        found = set(nums.dropna().unique())
        if found and found <= {1, 2} and 2 in found:      # original survey coding: 1 = No, 2 = Yes
            values = nums - 1
        else:
            yes = {"yes", "y", "true", "t", "1", "1.0"}
            no = {"no", "n", "false", "f", "0", "0.0"}
            values = text.map(lambda x: 1 if x in yes else (0 if x in no else np.nan))
    return values.astype(float), present & values.isna(), ~present


def parse_number(series):
    text = series.astype(str).str.strip()
    present = series.notna() & (text != "") & (text.str.lower() != "nan")
    values = pd.to_numeric(series, errors="coerce")
    return values.astype(float), present & values.isna(), ~present


def prepare(df, cancer):
    meta = load_meta(cancer)
    colmap = {norm(c): c for c in df.columns}
    X = pd.DataFrame(index=df.index)
    errors, filled, missing_cols = {}, {}, []
    for f in meta["features"]:
        col = colmap.get(norm(f["name"]))
        if col is None:
            if f["name"] == "STDs_not_answered" and norm("STDs") in colmap:
                X[f["name"]] = df[colmap[norm("STDs")]].isna().astype(float)
            else:
                missing_cols.append(label(f["name"]))
                X[f["name"]] = np.nan
            continue
        if f["type"] == "binary":
            values, bad, empty = parse_binary(df[col], f["name"])
            hint = "Use Yes or No."
        else:
            values, bad, empty = parse_number(df[col])
            hint = "Use a number."
        for i in df.index[bad]:
            errors.setdefault(i, []).append(f'{label(f["name"])} says "{html.escape(str(df.at[i, col]))}". {hint}')
        for i in df.index[empty]:
            filled.setdefault(i, []).append(label(f["name"]))
        X[f["name"]] = values

    id_col = next((colmap[k] for k in ID_COLUMNS if k in colmap), None)
    ids = df[id_col].astype(str) if id_col else pd.Series([f"Row {i + 1}" for i in range(len(df))], index=df.index)

    actual = None
    for k in LABEL_COLUMNS[cancer]:
        if k in colmap:
            t = df[colmap[k]].astype(str).str.strip().str.lower()
            actual = t.map({"yes": 1, "1": 1, "1.0": 1, "m": 1, "malignant": 1, "true": 1,
                            "no": 0, "0": 0, "0.0": 0, "b": 0, "benign": 0, "false": 0})
            break
    ok = [i for i in df.index if i not in errors]
    return X.loc[ok], ids, actual, errors, filled, missing_cols, ok


def page_upload():
    hero("Check many patients at once",
         "Upload a CSV file with one patient per row. The app works out which cancer it is for, "
         "then all three models check every patient.")
    st.write("")
    main, side = st.columns([2, 1], gap="medium")
    with side:
        with card():
            H('<p class="cc-title" style="font-size:15px">Need the right format?</p>'
              '<p class="cc-text">Start from a template so the column names match. Each one includes '
              '5 example patients, so it also works as a demo file.</p>')
            for c in available():
                st.download_button(f"{CANCERS[c]['tab']} template", template_csv(c), file_name=f"{c}_template.csv",
                                   mime="text/csv", icon=":material/download:", key=f"tpl_{c}", width="stretch")
            H('<p class="cc-text" style="margin-top:8px">Yes or No answers can be written as YES/NO or 1/0. '
              'Extra columns such as a patient ID are kept. A column with the real result '
              '(for example LUNG_CANCER, diagnosis or Biopsy) is used to check the models.</p>')
        box("plain", "Your file is only used for this check. It is never saved or shared.", "lock")

    with main:
        with card():
            H('<p class="cc-title" style="font-size:15px">1. Upload the file</p>')
            file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed",
                                    help=f"CSV only, up to {MAX_UPLOAD_MB} MB. The cancer type is detected from the column names.")
            if file is None:
                H(f'<p class="cc-text">CSV only, up to {MAX_UPLOAD_MB} MB. The cancer type is detected from the column names.</p>')
                return
            if file.size > MAX_UPLOAD_MB * 1024 * 1024:
                box("bad", f"This file is larger than {MAX_UPLOAD_MB} MB. Split it into smaller files.", "warn")
                return
            try:
                df = pd.read_csv(file, na_values=["?", "", " "], keep_default_na=True)
            except Exception:
                box("bad", "This file could not be read as a CSV. Save it as CSV (comma separated) and try again.", "warn")
                return
            if df.empty:
                box("bad", "The file has no rows.", "warn")
                return
            if len(df) > MAX_ROWS:
                box("warn", f"Only the first {MAX_ROWS} rows are checked.", "warn")
                df = df.head(MAX_ROWS)
            df = df.reset_index(drop=True)

            best, scores = detect(df)
            share, matched, total = scores[best]
            cancers = available()
            pick = st.selectbox("Cancer type", cancers, index=cancers.index(best),
                                format_func=lambda c: CANCERS[c]["name"], key=f"type_{file.name}_{file.size}",
                                help="Detected from the column names. Change it if it is wrong.")
            if pick == best and share >= 0.8:
                extra = " plus the three screening tests" if best == "cervical_screening" else ""
                box("info", f"<b>Detected: {CANCERS[best]['name'].lower()}.</b> {matched} of {total} expected "
                            f"columns were found{extra}.", "check")
            elif pick == best:
                box("warn", f"Only {matched} of {total} columns match any cancer type. Check the column names "
                            "against a template, or choose the type above.", "warn")
            else:
                box("plain", "You chose this type yourself.", "info")

    with st.spinner(f"Checking {len(df)} patients…"):
        X, ids, actual, errors, filled, missing_cols, ok = prepare(df, pick)
        models = load_models(pick)
        preds = predict_all(models, X) if len(X) else {}

    with card():
        H(f'<p class="cc-title" style="font-size:15px">2. File check</p>')
        chips = f'<span class="cc-badge green">{len(ok)} ready</span>'
        if errors:
            chips += f' <span class="cc-badge amber">{len(errors)} need fixing</span>'
        H(f'<div style="display:flex;gap:8px;margin:4px 0 8px 0">{chips}</div>')
        if missing_cols:
            box("warn", "These columns are missing, so typical values were used: "
                        + ", ".join(missing_cols[:8]) + ("…" if len(missing_cols) > 8 else ""), "warn")
        if errors:
            lines = [f"<b>Row {i + 1}:</b> " + " ".join(m) for i, m in list(errors.items())[:6]]
            more = f"<br>And {len(errors) - 6} more." if len(errors) > 6 else ""
            box("warn", "<br>".join(lines) + more + "<br>These rows are skipped. Fix them and upload again to include them.", "warn")
        n_filled = sum(1 for i in ok if i in filled)
        if n_filled:
            who = "1 patient" if n_filled == 1 else f"{n_filled} patients"
            box("plain", f"{who} had empty answers, which were filled with typical values.", "info")
    if not len(X):
        return

    votes = sum(p[0] for p in preds.values())
    conf = np.mean([p[1] for p in preds.values()], axis=0)
    res = pd.DataFrame({"Patient": ids.loc[ok].values, "votes": votes, "conf": conf}, index=ok)
    res["level"] = [level_of(v)[0] for v in votes]
    res["Concern"] = [level_of(v)[1] for v in votes]
    for n, (p, prob) in preds.items():
        res[f"{n} prediction"] = np.where(p == 1, "Cancer likely", "Cancer unlikely")
        res[f"{n} confidence"] = np.round(prob, 3)
    acc_line = None
    if actual is not None and actual.loc[ok].notna().any():
        res["Real result"] = actual.loc[ok].map({1: "Cancer", 0: "No cancer"})
        mask = actual.loc[ok].notna().values
        parts = [f"{n} {np.mean(p[mask] == actual.loc[ok].values[mask]):.0%}" for n, (p, _) in preds.items()]
        acc_line = f"Using the real results in your file, the models were correct for: {', '.join(parts)} of patients."
    res = res.sort_values(["votes", "conf"], ascending=False)

    counts = {"high": int((res.votes == 3).sum()), "mid": int((res.votes == 2).sum()), "low": int((res.votes <= 1).sum())}
    with card():
        a, b, c3 = st.columns([2, 1, 1], vertical_alignment="center")
        a.markdown(f"**3. Results for {len(res)} patients**")
        out = df.loc[res.index].copy()
        for col in [x for x in res.columns if x.endswith("prediction") or x.endswith("confidence")]:
            out[col] = res[col]
        out["Models saying cancer"] = res.votes
        out["Concern"] = res.Concern
        b.download_button("Results (CSV)", out.to_csv(index=False).encode("utf-8"),
                          file_name=f"{pick}_results.csv", mime="text/csv", type="primary",
                          icon=":material/download:", width="stretch")
        pdf_rows = [[str(r.Patient), f"{r.votes} of 3", f"{r.conf:.0%}", r.Concern] for r in res.head(60).itertuples()]
        c3.download_button("Report (PDF)", report.batch_pdf(CANCERS[pick]["name"], file.name, counts,
                                                            pdf_rows, len(res), acc_line),
                           file_name=f"{pick}_report.pdf", mime="application/pdf",
                           icon=":material/picture_as_pdf:", width="stretch")
        H(f'<div class="cc-tiles"><div class="cc-tile high"><span class="k">High concern</span><span class="n">{counts["high"]}</span></div>'
          f'<div class="cc-tile mid"><span class="k">Some concern</span><span class="n">{counts["mid"]}</span></div>'
          f'<div class="cc-tile low"><span class="k">Low concern</span><span class="n">{counts["low"]}</span></div></div>')
        if acc_line:
            box("info", acc_line, "check")
        show = st.segmented_control("Show", ["All", "High", "Some", "Low"], default="All", key="batch_filter")
        view = res
        if show == "High":
            view = res[res.votes == 3]
        elif show == "Some":
            view = res[res.votes == 2]
        elif show == "Low":
            view = res[res.votes <= 1]
        extra_cols = [k for k in ("AGE", "Age", "GENDER") if k in X.columns]
        head = "".join(f"<th>{h}</th>" for h in
                       ["Patient"] + [label(k).replace(" (years)", "") for k in extra_cols]
                       + ["Models saying cancer", "Avg confidence", "Concern"]
                       + (["Real result"] if "Real result" in res else []))
        body = []
        badge = {"high": "red", "mid": "amber", "low": "green"}
        for idx, r in view.head(200).iterrows():
            cells = [f"<td><b>{html.escape(str(r.Patient))}</b></td>"]
            for k in extra_cols:
                f = next(f for f in load_meta(pick)["features"] if f["name"] == k)
                cells.append(f"<td>{fmt(f, X.at[idx, k])}</td>")
            cells += [f"<td>{r.votes} of 3</td>", f'<td class="dim">{r.conf:.0%}</td>',
                      f'<td><span class="cc-badge {badge[r.level]}">{r.Concern}</span></td>']
            if "Real result" in res:
                cells.append(f"<td>{r['Real result'] if pd.notna(r['Real result']) else '-'}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        H(f'<div class="cc-table-wrap"><table class="cc-table"><thead><tr>{head}</tr></thead>'
          f'<tbody>{"".join(body)}</tbody></table></div>')
        st.caption(f"Showing {min(len(view), 200)} of {len(view)} patients, sorted by concern. "
                   "The CSV download has every patient and each model's answer.")


# ---------------------------------------------------------------------
# Page: Model results
# ---------------------------------------------------------------------
def data_note(cancer, n, pos):
    share = pos / n if n else 0
    notes = {
        "breast": "This file is much larger than the original 569-patient study, so it may be an expanded copy. "
                  "Scores may look better than they would on new patients.",
        "cervical": "Lifestyle answers alone carry little information about cervical cancer, so all three models "
                    "score low. This is a limit of the data, not of the methods.",
        "cervical_screening": "The three screening tests are close to a diagnosis already, so this model helps "
                              "after screening, not for early warning.",
    }
    if share > 0.7:
        return (f"Most patients in this data have cancer ({share:.0%}), so always guessing \"cancer\" would "
                f"already be right {share:.0%} of the time.")
    if share < 0.15:
        return (f"Only {pos} of {n} patients have cancer, so scores move a lot between tests. "
                + notes.get(cancer, ""))
    return notes.get(cancer)


def page_results():
    hero("How well do the models work?",
         "Each model was tested 10 times on different groups of patients. These are the average scores.")
    cancers = available()
    cancer = st.segmented_control("Dataset", cancers, format_func=lambda c: CANCERS[c]["tab"],
                                  default=cancers[0], key="res_cancer", label_visibility="collapsed") or cancers[0]
    tst, cvc = model_scores(cancer)
    if cvc.empty:
        box("warn", "No results found for this dataset. Run 03_train.py first.", "warn")
        return
    clean = load_clean(cancer)
    n, pos = len(clean), int(clean["target"].sum())

    ranked = cvc.sort_values("f1_mean", ascending=False)
    best, second = ranked.iloc[0], ranked.iloc[1]
    strength = "Strong" if best.f1_mean >= 0.85 else "Moderate" if best.f1_mean >= 0.6 else "Weak"
    if best.f1_mean - second.f1_mean < max(best.f1_std, second.f1_std):
        verdict = (f"{strength} results, and the three models are about equal. {best.Model} is slightly ahead, "
                   "but the gap is smaller than the normal ups and downs between the 10 tests.")
    else:
        verdict = f"{strength} results. {best.Model} performs best on this data."
    box("info", f"<b>{verdict}</b>", "check")

    steady = cvc.sort_values("f1_std").iloc[0]
    H(f'<div class="cc-tiles"><div class="cc-tile"><span class="k">Highest score</span><span class="n">{best.f1_mean:.2f}</span>'
      f'<span class="d">{best.Model}</span></div><div class="cc-tile"><span class="k">Most consistent</span>'
      f'<span class="n">± {steady.f1_std:.2f}</span><span class="d">{steady.Model}</span></div>'
      f'<div class="cc-tile"><span class="k">Patients in the data</span><span class="n">{n:,}</span>'
      f'<span class="d">{pos:,} with cancer, {n - pos:,} without</span></div></div>')

    tok = theme.TOKENS[current_theme()]
    with card():
        left, right = st.columns([1.5, 1], gap="medium")
        with left:
            H('<p class="cc-title" style="font-size:17px">Overall score (F1)</p>')
            order = [m for m in MODELS.values() if m in set(cvc.Model)]
            sub = cvc.set_index("Model").loc[order]
            fig = pgo.Figure(pgo.Bar(
                x=order, y=sub.f1_mean, marker_color=[tok["models"][m] for m in order],
                error_y=dict(type="data", array=sub.f1_std, color=tok["muted"], thickness=2, width=10),
                text=[f"{v:.2f}" for v in sub.f1_mean], textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#FFFFFF", size=15), hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=tok["ink"], family="Inter, Open Sans, sans-serif", size=13),
                              yaxis=dict(range=[0, 1.08], title="F1 score (10-fold average)", gridcolor=tok["track"],
                                         zeroline=False),
                              xaxis=dict(title=None), bargap=0.45)
            st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
        with right:
            H('<p class="cc-title" style="font-size:15px">Reading this chart</p><p class="cc-text">Taller is better, '
              'and 1.0 is perfect. The thin line is how much the score moved across the 10 tests. When the lines '
              'overlap, no model is clearly better.</p>')
            note = data_note(cancer, n, pos)
            if note:
                box("warn", note, "warn")

    rows = []
    cols = [("accuracy", "Accuracy"), ("recall", "Catches cancer (recall)"), ("precision", "Precision"),
            ("f1", "Overall (F1)"), ("roc_auc", "Ranking (ROC-AUC)")]
    for _, r in cvc.iterrows():
        cells = "".join(f'<td>{r[f"{k}_mean"]:.2f} <span style="color:var(--muted)">± {r[f"{k}_std"]:.2f}</span></td>'
                        if pd.notna(r[f"{k}_mean"]) else '<td class="dim">n/a</td>' for k, _ in cols)
        rows.append(f"<tr><td><b>{r.Model}</b></td>{cells}</tr>")
    head = "".join(f"<th>{h}</th>" for h in ["Model"] + [c[1] for c in cols])
    H(f'<div class="cc-table-wrap" style="margin-top:6px"><table class="cc-table"><thead><tr>{head}</tr></thead>'
      f'<tbody>{"".join(rows)}</tbody></table></div>')
    st.caption("Average of 10 tests on the training data, ± how much it varied.")

    H('<div class="cc-mini" style="margin-top:14px">'
      '<div><b>Accuracy</b><span>Share of all patients the model got right.</span></div>'
      '<div><b>Recall</b><span>Of the people with cancer, how many it caught. Matters most.</span></div>'
      '<div><b>F1</b><span>One score balancing caught cancers against false alarms.</span></div>'
      '<div><b>ROC-AUC</b><span>How well it ranks sick above healthy. 0.5 is guessing.</span></div></div>')

    st.write("")
    with st.expander("Scores on the single held-out test split"):
        if not tst.empty:
            trows = "".join(f"<tr><td><b>{r.Model}</b></td>" + "".join(f"<td>{r[k]:.3f}</td>" for k in
                            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train F1"]) + "</tr>"
                            for _, r in tst.iterrows())
            H(f'<div class="cc-table-wrap"><table class="cc-table"><thead><tr><th>Model</th><th>Accuracy</th>'
              f'<th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Train F1</th></tr></thead>'
              f'<tbody>{trows}</tbody></table></div>')
            st.caption("If Train F1 is much higher than test F1, the model memorised the training data (overfitting).")
    with st.expander("Charts: most important inputs, ROC curves and confusion matrices"):
        a, b = st.columns(2)
        for col, name, cap in [(a, f"importance_{cancer}.png", "Most important inputs (Random Forest)"),
                               (b, f"roc_{cancer}.png", "ROC curves")]:
            path = os.path.join(FIG_DIR, name)
            if os.path.exists(path):
                show_image(col, path, cap)
        cm = st.columns(3)
        for col, key in zip(cm, MODELS):
            path = os.path.join(FIG_DIR, f"cm_{cancer}_{key}.png")
            if os.path.exists(path):
                show_image(col, path)


def show_image(col, path, caption=None):
    """Full-width image on both newer and older Streamlit versions."""
    try:
        col.image(path, caption=caption, width="stretch")
    except Exception:
        col.image(path, caption=caption, use_container_width=True)


# ---------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------
def page_about():
    hero("About this project",
         "This app compares three machine learning methods, an Artificial Neural Network (ANN), a Support "
         "Vector Machine (SVM) and a Random Forest, on lung, breast and cervical cancer data. Every dataset "
         "goes through exactly the same steps, so the comparison is fair.")
    H('<h3 style="margin:22px 0 10px 0">How the models were built</h3><div class="cc-pipe">'
      '<div><span class="cc-num">1</span><b>Clean the data</b><span class="t">Remove duplicates and columns that give the answer away.</span></div>'
      '<div><span class="cc-num">2</span><b>Set aside 20%</b><span class="t">One in five patients is kept hidden for the final test.</span></div>'
      '<div><span class="cc-num">3</span><b>Prepare</b><span class="t">Fill gaps, put values on one scale and balance small groups, using training data only.</span></div>'
      '<div><span class="cc-num">4</span><b>Tune 10 times</b><span class="t">Each model\'s settings are tested on 10 different splits.</span></div>'
      '<div><span class="cc-num">5</span><b>Final test</b><span class="t">Scored on the hidden patients with five measures.</span></div></div>')

    data = [("Lung cancer survey", "lung", "Kaggle. Yes or no answers about habits and symptoms, plus age and gender.",
             "Small, and most patients have cancer, unlike real life."),
            ("Breast cell measurements", "breast", "Kaggle, based on the Wisconsin breast cancer data. 30 measurements of cells from a tissue sample.",
             "Much larger than the original 569 patients, so it may be an expanded copy. Scores may look better than reality."),
            ("Cervical risk factors", "cervical", "UCI, collected at a hospital in Caracas, Venezuela. Lifestyle and health history, with the biopsy result as the answer.",
             "Very few cancer cases, and lifestyle answers alone predict cancer poorly."),
            ("Cervical with screening tests", "cervical_screening", "The same patients, plus three screening results from the same file: Hinselmann, Schiller and Pap smear.",
             "These tests are close to a diagnosis already, so this helps after screening, not for early warning.")]
    cards = []
    for title, c, text, limit in data:
        size = ""
        if c in available():
            size = f"{len(load_clean(c)):,} patients"
        cards.append(f'<div class="cc-mcard"><div class="top">{title}<span>{size}</span></div>'
                     f'<p class="cc-text">{text}</p><div class="cc-box warn" style="margin:0">{svg("warn", 16)}'
                     f'<span>{limit}</span></div></div>')
    H('<h3 style="margin:24px 0 10px 0">The datasets and their limits</h3>'
      f'<div class="cc-grid3" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">{"".join(cards)}</div>')

    with card():
        H('<p class="cc-title">What we found</p><ul class="cc-text" style="font-size:15px;line-height:1.7;margin:0">'
          '<li>No single method wins. ANN, SVM and Random Forest score within a few points of each other on every dataset.</li>'
          '<li>The data matters more than the method: strong scores on breast and lung, weak scores on cervical lifestyle data.</li>'
          '<li>Screening tests make cervical cancer predictable, which supports regular screening.</li></ul>')
    a, b = st.columns(2)
    with a:
        box("plain", "<b>Privacy.</b> Answers and uploaded files are used only during your visit. Nothing is saved or shared.", "lock")
    with b:
        box("plain", "<b>Not a medical device.</b> Built for a university project. It must not be used to diagnose or rule out cancer.", "shield")
    H('<p class="cc-foot">Final Year Project, Bachelor of Computer Science (Hons), Universiti Teknologi PETRONAS. '
      'Student: Azim Syafi bin Azman. Supervisor: Ts. Dr. Manoranjitham Muniandy.</p>')


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
def current_theme():
    return ss.get("theme_choice") or "Light"


with st.sidebar:
    H(f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0 18px 4px">'
      f'<div class="cc-icon" style="width:36px;height:36px;background:var(--btn);color:var(--accentink)">{svg("pulse", 18)}</div>'
      f'<span style="font-family:var(--head);font-weight:700;font-size:17px">Cancer Check</span></div>')
    st.radio("Menu", list(NAV), key="nav", format_func=lambda p: f"{NAV[p]} {p}", label_visibility="collapsed")
    st.divider()
    st.segmented_control("Theme", ["Light", "Dark", "Aero"], default="Light", key="theme_choice", width="stretch")
    st.caption("Student research tool. It cannot diagnose cancer.")

def aero_background():
    """Use static/aero_bg.(jpg|png|webp) as the Aero wallpaper if it exists."""
    for name in ("aero_bg.jpg", "aero_bg.jpeg", "aero_bg.png", "aero_bg.webp"):
        if os.path.exists(os.path.join(BASE_DIR, "static", name)):
            return f"app/static/{name}"
    return None


st.markdown(theme.css(current_theme(), aero_background()).replace("\n", " "), unsafe_allow_html=True)

if not available():
    hero("Models not found", "Run 02_preprocess.py and 03_train.py first, so the models/ and results/ folders exist.")
    st.stop()

page = ss.nav
if page == "Check a patient":
    if ss.view == "questions" and ss.cancer in available():
        page_questions(ss.cancer)
    elif ss.view == "result" and ss.cancer in available():
        page_result(ss.cancer)
    else:
        ss.view = "home"
        page_home()
elif page == "Upload CSV":
    page_upload()
elif page == "Model results":
    page_results()
else:
    page_about()
