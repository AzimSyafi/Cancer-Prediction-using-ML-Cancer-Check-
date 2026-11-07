"""
Themes for the Cancer Check app: Light, Dark and Frutiger Aero.

All colours live in TOKENS. css(theme) returns one <style> block that
restyles Streamlit's own widgets and the app's custom HTML (classes
starting with "cc-"). Keeping it in one file makes it easy to fix if a
future Streamlit update changes its internal class names.
"""

import re

TOKENS = {
    "Light": {
        "bg": "#F4F7F7", "bgfull": "#F4F7F7", "side": "#FFFFFF",
        "surface": "#FFFFFF", "surface2": "#F4F7F7", "border": "#DCE5E5",
        "ink": "#10292C", "muted": "#4D6467",
        "accent": "#0F766E", "accentink": "#FFFFFF", "btn": "#0F766E", "btnhover": "#0B5A54",
        "accentsoft": "#E6F3F1", "accentsoftink": "#0B5A54", "input": "#FFFFFF",
        "red": "#B42318", "redsoft": "#FDECEA", "redink": "#912018",
        "ambersoft": "#FEF3E7", "amberink": "#93370D",
        "green": "#067647", "greensoft": "#E8F5EE", "greenink": "#05603A",
        "bluesoft": "#E6F0FA", "blueink": "#1F4E8C",
        "track": "#EEF2F2", "axis": "#A9B8BA",
        "font": "'Inter', system-ui, sans-serif", "head": "'Plus Jakarta Sans', 'Inter', sans-serif",
        "shadow": "none", "blur": "none", "radius": "16px", "pill": "10px",
        "textshadow": "none", "hero": "transparent", "heropad": "0",
        "models": {"ANN": "#2F6FB0", "SVM": "#B4540A", "Random Forest": "#0F766E"},
    },
    "Dark": {
        "bg": "#0C1719", "bgfull": "#0C1719", "side": "#142427",
        "surface": "#142427", "surface2": "#1A2E31", "border": "#23383C",
        "ink": "#E4EEEE", "muted": "#9FB4B7",
        "accent": "#2BB3A5", "accentink": "#04201D", "btn": "#2BB3A5", "btnhover": "#5FD3C6",
        "accentsoft": "#12322F", "accentsoftink": "#7EDCD1", "input": "#1A2E31",
        "red": "#F97066", "redsoft": "#3A1B19", "redink": "#FDA29B",
        "ambersoft": "#36290F", "amberink": "#FEC84B",
        "green": "#47CD89", "greensoft": "#10301F", "greenink": "#6CE9A6",
        "bluesoft": "#162B45", "blueink": "#A4C8FF",
        "track": "#23383C", "axis": "#3B5256",
        "font": "'Inter', system-ui, sans-serif", "head": "'Plus Jakarta Sans', 'Inter', sans-serif",
        "shadow": "none", "blur": "none", "radius": "16px", "pill": "10px",
        "textshadow": "none", "hero": "transparent", "heropad": "0",
        "models": {"ANN": "#84ADFF", "SVM": "#FDB022", "Random Forest": "#2BB3A5"},
    },
    "Aero": {
        "bg": "#DDF2FE",
        "bgfull": "linear-gradient(180deg, #A9DCFA 0%, #DDF2FE 30%, #F3FBFF 55%, #D9F2CC 85%, #A6DC8C 100%) fixed",
        "side": "rgba(255,255,255,0.6)",
        "surface": "rgba(255,255,255,0.62)", "surface2": "rgba(255,255,255,0.55)",
        "border": "rgba(255,255,255,0.95)",
        "ink": "#0B2A4A", "muted": "#2A4A68",
        "accent": "#1279C4", "accentink": "#FFFFFF",
        "btn": "linear-gradient(180deg, #5FB8EE 0%, #1F86CF 48%, #0B6AB2 52%, #1279C4 100%)",
        "btnhover": "linear-gradient(180deg, #7CC8F3 0%, #2B92DA 48%, #0F74BE 52%, #1A86D2 100%)",
        "accentsoft": "rgba(207,235,251,0.85)", "accentsoftink": "#0A4F86", "input": "rgba(255,255,255,0.88)",
        "red": "#D93A26", "redsoft": "rgba(253,224,220,0.92)", "redink": "#A82615",
        "ambersoft": "rgba(255,236,204,0.92)", "amberink": "#8A3B00",
        "green": "#2E8F3D", "greensoft": "rgba(214,240,200,0.92)", "greenink": "#1F5F24",
        "bluesoft": "rgba(207,235,251,0.92)", "blueink": "#0A4F86",
        "track": "#D3EAF8", "axis": "#7FAACB",
        "font": "'Open Sans', 'Segoe UI', sans-serif", "head": "'Open Sans', 'Segoe UI', sans-serif",
        "shadow": "0 10px 30px rgba(20,100,170,0.18), inset 0 1px 0 #FFFFFF",
        "blur": "blur(14px)", "radius": "22px", "pill": "999px",
        "textshadow": "0 1px 1px rgba(0,50,100,0.45)", "hero": "rgba(255,255,255,0.6)", "heropad": "20px 24px",
        "models": {"ANN": "#1F86CF", "SVM": "#F59E0B", "Random Forest": "#2E8F3D"},
    },
}

FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
         "&family=Plus+Jakarta+Sans:wght@600;700&family=Open+Sans:wght@400;600;700&display=swap")

CSS = """
@import url('__FONTS__');
:root {
  --bg: __bg__; --side: __side__; --surface: __surface__; --surface2: __surface2__;
  --border: __border__; --ink: __ink__; --muted: __muted__;
  --accent: __accent__; --accentink: __accentink__; --btn: __btn__; --btnhover: __btnhover__;
  --accentsoft: __accentsoft__; --accentsoftink: __accentsoftink__; --input: __input__;
  --red: __red__; --redsoft: __redsoft__; --redink: __redink__;
  --ambersoft: __ambersoft__; --amberink: __amberink__;
  --green: __green__; --greensoft: __greensoft__; --greenink: __greenink__;
  --bluesoft: __bluesoft__; --blueink: __blueink__;
  --track: __track__; --axis: __axis__;
  --font: __font__; --head: __head__;
  --shadow: __shadow__; --blur: __blur__; --radius: __radius__; --pill: __pill__;
  --textshadow: __textshadow__;
}

/* ---------- Page and sidebar ---------- */
.stApp { background: __bgfull__; color: var(--ink); }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; position: relative; z-index: 1; }
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, label, button, input, textarea,
[data-testid="stWidgetLabel"] p { font-family: var(--font); }
.stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
h1, h2, h3, h4 { font-family: var(--head) !important; color: var(--ink) !important; }

[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--border);
  backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: var(--ink); }
[data-testid="stSidebar"] [data-testid="stRadioOption"] { padding: 9px 12px; border-radius: 10px; width: 100%; margin: 1px 0; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] { background: var(--btn); }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] span { color: var(--accentink) !important; font-weight: 600; text-shadow: var(--textshadow); }
[data-testid="stSidebar"] [data-testid="stRadioOption"] p { font-size: 15px; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; width: 100%; }
/* Older Streamlit versions: plain labels with a round radio dot */
[data-testid="stSidebar"] [role="radiogroup"] > label { padding: 9px 12px; border-radius: 10px; width: 100%; margin: 1px 0; }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) { background: var(--btn); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p { color: var(--accentink) !important; font-weight: 600; text-shadow: var(--textshadow); }
[data-testid="stSidebar"] [role="radiogroup"] > label p { font-size: 15px; }

/* ---------- Bordered containers become cards (keys start with "card") ---------- */
[class*="st-key-card"] {
  background: var(--surface); border-color: var(--border) !important; border-radius: var(--radius) !important;
  box-shadow: var(--shadow); backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
  padding: 20px 22px !important;
}
.stApp .stMarkdown h3 { font-size: 21px !important; padding: 0; }

/* ---------- Buttons ---------- */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
  min-height: 44px; border-radius: var(--pill); font-weight: 600;
  background: var(--surface); color: var(--ink); border: 1px solid var(--border);
}
.stButton button:hover, .stDownloadButton button:hover { border-color: var(--accent); color: var(--ink); }
.stButton button p, .stDownloadButton button p { color: inherit; font-weight: 600; }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--btn) !important; color: var(--accentink) !important; border: 1px solid var(--accent) !important;
  text-shadow: var(--textshadow);
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: var(--btnhover) !important; }
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] { color: var(--accentink) !important; }
button[kind="tertiary"], [data-testid="stBaseButton-tertiary"] {
  background: transparent !important; border: none !important; color: var(--accent) !important;
  min-height: 32px !important; padding: 0 !important;
}
button[kind="tertiary"] p, [data-testid="stBaseButton-tertiary"] p { color: var(--accentsoftink) !important; }

/* Segmented controls (Yes/No, steps, theme picker) */
button[data-variant="segmented_control"], button[data-variant="pills"] {
  background: var(--input) !important; color: var(--ink) !important; border-color: var(--border) !important; min-height: 40px;
}
button[data-variant="segmented_control"] p, button[data-variant="pills"] p { color: inherit !important; }
button[data-variant="segmented_control"][aria-checked="true"], button[data-variant="pills"][aria-checked="true"],
button[data-variant="segmented_control"][data-selected="true"], button[data-variant="pills"][data-selected="true"] {
  background: var(--btn) !important; color: var(--accentink) !important; border-color: var(--accent) !important;
  text-shadow: var(--textshadow); font-weight: 600;
}
/* Older Streamlit versions name the segmented buttons by test id instead */
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-pills"] {
  background: var(--input) !important; color: var(--ink) !important; border-color: var(--border) !important; min-height: 40px;
}
[data-testid="stBaseButton-segmented_control"] p, [data-testid="stBaseButton-pills"] p,
[data-testid="stBaseButton-segmented_control"] [data-testid="stMarkdownContainer"] { color: var(--ink) !important; }
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stBaseButton-pillsActive"] {
  background: var(--btn) !important; color: var(--accentink) !important; border-color: var(--accent) !important;
  text-shadow: var(--textshadow); font-weight: 600;
}
[data-testid="stBaseButton-segmented_controlActive"] p, [data-testid="stBaseButton-pillsActive"] p,
[data-testid="stBaseButton-segmented_controlActive"] [data-testid="stMarkdownContainer"] { color: var(--accentink) !important; }
button[data-variant][aria-checked="true"] p, button[data-variant][aria-checked="true"] [data-testid="stMarkdownContainer"],
button[data-variant][data-selected="true"] p, button[data-variant][data-selected="true"] [data-testid="stMarkdownContainer"] {
  color: var(--accentink) !important;
}

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
[data-testid="stNumberInputContainer"], [data-baseweb="textarea"] {
  background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px !important;
}
input, textarea, [data-baseweb="select"] span { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { background: var(--input) !important; color: var(--ink) !important; }
[data-testid="stNumberInputStepDown"] svg, [data-testid="stNumberInputStepUp"] svg { fill: var(--ink) !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] { background: var(--surface); color: var(--ink); }

[data-testid="stFileUploaderDropzone"] {
  background: var(--accentsoft) !important; border: 2px dashed var(--accent) !important; border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }

[data-testid="stExpander"] details { background: var(--surface); border-color: var(--border) !important; border-radius: 14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--ink) !important; }
hr { border-color: var(--border) !important; }

/* ---------- App HTML pieces ---------- */
.cc-hero { background: __hero__; padding: __heropad__; border-radius: 22px; backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur); margin-bottom: 8px; }
.cc-hero h1 { margin: 0 0 6px 0; font-size: 32px; line-height: 1.2; text-shadow: var(--textshadow-head, none); }
.cc-hero p { margin: 0; font-size: 16px; line-height: 1.6; color: var(--muted); max-width: 680px; }
.cc-steps { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 18px; }
.cc-step { display: flex; align-items: center; gap: 8px; padding: 5px 12px 5px 5px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--surface); color: var(--muted); font-size: 13px; font-weight: 500; }
.cc-step b { width: 22px; height: 22px; border-radius: 999px; background: var(--track); display: flex;
  align-items: center; justify-content: center; font-size: 12px; color: var(--muted); }
.cc-step.on { background: var(--btn); color: var(--accentink); border-color: var(--accent); font-weight: 600; text-shadow: var(--textshadow); }
.cc-step.on b { background: #FFFFFF; color: var(--accent); text-shadow: none; }
.cc-line { width: 24px; height: 2px; background: var(--border); }

.cc-card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
.cc-icon { width: 44px; height: 44px; border-radius: 12px; background: var(--accentsoft); display: flex;
  align-items: center; justify-content: center; color: var(--accent); flex: none; }
.cc-title { font-family: var(--head); font-size: 19px; font-weight: 700; margin: 0 0 6px 0; color: var(--ink); }
.cc-text { font-size: 14px; line-height: 1.55; color: var(--muted); margin: 0; }
.cc-meta { display: flex; gap: 16px; font-size: 13px; color: var(--muted); margin: 10px 0 4px 0; }
.cc-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 700; white-space: nowrap; }
.cc-badge.green { background: var(--greensoft); color: var(--greenink); }
.cc-badge.blue { background: var(--bluesoft); color: var(--blueink); }
.cc-badge.amber { background: var(--ambersoft); color: var(--amberink); }
.cc-badge.red { background: var(--redsoft); color: var(--redink); }

.cc-box { display: flex; gap: 12px; align-items: flex-start; padding: 14px 16px; border-radius: 12px;
  font-size: 14px; line-height: 1.55; margin: 6px 0 10px 0; }
.cc-box svg { flex: none; margin-top: 2px; }
.cc-box.info { background: var(--accentsoft); color: var(--accentsoftink); }
.cc-box.warn { background: var(--ambersoft); color: var(--amberink); }
.cc-box.plain { background: var(--surface2); color: var(--ink); }
.cc-box.bad { background: var(--redsoft); color: var(--redink); }

.cc-progress { display: flex; align-items: center; gap: 12px; margin: 4px 0 6px 0; }
.cc-progress .bar { flex: 1; height: 6px; border-radius: 999px; background: var(--track); overflow: hidden; }
.cc-progress .fill { height: 100%; border-radius: 999px; background: var(--btn); }
.cc-progress span { font-size: 13px; color: var(--muted); font-weight: 500; }

.cc-q { padding: 4px 0 2px 0; }
.cc-q .l { font-size: 15px; font-weight: 600; color: var(--ink); }
.cc-q .s { font-size: 13px; color: var(--muted); }
.cc-sum { display: flex; justify-content: space-between; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--border); font-size: 14px; }
.cc-sum span:first-child { color: var(--muted); }
.cc-sum span:last-child { font-weight: 600; color: var(--ink); }

.cc-hero-result { display: flex; flex-wrap: wrap; gap: 28px; align-items: center; }
.cc-hero-result .txt { flex: 1 1 340px; display: flex; flex-direction: column; gap: 10px; }
.cc-hero-result h2 { margin: 0; font-size: 23px !important; line-height: 1.3; }
.cc-grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 6px 0 4px 0; }
.cc-mcard { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px;
  box-shadow: var(--shadow); backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
  display: flex; flex-direction: column; gap: 10px; }
.cc-mcard .top { display: flex; justify-content: space-between; align-items: center; font-weight: 700; font-size: 15px; color: var(--ink); }
.cc-mcard .top span { font-size: 12px; font-weight: 400; color: var(--muted); }
.cc-mcard .res { font-size: 17px; font-weight: 700; }
.cc-mcard .res.yes { color: var(--redink); } .cc-mcard .res.no { color: var(--greenink); }
.cc-mbar { height: 8px; border-radius: 999px; background: var(--track); overflow: hidden; }
.cc-mbar div { height: 100%; border-radius: 999px; }
.cc-mcard .conf { font-size: 13px; color: var(--muted); }

.cc-why { display: grid; grid-template-columns: minmax(140px, 220px) minmax(0, 1fr) 96px; gap: 14px; align-items: center; margin: 8px 0; font-size: 14px; color: var(--ink); }
.cc-why .track { display: flex; height: 20px; }
.cc-why .neg { flex: 1; display: flex; justify-content: flex-end; border-right: 2px solid var(--axis); }
.cc-why .pos { flex: 1; display: flex; }
.cc-why .up { background: var(--red); border-radius: 0 6px 6px 0; }
.cc-why .down { background: var(--green); border-radius: 6px 0 0 6px; }
.cc-why .v { font-size: 13px; font-weight: 700; }
.cc-why .v.up-t { color: var(--redink); } .cc-why .v.down-t { color: var(--greenink); }

.cc-panel { border-radius: var(--radius); padding: 20px; display: flex; flex-direction: column; gap: 8px; }
.cc-panel.high { background: var(--redsoft); } .cc-panel.mid { background: var(--ambersoft); } .cc-panel.low { background: var(--greensoft); }
.cc-panel .big { font-family: var(--head); font-size: 42px; font-weight: 700; line-height: 1.1; color: var(--ink); }
.cc-panel .small { font-size: 13px; color: var(--muted); }

.cc-num { width: 30px; height: 30px; border-radius: 999px; background: var(--btn); color: var(--accentink);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; }
.cc-next h3 { margin: 4px 0 0 0; font-size: 16px !important; }
.cc-next p { margin: 0; font-size: 14px; line-height: 1.55; color: var(--muted); }

.cc-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 8px 0 12px 0; }
.cc-tile { border-radius: 14px; padding: 14px 16px; display: flex; flex-direction: column; gap: 4px;
  background: var(--surface); border: 1px solid var(--border); box-shadow: var(--shadow); }
.cc-tile .k { font-size: 13px; color: var(--muted); font-weight: 600; }
.cc-tile .n { font-family: var(--head); font-size: 26px; font-weight: 700; color: var(--ink); }
.cc-tile .d { font-size: 13px; color: var(--muted); }
.cc-tile.high { background: var(--redsoft); } .cc-tile.high .k, .cc-tile.high .n { color: var(--redink); }
.cc-tile.mid { background: var(--ambersoft); } .cc-tile.mid .k, .cc-tile.mid .n { color: var(--amberink); }
.cc-tile.low { background: var(--greensoft); } .cc-tile.low .k, .cc-tile.low .n { color: var(--greenink); }

.cc-table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); max-height: 520px; overflow-y: auto; }
.cc-table { width: 100%; min-width: 620px; border-collapse: collapse; font-size: 14px; color: var(--ink); }
.cc-table th { text-align: left; padding: 11px 14px; font-size: 13px; color: var(--muted); font-weight: 600;
  background: var(--surface2); position: sticky; top: 0; }
.cc-table td { padding: 11px 14px; border-top: 1px solid var(--border); }
.cc-table td.dim { color: var(--muted); }

.cc-mini { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.cc-mini div { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
.cc-mini b { display: block; font-size: 14px; margin-bottom: 4px; color: var(--ink); }
.cc-mini span { font-size: 13px; line-height: 1.5; color: var(--muted); }
.cc-pipe { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.cc-pipe div { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px; box-shadow: var(--shadow);
  display: flex; flex-direction: column; gap: 8px; }
.cc-pipe b { font-size: 15px; color: var(--ink); } .cc-pipe span.t { font-size: 13px; line-height: 1.5; color: var(--muted); }
.cc-foot { font-size: 13px; color: var(--muted); margin-top: 8px; }

/* Aero bubbles */
.cc-bubbles { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
.cc-bubbles span { position: absolute; border-radius: 999px; border: 1px solid rgba(255,255,255,0.7);
  background: radial-gradient(circle at 32% 28%, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.25) 35%,
  rgba(120,200,250,0.18) 70%, rgba(255,255,255,0.5) 100%); }
.cc-hill { position: fixed; left: -10%; right: -10%; bottom: -220px; height: 440px; border-radius: 50%;
  background: radial-gradient(ellipse at 50% 30%, #B9EAA0 0%, #7FCB63 55%, #5BB447 100%); opacity: 0.5;
  pointer-events: none; z-index: 0; }
"""

HILL = '<div class="cc-hill" aria-hidden="true"></div>'

BUBBLES = """
<div class="cc-bubbles" aria-hidden="true">
  <span style="top: 8%; right: 6%; width: 150px; height: 150px"></span>
  <span style="top: 26%; right: 2%; width: 60px; height: 60px"></span>
  <span style="top: 62%; left: 20%; width: 90px; height: 90px"></span>
  <span style="top: 78%; right: 12%; width: 120px; height: 120px"></span>
  <span style="top: 40%; left: 2%; width: 46px; height: 46px"></span>
</div>
"""


def _scope_app_pieces(text):
    """Prefix the app's own classes with .stApp so they beat Streamlit's default text styles."""
    marker = "/* ---------- App HTML pieces ---------- */"
    head, tail = text.split(marker, 1)
    tail = re.sub(r"(^|[},]\s*)\.cc-", r"\1.stApp .cc-", tail, flags=re.M)
    return head + marker + tail


def css(theme, background_url=None):
    """background_url: optional wallpaper for the Aero theme (served from the static folder)."""
    t = dict(TOKENS.get(theme, TOKENS["Light"]))
    if theme == "Aero" and background_url:
        # A soft white wash keeps text readable on top of the wallpaper.
        t["bgfull"] = (f"linear-gradient(rgba(255,255,255,0.18), rgba(255,255,255,0.18)), "
                       f"url('{background_url}') center / cover no-repeat fixed")
    out = _scope_app_pieces(CSS).replace("__FONTS__", FONTS)
    for key, value in t.items():
        if isinstance(value, str):
            out = out.replace(f"__{key}__", value)
    extra = ""
    if theme == "Aero":
        extra = BUBBLES if background_url else HILL + BUBBLES
    return f"<style>{out}</style>" + extra
