import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import io
import time

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Comment Bucketizer",
    page_icon="🗂️",
    layout="centered",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

/* Main font styling */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main container width mapping */
.block-container {
    max-width: 760px;
    padding-top: 2.5rem;
}

/* Header Component */
.app-header {
    background-color: var(--secondary-background-color);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.8rem;
    border: 1px solid var(--divider-color);
    text-align: center;
}
.app-header h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    color: var(--text-color);
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.app-header p {
    color: var(--text-color);
    opacity: 0.7;
    font-size: 0.92rem;
    margin: 0;
    line-height: 1.5;
}
.accent { color: #1E40AF; }
@supports (color: var(--primary-color)) {
    .accent { color: var(--primary-color); }
}

/* Card Titles */
.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-color);
    opacity: 0.6;
    margin-bottom: 0.8rem;
}

/* Stat boxes */
.stat-box {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--divider-color);
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1;
}
.accent-num { color: #1E40AF; }
@supports (color: var(--primary-color)) {
    .accent-num { color: var(--primary-color); }
}
.stat-label { 
    font-size: 0.75rem; 
    color: var(--text-color); 
    opacity: 0.6;
    margin-top: 6px; 
}

/* Uniform Buttons override (Clean Mono Style) */
.stDownloadButton > button, .stButton > button {
    width: 100% !important;
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--divider-color) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease;
}

.stDownloadButton > button:hover, .stButton > button:hover {
    background-color: var(--background-color) !important;
    border-color: var(--text-color) !important;
}

/* SPECIFIC BLUE RUN BUTTON STYLING (Clean Formal Blue) */
div[data-testid="stMainBlockContainer"] div.element-container:has(button[key="run_btn"]) button {
    background-color: #1E40AF !important; /* Formal Dark Blue */
    color: #FFFFFF !important;            /* High-contrast crisp white text */
    border: 1px solid #1D4ED8 !important;
}

div[data-testid="stMainBlockContainer"] div.element-container:has(button[key="run_btn"]) button:hover {
    background-color: #1D4ED8 !important; /* Slightly lighter blue on hover */
    border-color: #3B82F6 !important;
}

div[data-testid="stMainBlockContainer"] div.element-container:has(button[key="run_btn"]) button:disabled {
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    opacity: 0.4 !important;
    border: 1px solid var(--divider-color) !important;
}

/* Interactive feedback for delete buttons specifically */
div[data-testid="stHorizontalBlock"] button[id^="b'rm_"] {
    font-size: 0.8rem !important;
    padding: 0.3rem 0.5rem !important;
}

/* Streamlit Progress Bar Alignment */
.stProgress > div > div { background-color: #1E40AF !important; }
</style>
""", unsafe_allow_html=True)



# ── Model cache ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Core logic ──────────────────────────────────────────────────────────────────
def run_bucketization(df, comment_col, categories, output_col, progress_cb):
    model = load_model()
    anchors = categories

    cat_emb = model.encode(
        anchors, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )

    comments = df[comment_col].fillna("").astype(str).tolist()
    BATCH = 256
    all_emb = []

    for i in range(0, len(comments), BATCH):
        batch = comments[i : i + BATCH]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        all_emb.append(emb)
        progress_cb(min((i + BATCH) / len(comments), 1.0))

    comment_emb = np.vstack(all_emb)
    scores = comment_emb @ cat_emb.T
    best_idx = np.argmax(scores, axis=1)
    assigned = [categories[i] for i in best_idx]

    col_pos = df.columns.get_loc(comment_col) + 1
    df = df.copy()
    df.insert(col_pos, output_col, assigned)
    return df


# ── Session state defaults ──────────────────────────────────────────────────────
if "categories" not in st.session_state:
    st.session_state.categories = [
        "Quality Issues (Frame/Lens)",
        "Staff Assistance",
        "Delivery issues",
        "Pricing/Offers/Variety at store",
        "Poor Eye Test / Vision related issues",
        "Others",
        "NA"
    ]
if "result_df" not in st.session_state:
    st.session_state.result_df = None


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🗂️ Comment <span class="accent">Bucketizer</span></h1>
  <p>Upload an Excel file, define your categories, and get every comment classified</p>
</div>
""", unsafe_allow_html=True)


# ── Step 1: Upload ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-title">① Upload Excel File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["xlsx", "xls"], label_visibility="collapsed")

df = None
if uploaded:
    try:
        df = pd.read_excel(uploaded)
        st.success(f"Loaded **{len(df):,} rows** · {len(df.columns)} columns", icon="✅")
    except Exception as e:
        st.error(f"Could not read file: {e}")


# ── Step 2: Config ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-title">② Configure</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        comment_col = st.selectbox(
            "Comment column",
            options=df.columns.tolist() if df is not None else ["— upload a file first —"],
            disabled=df is None,
        )
    with col_b:
        output_col = st.text_input("New bucket column name", value="Bucket")


# ── Step 3: Categories ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-title">③ Define Categories</div>', unsafe_allow_html=True)

    add_col, btn_col = st.columns([3, 1])
    with add_col:
        new_cat = st.text_input("Add a category", placeholder="e.g. Refund Request", label_visibility="collapsed")
    with btn_col:
        add_clicked = st.button("＋ Add")

    if add_clicked and new_cat.strip():
        cat = new_cat.strip()
        if cat not in st.session_state.categories:
            st.session_state.categories.append(cat)
        st.rerun()

    # Render tags with remove buttons cleanly
    cats = st.session_state.categories
    if cats:
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        cols = st.columns(3) 
        to_remove = None
        for i, cat in enumerate(cats):
            with cols[i % 3]:
                if st.button(f"✕  {cat}", key=f"rm_{i}"):
                    to_remove = cat
        if to_remove:
            st.session_state.categories.remove(to_remove)
            st.rerun()
    else:
        st.warning("Add at least one category.")


# ── Step 4: Run ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
ready = df is not None and len(st.session_state.categories) > 0

# Added key="run_btn" so the CSS hooks onto this specific element perfectly
if st.button("🚀  Run", disabled=not ready, key="run_btn"):
    st.session_state.result_df = None
    prog_bar = st.progress(0, text="Encoding comments…")
    t0 = time.time()

    with st.spinner(""):
        result = run_bucketization(
            df,
            comment_col,
            st.session_state.categories,
            output_col,
            lambda p: prog_bar.progress(p, text=f"Processing… {p*100:.0f}%"),
        )

    elapsed = time.time() - t0
    prog_bar.empty()
    st.session_state.result_df = result
    st.session_state.elapsed = elapsed

# ── Step 5: Results ─────────────────────────────────────────────────────────────
if st.session_state.result_df is not None:
    result = st.session_state.result_df
    elapsed = st.session_state.get("elapsed", 0)
    counts = result[output_col].value_counts()

    st.markdown("---")

    # Stats Summary Layout
    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(result):,}</div><div class="stat-label">rows classified</div></div>', unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(counts)}</div><div class="stat-label">buckets used</div></div>', unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{elapsed:.1f}s</div><div class="stat-label">processing time</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribution chart
    st.markdown("**Bucket distribution**")
    st.bar_chart(counts)

    # Preview
    with st.expander("Preview results", expanded=False):
        st.dataframe(result[[comment_col, output_col]].head(50), use_container_width=True)

    # Download
    buf = io.BytesIO()
    result.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    fname = (uploaded.name.replace(".xlsx", "").replace(".xls", "") if uploaded else "output") + "_bucketed.xlsx"

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇  Download Bucketed Excel",
        data=buf,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )