import numpy as np
import pandas as pd
import streamlit as st
import joblib
from catboost import CatBoostRegressor

# =========================
# BASIC CONFIG
# =========================
st.set_page_config(
    page_title="AI Penaksir Harga Rumah",
    layout="wide"
)

MODEL_PATH = "house_price_catboost_log.cbm"
META_PATH = "house_price_catboost_meta.pkl"
TRAIN_CSV_PATH = "train_100k.csv"  # used for ranges & dropdown options

# CV metrics from training (for info & uncertainty band)
CV_R2_MEAN = 0.9241
CV_RMSE_MEAN = 20117.65


# =========================
# LOAD MODEL / META / STATS
# =========================
@st.cache_resource
def load_model_and_meta():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta

@st.cache_data
def load_training_stats():
    """Load train_100k.csv to infer sensible ranges & categorical options."""
    df = pd.read_csv(TRAIN_CSV_PATH)

    stats = {}
    numeric_cols = [
        "LotArea", "OverallQual", "YearBuilt",
        "GrLivArea", "GarageCars", "FullBath", "Bedrooms"
    ]
    for col in numeric_cols:
        stats[col] = {
            "min": int(df[col].min()),
            "max": int(df[col].max()),
            "median": int(df[col].median()),
            "q10": int(df[col].quantile(0.10)),
            "q90": int(df[col].quantile(0.90)),
        }

    cat_options = {}
    for col in ["Neighborhood", "BuildingType", "RoofStyle"]:
        cat_options[col] = sorted(df[col].unique().tolist())

    return stats, cat_options


model, meta = load_model_and_meta()
feature_names = meta["feature_names"]
cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]
cat_feature_indices = meta["cat_feature_indices"]

try:
    stats, cat_options = load_training_stats()
except Exception:
    st.error(
        "Tidak dapat membaca `train_100k.csv`. "
        "Pastikan file tersebut berada di folder yang sama dengan `app.py`."
    )
    st.stop()


# =========================
# GLOBAL STYLING (SPICED UP)
# =========================
st.markdown(
    """
    <style>
    /* Center content & constrain width */
    .main > div {
        max-width: 1150px;
        margin: 0 auto;
    }

    /* Hero gradient band */
    .hero-wrapper {
        position: relative;
        padding: 1.4rem 1.6rem 1.3rem 1.6rem;
        border-radius: 1.2rem;
        margin-bottom: 1.4rem;
        background: radial-gradient(circle at 0% 0%, rgba(45,212,191,0.28), transparent 55%),
                    radial-gradient(circle at 100% 0%, rgba(59,130,246,0.24), transparent 55%),
                    rgba(15,23,42,0.8);
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 18px 45px rgba(15,23,42,0.65);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
        letter-spacing: 0.02em;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.85;
        margin-bottom: 0.9rem;
    }
    .meta-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }
    .meta-pill {
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        border: 1px solid rgba(148,163,184,0.6);
        font-size: 0.8rem;
        opacity: 0.95;
        background: rgba(15,23,42,0.85);
    }

    .section-label {
        font-size: 1.1rem;
        font-weight: 750;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .group-card {
        padding: 0.9rem 1.0rem 0.8rem 1.0rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(148,163,184,0.35);
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.85));
        margin-bottom: 0.7rem;
        box-shadow: 0 10px 25px rgba(15,23,42,0.6);
    }
    .group-title {
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        color: rgb(94, 234, 212);
    }

    .price-card {
        padding: 1.5rem 1.6rem;
        border-radius: 1.1rem;
        border: 1px solid rgba(74,222,128,0.65);
        background: radial-gradient(circle at 10% 0%, rgba(74,222,128,0.2), transparent 55%),
                    radial-gradient(circle at 100% 100%, rgba(45,212,191,0.15), transparent 55%),
                    rgba(15,23,42,0.95);
        margin-top: 0.8rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 20px 50px rgba(22,163,74,0.55);
    }
    .price-main {
        font-size: 2.2rem;
        font-weight: 900;
        margin-bottom: 0.15rem;
    }
    .price-sub {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Accent color for sliders and selectboxes */
    .stSlider > div[data-baseweb="slider"] span[data-baseweb="slider-handle"] {
        box-shadow: 0 0 0 1px rgba(34,197,94,0.7);
    }
    .stSlider > div[data-baseweb="slider"] div[role="slider"] {
        background-color: rgb(34,197,94) !important;
    }
    .stSlider > div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, rgb(34,197,94), rgb(45,212,191)) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 999px;
        background: linear-gradient(90deg, rgb(37,99,235), rgb(45,212,191));
        color: white;
        border: none;
        padding: 0.45rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 12px 30px rgba(37,99,235,0.35);
    }
    .stButton > button:hover {
        filter: brightness(1.07);
        box-shadow: 0 14px 34px rgba(37,99,235,0.55);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# FEATURE ENGINEERING (same as training)
# =========================
def add_engineered_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["log_LotArea"] = np.log1p(df["LotArea"])
    df["log_GrLivArea"] = np.log1p(df["GrLivArea"])

    max_year = df["YearBuilt"].max()
    df["HouseAge"] = max_year - df["YearBuilt"]

    df["Qual_x_GrLiv"] = df["OverallQual"] * df["GrLivArea"]
    df["Baths_per_Bed"] = df["FullBath"] / (df["Bedrooms"] + 0.5)
    df["RoomsPlusBaths"] = df["Bedrooms"] + df["FullBath"]

    return df


def build_input_dataframe(
    lotarea,
    overallqual,
    yearbuilt,
    grlivarea,
    garagecars,
    fullbath,
    bedrooms,
    neighborhood,
    building_type,
    roof_style,
):
    base = pd.DataFrame(
        [{
            "LotArea": lotarea,
            "OverallQual": overallqual,
            "YearBuilt": yearbuilt,
            "GrLivArea": grlivarea,
            "GarageCars": garagecars,
            "FullBath": fullbath,
            "Bedrooms": bedrooms,
            "Neighborhood": neighborhood,
            "BuildingType": building_type,
            "RoofStyle": roof_style,
        }]
    )
    fe = add_engineered_features(base)
    fe = fe[feature_names]
    return fe


# =========================
# HERO HEADER
# =========================
st.markdown(
    """
    <div class="hero-wrapper">
      <div class="hero-title"> AI Penaksir Harga Rumah</div>
      <div class="hero-subtitle">
        Masukkan karakteristik rumah, dan model AI akan mengestimasi harga jualnya secara real-time.
      </div>
      <div class="meta-badges">
        <div class="meta-pill">CatBoost · log-target</div>
        <div class="meta-pill">Data latih: 100k+ baris</div>
        <div class="meta-pill">R² (5-fold CV): %.3f</div>
        <div class="meta-pill">RMSE (CV): %.0f</div>
      </div>
    </div>
    """ % (CV_R2_MEAN, CV_RMSE_MEAN),
    unsafe_allow_html=True,
)


# =========================
# LAYOUT: INPUTS (LEFT) & RESULT (RIGHT)
# =========================
left_col, right_col = st.columns([1.6, 1.0])


# ---------- LEFT: INPUT PANEL ----------
with left_col:
    st.markdown("<div class='section-label'> Atur parameter rumah</div>", unsafe_allow_html=True)

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    # --- Lokasi & Tahun ---
    with row1_col1:
        st.markdown("<div class='group-card'>", unsafe_allow_html=True)
        st.markdown("<div class='group-title'> Lokasi & Tahun</div>", unsafe_allow_html=True)

        neighborhood = st.selectbox(
            "Lokasi",
            options=cat_options["Neighborhood"],
            help="Lokasi/lingkungan rumah (diambil dari data latih).",
            label_visibility="visible"
        )

        yb = stats["YearBuilt"]
        yearbuilt = st.slider(
            "Tahun Bangun",
            min_value=yb["min"],
            max_value=yb["max"],
            value=yb["median"],
            step=1,
            help="Tahun rumah dibangun.",
            label_visibility="visible"
        )

        garagecars = st.selectbox(
            "Garasi (mobil)",
            options=list(range(stats["GarageCars"]["min"], stats["GarageCars"]["max"] + 1)),
            index=list(range(stats["GarageCars"]["min"], stats["GarageCars"]["max"] + 1)).index(stats["GarageCars"]["median"]),
            help="Berapa mobil yang dapat ditampung garasi.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Fisik ---
    with row1_col2:
        st.markdown("<div class='group-card'>", unsafe_allow_html=True)
        st.markdown("<div class='group-title'> Fisik</div>", unsafe_allow_html=True)

        oq = stats["OverallQual"]
        overallqual = st.slider(
            "Kualitas (1–9)",
            min_value=oq["min"],
            max_value=oq["max"],
            value=oq["median"],
            step=1,
            help="Kualitas material dan finishing secara keseluruhan.",
            label_visibility="visible"
        )

        gl = stats["GrLivArea"]
        grlivarea = st.slider(
            "Luas Bangunan (sq ft)",
            min_value=gl["q10"],
            max_value=min(gl["q90"], 10_000),
            value=gl["median"],
            step=50,
            help="Luas area tinggal di atas tanah (tidak termasuk basement).",
            label_visibility="visible"
        )

        la = stats["LotArea"]
        lotarea = st.slider(
            "Luas Tanah (sq ft)",
            min_value=la["q10"],
            max_value=min(la["q90"], 200_000),
            value=la["median"],
            step=100,
            help="Total luas tanah yang dimiliki.",
            label_visibility="visible"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Ruangan ---
    with row1_col3:
        st.markdown("<div class='group-card'>", unsafe_allow_html=True)
        st.markdown("<div class='group-title'> Ruangan</div>", unsafe_allow_html=True)

        fb = stats["FullBath"]
        fullbath = st.slider(
            "Kamar Mandi (full)",
            min_value=fb["min"],
            max_value=fb["max"],
            value=fb["median"],
            step=1,
            help="Full bathroom (ada wastafel, toilet, dan shower/bathtub).",
            label_visibility="visible"
        )

        bd = stats["Bedrooms"]
        bedrooms = st.slider(
            "Kamar Tidur",
            min_value=bd["min"],
            max_value=bd["max"],
            value=bd["median"],
            step=1,
            help="Jumlah kamar tidur di atas tanah.",
            label_visibility="visible"
        )

        building_type = st.selectbox(
            "Tipe Bangunan",
            options=cat_options["BuildingType"],
            help="Jenis rumah (single family, townhouse, dll).",
        )

        roof_style = st.selectbox(
            "Tipe Atap",
            options=cat_options["RoofStyle"],
            help="Bentuk atap rumah.",
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- RIGHT: RESULT PANEL ----------
with right_col:
    st.markdown("<div class='section-label'> Hasil estimasi</div>", unsafe_allow_html=True)

    # Build input df & predict (real-time)
    input_df = build_input_dataframe(
        lotarea=lotarea,
        overallqual=overallqual,
        yearbuilt=yearbuilt,
        grlivarea=grlivarea,
        garagecars=int(garagecars),
        fullbath=fullbath,
        bedrooms=bedrooms,
        neighborhood=neighborhood,
        building_type=building_type,
        roof_style=roof_style,
    )

    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    lower = max(pred_price - CV_RMSE_MEAN, 0)
    upper = pred_price + CV_RMSE_MEAN

    st.markdown(
        f"""
        <div class="price-card">
          <div class="price-main">Estimasi Harga: ${pred_price:,.0f}</div>
          <div class="price-sub">
            Rentang perkiraan (±1×RMSE): <b>${lower:,.0f}</b> – <b>${upper:,.0f}</b><br/>
            Rentang ini menggambarkan galat rata-rata model saat diuji pada data yang tidak dilatih.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Catatan: ini merupakan estimasi statistik berdasarkan data historis, bukan penilaian resmi."
    )

    with st.expander("Detail fitur yang dipertimbangkan model"):
        st.write("**Fitur numerik asli:**")
        st.code(", ".join([
            "LotArea", "OverallQual", "YearBuilt", "GrLivArea",
            "GarageCars", "FullBath", "Bedrooms"
        ]))

        st.write("**Fitur kategorikal:**")
        st.code(", ".join(["Neighborhood", "BuildingType", "RoofStyle"]))

        st.write("**Fitur turunan (feature engineering):**")
        st.code(
            """log_LotArea = log1p(LotArea)
log_GrLivArea = log1p(GrLivArea)
HouseAge = max(YearBuilt) - YearBuilt
Qual_x_GrLiv = OverallQual * GrLivArea
Baths_per_Bed = FullBath / (Bedrooms + 0.5)
RoomsPlusBaths = Bedrooms + FullBath"""
        )
        st.caption(
            "Fitur turunan membantu model menangkap hubungan non-linear seperti pengaruh gabungan "
            "antara luas & kualitas, serta efek usia bangunan."
        )
