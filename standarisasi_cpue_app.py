import base64
import io
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import bs, dmatrix
from PIL import Image
from scipy import stats
from scipy.interpolate import make_interp_spline
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# =========================================================
# DEFINISI NAMA BULAN
# =========================================================
month_map = {
    "1": "Jan",
    "2": "Feb",
    "3": "Mar",
    "4": "Apr",
    "5": "Mei",
    "6": "Jun",
    "7": "Jul",
    "8": "Agu",
    "9": "Sep",
    "10": "Okt",
    "11": "Nov",
    "12": "Des",
}


# Helper Function Format Angka
def fmt_num(val, decimals=2):
    if pd.isna(val) or val is None:
        return "-"
    try:
        val = float(val)
        formatted = f"{val:,.{decimals}f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(val)


def fmt_int(val):
    if pd.isna(val) or val is None:
        return "-"
    try:
        val = int(val)
        return f"{val:,}".replace(",", ".")
    except Exception:
        return str(val)


# Helper Function Konversi Plot Matplotlib ke Base64 String
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


# =========================================================
# FUNGSI CALCULATE EMMEANS PROPORTIONAL (DIPERBARUI)
# =========================================================
def calculate_emmeans_proportional(
    model_obj,
    target_col,
    df_orig,
    valid_cats,
    valid_nums,
    offset_col="log_effort",
    offset_val=1.0,
):
    other_cats = [c for c in valid_cats if c != target_col]
    if other_cats:
        cat_grid = df_orig.groupby(other_cats).size().reset_index(name="count")
        cat_grid["weight"] = cat_grid["count"] / cat_grid["count"].sum()
        for c in other_cats:
            cat_grid[c] = cat_grid[c].astype(str)
    else:
        cat_grid = pd.DataFrame({"weight": [1.0]})

    num_means = {c: df_orig[c].mean() for c in valid_nums}

    results = []
    target_levels = sorted(
        df_orig[target_col].dropna().unique(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )

    try:
        df_resid = model_obj.df_resid
    except Exception:
        df_resid = np.nan

    # --- Parameter model & matriks kovarians (dipakai untuk SE metode delta) ---
    beta = np.asarray(model_obj.params, dtype=float)
    cov = np.asarray(model_obj.cov_params(), dtype=float)

    def design_rows(grid_df):
        """Matriks desain untuk setiap baris grid, dibangun dengan aturan
        yang sama seperti saat model dilatih (kategori, spline bs(), dan
        penghalus GAM jika ada)."""
        mod = model_obj.model
        
        # Pengecekan bertingkat untuk design_info / formula pada berbagai versi statsmodels & GLMGam
        design_info = getattr(mod.data, "design_info", None)
        if design_info is None:
            design_info = getattr(mod, "design_info", None)

        if design_info is not None:
            X = np.asarray(
                dmatrix(design_info, grid_df, return_type="dataframe"),
                dtype=float,
            )
        else:
            formula_str = getattr(mod, "formula", None) or getattr(mod.data, "formula", None)
            if formula_str is not None:
                rhs = formula_str.split("~")[1] if "~" in formula_str else formula_str
                X = np.asarray(
                    dmatrix(rhs, grid_df, return_type="dataframe"),
                    dtype=float,
                )
            else:
                raise AttributeError("Objek model tidak memiliki 'design_info' maupun 'formula' yang valid.")

        smoother = getattr(mod, "smoother", None)
        if smoother is not None:  # GLMGam: tambahkan basis penghalus
            smooth_vars = list(smoother.variable_names)
            X = np.column_stack(
                [X, smoother.transform(grid_df[smooth_vars].to_numpy(dtype=float))]
            )
        if X.shape[1] != beta.shape[0]:
            raise ValueError(
                f"Jumlah kolom desain ({X.shape[1]}) tidak sama dengan jumlah"
                f" parameter model ({beta.shape[0]})."
            )
        return X

    for level in target_levels:
        grid = cat_grid.copy()
        grid[target_col] = str(level)

        for num_col, mean_val in num_means.items():
            grid[num_col] = mean_val

        grid[offset_col] = np.log(offset_val)

        try:
            X = design_rows(grid)
            w = grid["weight"].to_numpy(dtype=float)

            # CPUE terstandar = rata-rata tertimbang dari prediksi tiap kombinasi
            eta = X @ beta + np.log(offset_val)
            mu = np.exp(eta)
            weighted_mean = float(np.sum(w * mu))

            # SE metode delta: gradien estimator terhadap parameter, lalu
            # SE = sqrt(g' V g) dengan V = matriks kovarians parameter
            grad = X.T @ (w * mu)
            weighted_se = float(np.sqrt(grad @ cov @ grad))

            # CI 95% dihitung pada skala log agar selalu positif:
            # exp(log(CPUE) ± 1,96 * SE/CPUE)
            se_log = weighted_se / weighted_mean
            weighted_lower = weighted_mean * np.exp(-1.96 * se_log)
            weighted_upper = weighted_mean * np.exp(1.96 * se_log)
        except Exception as e:
            st.error(
                f"Gagal menghitung SE/CI CPUE terstandar untuk {target_col} ="
                f" {level}: {e}"
            )
            st.stop()

        results.append({
            target_col: str(level),
            "CPUE_std (kg/hari)": weighted_mean,
            "SE": weighted_se,
            "df": df_resid,
            "Lower CI": weighted_lower,
            "Upper CI": weighted_upper,
        })

    return pd.DataFrame(results)


# Helper Function Generator Laporan HTML
def generate_html_report(
    best_model_name,
    metrics_df,
    df_disp_table,
    df_stat_summary,
    norm_info,
    df_het,
    df_vif,
    grid_yr_display,
    grid_tm_display,
    len_data,
    time_cat,
    img_res_b64,
    img_grid_b64,
    img_yr_b64,
    img_tm_b64,
    partial_interp_html,
):
    best_row = metrics_df[metrics_df["Model"] == best_model_name].iloc[0]
    raw_r2_val = best_row["Pseudo_R2"]
    if isinstance(raw_r2_val, str):
        raw_r2_val = float(raw_r2_val.replace(".", "").replace(",", "."))

    raw_aic_val = best_row["AIC"]
    if isinstance(raw_aic_val, str):
        raw_aic_val = float(raw_aic_val.replace(".", "").replace(",", "."))

    best_aic = fmt_num(raw_aic_val, 2)
    best_r2 = fmt_num(raw_r2_val * 100, 2)
    
    # --- Format angka tabel Evaluasi Model---
    metrics_df_html = metrics_df.copy()
    for col in ["AIC", "Deviance", "Null_Deviance", "Pseudo_R2", "Overdispersion_Ratio", "Delta_AIC"]:
        if col in metrics_df_html.columns:
            metrics_df_html[col] = metrics_df_html[col].apply(lambda x: fmt_num(x, 2))
    if "N" in metrics_df_html.columns:
        metrics_df_html["N"] = metrics_df_html["N"].apply(fmt_int)

    stat_html = (
        df_stat_summary.to_html(index=False)
        if df_stat_summary is not None
        else "<p>Tidak ada data deskriptif.</p>"
    )
    het_html = (
        df_het.to_html(index=False)
        if df_het is not None and not df_het.empty
        else "<p>Tidak ada data uji Levene.</p>"
    )
    vif_html = (
        df_vif.to_html(index=False)
        if df_vif is not None and not df_vif.empty
        else "<p>Tidak ada data VIF.</p>"
    )

    yr_interp = ""
    yr_html = ""
    if grid_yr_display is not None and not grid_yr_display.empty:
        yr_html = grid_yr_display.to_html(index=False)
        max_yr_row = grid_yr_display.loc[
            grid_yr_display["CPUE_std (kg/hari)"]
            .apply(
                lambda x: (
                    float(str(x).replace(".", "").replace(",", "."))
                    if str(x) != "-"
                    else 0
                )
            )
            .idxmax()
        ]
        min_yr_row = grid_yr_display.loc[
            grid_yr_display["CPUE_std (kg/hari)"]
            .apply(
                lambda x: (
                    float(str(x).replace(".", "").replace(",", "."))
                    if str(x) != "-"
                    else 0
                )
            )
            .idxmin()
        ]
        yr_interp = f"""
        <div class="interpretation">
            <strong>Interpretasi CPUE Tahunan:</strong><br>
            Hasil standarisasi CPUE tahunan menunjukkan fluktuasi kelimpahan relatif ikan Yellowfin Tuna (YFT). 
            Tingkat CPUE terstandar tertinggi dicapai pada tahun <strong>{max_yr_row['tahun']}</strong> yaitu sebesar 
            <strong>{max_yr_row['CPUE_std (kg/hari)']} kg/hari</strong>, sedangkan CPUE terendah tercatat pada tahun 
            <strong>{min_yr_row['tahun']}</strong> sebesar <strong>{min_yr_row['CPUE_std (kg/hari)']} kg/hari</strong>. 
            Pita cakupan interval kepercayaan (CI 95%) mencerminkan tingkat presisi estimasi model terhadap dinamika stok tahunan.
        </div>
        """

    tm_interp = ""
    tm_html = ""
    if grid_tm_display is not None and not grid_tm_display.empty:
        tm_html = grid_tm_display.to_html(index=False)
        time_label = time_cat.title() if time_cat else "Waktu"
        max_tm_row = grid_tm_display.loc[
            grid_tm_display["CPUE_std (kg/hari)"]
            .apply(
                lambda x: (
                    float(str(x).replace(".", "").replace(",", "."))
                    if str(x) != "-"
                    else 0
                )
            )
            .idxmax()
        ]
        min_tm_row = grid_tm_display.loc[
            grid_tm_display["CPUE_std (kg/hari)"]
            .apply(
                lambda x: (
                    float(str(x).replace(".", "").replace(",", "."))
                    if str(x) != "-"
                    else 0
                )
            )
            .idxmin()
        ]
        tm_interp = f"""
        <div class="interpretation">
            <strong>Interpretasi CPUE {time_label}:</strong><br>
            Standarisasi CPUE berdasarkan <strong>{time_label}</strong> mengidentifikasi pola musim penangkapan ikan. 
            Puncak kelimpahan relatif (musim puncak penangkapan) terjadi pada <strong>{max_tm_row[time_cat]}</strong> 
            dengan nilai rata-rata CPUE terstandar sebesar <strong>{max_tm_row['CPUE_std (kg/hari)']} kg/hari</strong>, 
            sedangkan periode dengan CPUE terendah terjadi pada <strong>{min_tm_row[time_cat]}</strong> 
            sebesar <strong>{min_tm_row['CPUE_std (kg/hari)']} kg/hari</strong>.
        </div>
        """

    img_res_tag = (
        f'<img src="{img_res_b64}" class="chart-img">' if img_res_b64 else ""
    )
    img_grid_tag = (
        f'<img src="{img_grid_b64}" class="chart-img">' if img_grid_b64 else ""
    )
    img_yr_tag = (
        f'<img src="{img_yr_b64}" class="chart-img">' if img_yr_b64 else ""
    )
    img_tm_tag = (
        f'<img src="{img_tm_b64}" class="chart-img">' if img_tm_b64 else ""
    )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Laporan Hasil Uji Standarisasi CPUE</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; color: #333; line-height: 1.6; background-color: #f8f9fa; }}
            .container {{ max-width: 950px; margin: auto; background: #fff; padding: 35px; border-radius: 8px; box-shadow: 0 0 12px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; border-bottom: 3px solid #0E4C92; padding-bottom: 15px; margin-bottom: 25px; }}
            .header h1 {{ color: #0E4C92; margin: 0; font-size: 24px; }}
            .header p {{ color: #666; margin: 5px 0 0 0; font-size: 14px; }}
            .card {{ background: #f0f4f8; border-left: 5px solid #0E4C92; padding: 15px 20px; border-radius: 5px; margin-bottom: 20px; }}
            h2 {{ color: #0E4C92; font-size: 18px; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; page-break-after: avoid; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 12px; margin-bottom: 15px; font-size: 13px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
            th {{ background-color: #0E4C92; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .interpretation {{ background-color: #eef6fc; border: 1px solid #b8daff; border-radius: 5px; padding: 12px 15px; font-size: 13px; color: #004085; margin-bottom: 20px; line-height: 1.5; }}
            .chart-img {{ width: 100%; max-width: 850px; display: block; margin: 15px auto; border: 1px solid #ddd; border-radius: 6px; page-break-inside: avoid; }}
            .print-btn {{ text-align: center; margin-bottom: 20px; }}
            .btn {{ background-color: #0E4C92; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 14px; cursor: pointer; font-weight: bold; }}
            .btn:hover {{ background-color: #0a3871; }}
            .footer {{ text-align: center; font-size: 12px; color: #888; margin-top: 35px; border-top: 1px solid #ddd; padding-top: 12px; }}
            
            @media print {{
                body {{ background-color: #fff; margin: 0; padding: 0; }}
                .container {{ max-width: 100%; box-shadow: none; padding: 15px; }}
                .no-print {{ display: none !important; }}
                .page-break {{ page-break-before: always; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="print-btn no-print">
                <button class="btn" onclick="window.print()">🖨️ Cetak / Simpan sebagai PDF</button>
            </div>

            <div class="header">
                <h1>Laporan Hasil Uji Standarisasi CPUE</h1>
                <p>Aplikasi Pemodelan GLM & GAM — Yayasan MDPI (2026)</p>
            </div>
            
            <div class="card">
                <strong>Model Terbaik Terpilih:</strong> {best_model_name}<br>
                <strong>Total Sampel Valid:</strong> {fmt_int(len_data)} Observasi Trip<br>
                <strong>AIC Model Terpilih:</strong> {best_aic} | <strong>R² (Deviance Explained):</strong> {best_r2}%
            </div>
            
            <h2>1. Ringkasan Statistik Deskriptif Variabel</h2>
            {stat_html}

            <h2>2. Uji Asumsi Statistik</h2>
            <div class="interpretation">
                <strong>1. Uji Normalitas (berat_kg):</strong> {norm_info['test_name']}<br>
                Statistik Test = {fmt_num(norm_info['stat_val'], 4)} | p-value = {norm_info['p_val']}<br>
                <em>{norm_info['kesimpulan']}</em>
            </div>
            
            <strong>2. Uji Heterogenitas Varians (Levene's Test):</strong>
            {het_html}

            <strong>3. Uji Multikolinearitas (Variance Inflation Factor - VIF):</strong>
            {vif_html}

            <div class="page-break"></div>
            
            <h2>3. Evaluasi & Perbandingan Model</h2>
            {metrics_df_html.to_html(index=False)}
            <div class="interpretation">
                <strong>Interpretasi Evaluasi Model:</strong><br>
                Sesuai petunjuk dalam buku pedoman standarisasi CPUE, model <strong>{best_model_name}</strong> terpilih sebagai model terbaik berdasarkan kriteria <strong>Rasio Overdispersi terendah</strong> dan <strong>AIC terendah</strong>.
            </div>
            
            <h2>4. Evaluasi Dispersi Varians Seluruh Model</h2>
            {df_disp_table.to_html(index=False)}
            
            <h2>5. Residual Plot Model</h2>
            {img_res_tag}

            <div class="page-break"></div>

            <h2>6. Plot Efek Parsial Parameter ({best_model_name})</h2>
            {img_grid_tag}
            <div class="interpretation">
                <strong>Interpretasi Efek Parsial Parameter:</strong><br>
                {partial_interp_html}
            </div>

            <h2>7. Hasil Standarisasi CPUE Tahunan</h2>
            {yr_html}
            {img_yr_tag}
            {yr_interp}
            
            <h2>8. Hasil Standarisasi CPUE Bulanan / Musiman</h2>
            {tm_html}
            {img_tm_tag}
            {tm_interp}
            
            <div class="footer">
                &copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i>.
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


# Helper Function Seleksi Model Mundur (Backward Elimination berbasis AIC)
def backward_elimination_aic(base_terms, term_labels, df_data, offset_col, response="berat_kg"):
    """
    Mereplikasi logika drop1()/stepwise regression pada buku pedoman:
    setiap iterasi mencoba menghapus satu term, lalu term yang jika dihapus
    justru menurunkan AIC model akan dibuang secara permanen dari model.
    Proses berhenti ketika tidak ada lagi term yang jika dihapus menurunkan AIC,
    atau ketika tinggal 1 term tersisa. Model dasar yang dipakai untuk seleksi
    adalah GLM Poisson (konsisten dengan tahap awal pemilihan model di buku).
    """
    current_terms = list(base_terms)
    log_rows = []

    def fit_aic(terms_list):
        formula = f"{response} ~ " + " + ".join(terms_list) if terms_list else f"{response} ~ 1"
        mod = smf.glm(
            formula=formula,
            data=df_data,
            offset=df_data[offset_col],
            family=sm.families.Poisson(link=sm.families.links.Log()),
        ).fit()
        return mod.aic

    try:
        aic_current = fit_aic(current_terms)
    except Exception:
        return current_terms, pd.DataFrame([{
            "Iterasi": "-",
            "Term Dievaluasi": "-",
            "AIC Jika Dihapus": "-",
            "Keputusan": "Gagal menghitung AIC awal — seleksi model dilewati.",
        }])

    iterasi = 0
    while len(current_terms) > 1:
        iterasi += 1
        aic_if_dropped = {}
        for term in current_terms:
            reduced = [t for t in current_terms if t != term]
            try:
                aic_if_dropped[term] = fit_aic(reduced)
            except Exception:
                aic_if_dropped[term] = np.inf

        for term in current_terms:
            log_rows.append({
                "Iterasi": iterasi,
                "Term Dievaluasi": term_labels.get(term, term),
                "AIC Model Penuh": fmt_num(aic_current, 2),
                "AIC Jika Dihapus": fmt_num(aic_if_dropped[term], 2),
                "Keputusan": "-",
            })

        term_to_drop = min(aic_if_dropped, key=aic_if_dropped.get)
        best_aic_after_drop = aic_if_dropped[term_to_drop]

        if best_aic_after_drop < aic_current:
            for row in log_rows[-len(current_terms):]:
                if row["Term Dievaluasi"] == term_labels.get(term_to_drop, term_to_drop):
                    row["Keputusan"] = "Dihapus (AIC menurun)"
                else:
                    row["Keputusan"] = "Dipertahankan pada iterasi ini"
            current_terms.remove(term_to_drop)
            aic_current = best_aic_after_drop
        else:
            for row in log_rows[-len(current_terms):]:
                row["Keputusan"] = "Dipertahankan (menghapus term manapun menaikkan AIC)"
            break

    return current_terms, pd.DataFrame(log_rows)


# =========================================================
# 1. KONFIGURASI HALAMAN & STYLING STREAMLIT
# =========================================================
try:
    logo = Image.open("_ MDPI Primary Logo.png")
except Exception:
    logo = "🐟"

st.set_page_config(
    page_title="Standarisasi CPUE YFT - MDPI", page_icon=logo, layout="wide"
)

st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #f7f9fc;
        border: 1px solid #e1e4e8;
        padding: 12px 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #0E4C92;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 20px !important;
        font-weight: bold;
        color: #0E4C92;
        word-break: break-word;
    }
    div[data-testid="stMetricLabel"] > label {
        font-size: 13px !important;
        color: #555555;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None


logo_base64 = get_base64_image("_ MDPI Primary Logo.png")


def render_footer():
    st.markdown("---")
    if logo_base64:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 15px; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_base64}" width="170" style="margin-bottom: 10px;">
                <p style="font-size: 13px; color: #666666; margin: 0;">
                    &copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 15px; margin-bottom: 20px; color: #666666;">
                <p style="font-size: 13px; margin: 0;">
                    &copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <h1 style='color:#0E4C92; margin-bottom:0px;'>Aplikasi Standarisasi Catch Per Unit Effort (CPUE)</h1>
    <h3 style='color:#444444; margin-top:5px;'>Pemodelan GLM & GAM untuk Standarisasi Catch Per Unit Effort</h3>
    <p style='color:#666666;'>*Tahap Pengembangan</p>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# 2. PETUNJUK STRUKTUR & UPLOAD FILE DATA EXCEL
# =========================================================
uploaded_file = st.file_uploader(
    "Upload file Excel Data Catch & Effort (.xlsx)", type=["xlsx"]
)

if uploaded_file is None:
    st.info(
        "**Silakan unggah file Excel (.xlsx)** yang berisi data operasional"
        " penangkapan dan lingkungan penangkapan untuk memulai proses analisis"
        " (minimal selama 5 tahun)."
    )

    st.markdown(
        """
        <h3 style="font-size:18px; color:#0E4C92; margin-top:20px;">
        Petunjuk Struktur Kolom File Excel
        </h3>
        """,
        unsafe_allow_html=True,
    )

    petunjuk = pd.DataFrame({
        "Nama Kolom": [
            "berat_kg",
            "id_trip",
            "tahun",
            "bulan",
            "musim",
            "quarter",
            "jumlah_hari_memancing",
            "abk",
            "kapasitas_mesin",
            "panjang_kapal",
            "gross_tonnage",
            "teknik_penangkapan",
            "jenis_alat_tangkap",
            "daerah_spasial",
            "sst",
            "chl_a",
        ],
        "Tipe Data": [
            "Numerik (kg)",
            "Teks / Numerik",
            "Numerik (YYYY)",
            "Numerik (1–12)",
            "Teks (Barat / Timur / Peralihan)",
            "Numerik (1–4)",
            "Numerik (Hari)",
            "Numerik (Orang)",
            "Numerik (PK / HP)",
            "Numerik (Meter)",
            "Numerik (GT)",
            "Teks (Rumpon / Non Rumpon / Campuran)",
            "Teks (Handline / Longline / dll)",
            "Teks (Grid / WPP / Nama Perairan)",
            "Numerik (°C)",
            "Numerik (mg/m³)",
        ],
        "Status": ["Wajib (Target)"] + ["Opsional"] * 15,
        "Keterangan": [
            "Total berat hasil tangkapan (Variabel Target)",
            "Identitas unik perjalanan/trip penangkapan (Identifier)",
            "Tahun operasional penangkapan",
            "Bulan operasional penangkapan",
            "Musim penangkapan",
            "Kuartal tahunan (Q1–Q4)",
            "Jumlah hari memancing dalam 1 trip (Effort/Offset)",
            "Jumlah Anak Buah Kapal",
            "Kapasitas daya mesin kapal",
            "Panjang dimensi kapal",
            "Ukuran tonase kotor kapal (GT)",
            "Metode/teknik penangkapan",
            "Spesifikasi alat tangkap yang digunakan",
            "Lokasi/Grid/WPP/Wilayah spasial penangkapan",
            "Suhu Permukaan Laut (Sea Surface Temperature)",
            "Konsentrasi Klorofil-a",
        ],
    })

    st.dataframe(petunjuk, use_container_width=False, hide_index=True)

    render_footer()
    st.stop()

df = pd.read_excel(uploaded_file)

if "berat" in df.columns and "berat_kg" not in df.columns:
    df["berat_kg"] = df["berat"]

if "berat_kg" not in df.columns:
    st.error(
        "❌ Kolom target **'berat'** atau **'berat_kg'** tidak ditemukan dalam file Excel. Mohon"
        " pastikan nama kolom target sesuai."
    )
    render_footer()
    st.stop()

# =========================================================
# 3. PRE-PROCESSING DATA
# =========================================================
cat_candidates = [
    "tahun",
    "bulan",
    "musim",
    "quarter",
    "alat_tangkap",
    "teknik_penangkapan",
    "jenis_alat_tangkap",
    "daerah_spasial",
    "daerah",
]
num_candidates = [
    "abk",
    "panjang_kapal",
    "kapasitas_mesin",
    "gross_tonnage",
    "gt",
    "sst",
    "chl_a",
]
effort_candidates = ["jumlah_hari_memancing", "days_at_sea", "das", "effort"]

avail_cats = [c for c in cat_candidates if c in df.columns]
avail_nums = [c for c in num_candidates if c in df.columns]
effort_col = next((c for c in effort_candidates if c in df.columns), None)

df["berat_kg"] = pd.to_numeric(df["berat_kg"], errors="coerce")
for c in avail_nums:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if effort_col:
    df[effort_col] = pd.to_numeric(df[effort_col], errors="coerce")
    df = df[df[effort_col] > 0].copy()

used_cols = ["berat_kg"] + avail_cats + avail_nums
if effort_col:
    used_cols.append(effort_col)

df_model = df.dropna(subset=used_cols).copy()

with st.expander(
    "Deteksi Outlier & Nilai Ekstrem", expanded=False
):
    st.caption(
        "Pemeriksaan visual Boxplot dan filter statistik IQR untuk mencegah"
        " kesalahan input berat tangkapan/effort."
    )

    q1_target = df_model["berat_kg"].quantile(0.25)
    q3_target = df_model["berat_kg"].quantile(0.75)
    iqr_target = q3_target - q1_target
    lower_target = max(0.0, q1_target - 1.5 * iqr_target)
    upper_target = q3_target + 1.5 * iqr_target

    outliers_target = df_model[
        (df_model["berat_kg"] < lower_target)
        | (df_model["berat_kg"] > upper_target)
    ]

    col_out1, col_plot_out = st.columns([1, 2])
    with col_out1:
        st.metric(
            "Pencilan Terdeteksi (berat_kg)", fmt_int(len(outliers_target))
        )
        st.caption(
            f"Batas Wajar IQR: **{fmt_num(lower_target)}** kg s/d"
            f" **{fmt_num(upper_target)}** kg"
        )
        filter_outliers = st.checkbox(
            "❌ Filter / Keluarkan Data Pencilan Sebelum Pemodelan"
        )

    with col_plot_out:
        fig_box, (ax_box1, ax_box2) = plt.subplots(1, 2, figsize=(8, 2.5))
        sns.boxplot(y=df_model["berat_kg"], ax=ax_box1, color="#0E4C92")
        ax_box1.set_title(
            "Boxplot berat_kg", fontsize=9, fontweight="bold"
        )

        if effort_col:
            sns.boxplot(
                y=df_model[effort_col], ax=ax_box2, color="#E67E22"
            )
            ax_box2.set_title(
                f"Boxplot {effort_col}", fontsize=9, fontweight="bold"
            )
        else:
            ax_box2.axis("off")

        plt.tight_layout()
        st.pyplot(fig_box)
        plt.close(fig_box)

    if filter_outliers:
        df_model = df_model[
            (df_model["berat_kg"] >= lower_target)
            & (df_model["berat_kg"] <= upper_target)
        ].copy()
        st.success(
            "Berhasil menyaring pencilan! Sampel tersisa:"
            f" **{fmt_int(len(df_model))}** data."
        )

if effort_col:
    df_model["log_effort"] = np.log(df_model[effort_col])
else:
    df_model["log_effort"] = 0.0

valid_cats = []
for c in avail_cats:
    n_unq = df_model[c].nunique()
    if 1 < n_unq < (len(df_model) * 0.5):
        df_model[c] = df_model[c].astype(str)
        valid_cats.append(c)

valid_nums = []
for c in avail_nums:
    if df_model[c].nunique() > 1:
        valid_nums.append(c)

if len(df_model) < 10:
    st.error("❌ Jumlah data valid terlalu sedikit untuk melakukan pemodelan.")
    st.stop()

# =========================================================
# 4. PEMBENTUKAN FORMULA & PEMODELAN
# =========================================================
glm_terms = [f"C({c})" for c in valid_cats] + valid_nums
if not glm_terms:
    st.error("❌ Tidak ada variabel prediktor yang valid untuk dimodelkan.")
    st.stop()

term_labels = {f"C({c})": c for c in valid_cats}
term_labels.update({c: c for c in valid_nums})

with st.spinner("Melakukan seleksi model (backward elimination berbasis AIC)..."):
    selected_terms, selection_log_df = backward_elimination_aic(
        glm_terms, term_labels, df_model, "log_effort", response="berat_kg"
    )

dropped_terms = [t for t in glm_terms if t not in selected_terms]
dropped_labels = [term_labels.get(t, t) for t in dropped_terms]
kept_labels = [term_labels.get(t, t) for t in selected_terms]

final_terms = selected_terms if selected_terms else glm_terms
formula_glm = "berat_kg ~ " + " + ".join(final_terms)

selected_cats = [c for c in valid_cats if f"C({c})" in final_terms]
selected_nums = [c for c in valid_nums if c in final_terms]

models = {}
with st.spinner("Sedang melatih model GLM & GAM..."):
    # 1. GLM Poisson
    try:
        pois_model = smf.glm(
            formula=formula_glm,
            data=df_model,
            offset=df_model["log_effort"],
            family=sm.families.Poisson(link=sm.families.links.Log()),
        ).fit()
        models["GLM Poisson"] = pois_model
    except Exception:
        pass

    # 2. Estimasi Parameter Dispersi Alpha
    est_alpha = 1.0
    if "GLM Poisson" in models:
        try:
            disp_p = pois_model.pearson_chi2 / pois_model.df_resid
            est_alpha = max(0.001, (disp_p - 1) / pois_model.mu.mean())
        except Exception:
            est_alpha = 1.0

    # 3. GLM Negative Binomial
    try:
        models["GLM Negative Binomial"] = smf.glm(
            formula=formula_glm,
            data=df_model,
            offset=df_model["log_effort"],
            family=sm.families.NegativeBinomial(
                alpha=est_alpha, link=sm.families.links.Log()
            ),
        ).fit()
    except Exception:
        pass

    # 4. Tweedie Compound Poisson
    best_p = 1.5
    best_llf = -np.inf
    for p in np.arange(1.1, 2.0, 0.1):
        try:
            tw_temp = smf.glm(
                formula=formula_glm,
                data=df_model,
                offset=df_model["log_effort"],
                family=sm.families.Tweedie(
                    var_power=p, link=sm.families.links.Log()
                ),
            ).fit()
            if tw_temp.llf > best_llf:
                best_llf = tw_temp.llf
                best_p = p
        except Exception:
            continue

    try:
        models["Tweedie"] = smf.glm(
            formula=formula_glm,
            data=df_model,
            offset=df_model["log_effort"],
            family=sm.families.Tweedie(
                var_power=best_p, link=sm.families.links.Log()
            ),
        ).fit()
    except Exception:
        pass

    # 5. GAM Negative Binomial
    gam_success = False
    if selected_nums:
        try:
            gam_linear_terms = [f"C({c})" for c in selected_cats]
            formula_gam_lin = (
                "berat_kg ~ " + " + ".join(gam_linear_terms)
                if gam_linear_terms
                else "berat_kg ~ 1"
            )
            x_spline = df_model[selected_nums]
            bs_obj = BSplines(
                x_spline, df=[4] * len(selected_nums), degree=[3] * len(selected_nums)
            )

            gam_model = GLMGam.from_formula(
                formula_gam_lin,
                data=df_model,
                smoother=bs_obj,
                offset=df_model["log_effort"],
                family=sm.families.NegativeBinomial(
                    alpha=est_alpha, link=sm.families.links.Log()
                ),
            )
            alpha_penalties = gam_model.select_penalties(gam_model.fit())
            models["GAM / Spline Negative Binomial"] = gam_model.fit(
                penalties=alpha_penalties
            )
            gam_success = True
        except Exception:
            pass

        if not gam_success:
            try:
                gam_terms = [f"C({c})" for c in selected_cats]
                for c in selected_nums:
                    if df_model[c].nunique() > 4:
                        gam_terms.append(f"bs({c}, df=4)")
                    else:
                        gam_terms.append(c)
                formula_gam = "berat_kg ~ " + " + ".join(gam_terms)
                models["GAM / Spline Negative Binomial"] = smf.glm(
                    formula=formula_gam,
                    data=df_model,
                    offset=df_model["log_effort"],
                    family=sm.families.NegativeBinomial(
                        alpha=est_alpha, link=sm.families.links.Log()
                    ),
                ).fit()
            except Exception:
                pass
    else:
        if "GLM Negative Binomial" in models:
            models["GAM / Spline Negative Binomial"] = models["GLM Negative Binomial"]

if not models:
    st.error("❌ Seluruh model gagal konvergen.")
    st.stop()

# =========================================================
# SELEKSI MODEL (BERDASARKAN RASIO OVERDISPERSI TERENDAH KEMUDIAN AIC TERENDAH)
# =========================================================
metrics = []
for name, mod in models.items():
    if np.isinf(mod.aic) or np.isnan(mod.aic) or mod.aic < -1e6:
        continue

    pseudo_r2 = 1 - (mod.deviance / mod.null_deviance)
    disp_ratio = mod.pearson_chi2 / mod.df_resid

    metrics.append({
        "Model": name,
        "AIC": mod.aic,
        "Deviance": mod.deviance,
        "Null_Deviance": mod.null_deviance,
        "Pseudo_R2": max(0.0, pseudo_r2),
        "Overdispersion_Ratio": disp_ratio,
        "N": int(mod.nobs),
    })

if not metrics:
    st.error("❌ Tidak ada model valid yang berhasil dilatih.")
    st.stop()

metrics_df = pd.DataFrame(metrics)

# Urutkan berdasarkan Rasio Overdispersi terendah lalu AIC terendah
metrics_df = metrics_df.sort_values(
    by=["Overdispersion_Ratio", "AIC"], ascending=[True, True]
).reset_index(drop=True)

best_model_name = metrics_df.iloc[0]["Model"]
best_model_aic = metrics_df.iloc[0]["AIC"]
metrics_df["Delta_AIC"] = metrics_df["AIC"] - best_model_aic
valid_model_list = list(metrics_df["Model"])

# PREPARASI TABEL RINGKASAN STATISTIK DESKRIPTIF
stat_rows = []
num_list = list(
    dict.fromkeys(
        ["berat_kg"] + ([effort_col] if effort_col else []) + valid_nums
    )
)

for col in num_list:
    if col in df_model.columns:
        s = df_model[col]
        stat_rows.append({
            "Nama Variabel": col,
            "Tipe Data": "Numerik",
            "Jumlah (N)": fmt_int(len(s)),
            "Mean ± Std": f"{fmt_num(s.mean(), 2)} ± {fmt_num(s.std(), 2)}",
            "Min": fmt_num(s.min(), 2),
            "Median": fmt_num(s.median(), 2),
            "Max": fmt_num(s.max(), 2),
            "Keterangan / Modus": "-",
        })

for col in valid_cats:
    if col in df_model.columns:
        s = df_model[col]
        mode_val = s.mode()[0] if not s.mode().empty else "-"
        stat_rows.append({
            "Nama Variabel": col,
            "Tipe Data": "Kategorikal",
            "Jumlah (N)": fmt_int(len(s)),
            "Mean ± Std": "-",
            "Min": "-",
            "Median": "-",
            "Max": "-",
            "Keterangan / Modus": (
                f"{fmt_int(s.nunique())} Kat. (Modus: {mode_val})"
            ),
        })

df_stat_summary = pd.DataFrame(stat_rows)

# =========================================================
# 5. DASHBOARD & HASIL ANALISIS
# =========================================================
st.caption(
    f"**Status Analisis:** Berhasil memproses **{fmt_int(len(df_model))}**"
    f" observasi trip. Prediktor aktif: **{fmt_int(len(valid_cats))}**"
    f" Kategorikal, **{fmt_int(len(valid_nums))}** Numerik."
)

tab0, tab1, tab2, tab3 = st.tabs(
    [
        "Uji Asumsi & Statistik",
        "Evaluasi Model",
        "Efek Parsial Parameter",
        "CPUE Terstandarisasi",
    ]
)

norm_info = {}
df_het = None
df_vif = None

# --- TAB 0: UJI ASUMSI & STATISTIK ---
with tab0:
    st.subheader("Ringkasan Statistik Deskriptif Variabel")
    col_stat, _ = st.columns([4, 1])
    with col_stat:
        st.dataframe(df_stat_summary, use_container_width=False, hide_index=True)

    st.markdown("---")
    st.subheader("Visualisasi Sebaran Data (Boxplot)")

    st.markdown("**Boxplot Variabel Target (`berat_kg`) Berdasarkan Kategori Utama & Faktor Lingkungan**")

    # BARIS 1: SELURUH DATA, PER TAHUN, DAN PER BULAN (AGREGAT SELURUH TAHUN)
    fig_bkg1, (ax_b0, ax_b1, ax_b2) = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Seluruh Data
    sns.boxplot(y=df_model["berat_kg"], ax=ax_b0, color="#0E4C92")
    ax_b0.set_title("Berat Ikan (Seluruh Data)", fontsize=9, fontweight="bold")
    ax_b0.set_ylabel("Berat (kg)", fontsize=8)

    # 2. Per Tahun
    if "tahun" in df_model.columns:
        df_sort_yr = df_model.copy()
        df_sort_yr["tahun_sort"] = pd.to_numeric(df_sort_yr["tahun"], errors="coerce")
        df_sort_yr = df_sort_yr.sort_values("tahun_sort", na_position="last")
        sns.boxplot(x="tahun", y="berat_kg", data=df_sort_yr, ax=ax_b1, palette="Blues")
        ax_b1.set_title("Berat Ikan Per Tahun", fontsize=9, fontweight="bold")
        ax_b1.set_xlabel("Tahun", fontsize=8)
        ax_b1.set_ylabel("Berat (kg)", fontsize=8)
        ax_b1.set_xticklabels(ax_b1.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    else:
        ax_b1.axis("off")

    # 3. Per Bulan (Agregat Seluruh Tahun)
    if "bulan" in df_model.columns:
        df_sort_mo = df_model.copy()
        df_sort_mo["bulan_num"] = pd.to_numeric(df_sort_mo["bulan"], errors="coerce")
        df_sort_mo = df_sort_mo.sort_values("bulan_num")
        df_sort_mo["bulan_lbl"] = df_sort_mo["bulan_num"].astype(str).map(
            lambda x: month_map.get(str(int(float(x))) if str(x).replace('.', '').isdigit() else str(x), str(x))
        )
        sns.boxplot(x="bulan_lbl", y="berat_kg", data=df_sort_mo, ax=ax_b2, palette="Greens")
        ax_b2.set_title("Berat Ikan Per Bulan (Agregat)", fontsize=9, fontweight="bold")
        ax_b2.set_xlabel("Bulan", fontsize=8)
        ax_b2.set_ylabel("Berat (kg)", fontsize=8)
        ax_b2.set_xticklabels(ax_b2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    else:
        ax_b2.axis("off")

    plt.tight_layout()
    st.pyplot(fig_bkg1)
    plt.close(fig_bkg1)


    # MENGGABUNGKAN SISA PLOT (Kategori Lingkungan & Variabel Numerik Lainnya)
    plot_tasks = []
    
    if "quarter" in df_model.columns:
        df_sort_q = df_model.copy()
        df_sort_q["quarter_num"] = pd.to_numeric(df_sort_q["quarter"], errors="coerce")
        df_sort_q = df_sort_q.sort_values("quarter_num", na_position="last")
        plot_tasks.append({
            "type": "cat", "x": "quarter", "y": "berat_kg", "data": df_sort_q,
            "title": "Berat Ikan Per Kuartal", "xlabel": "Kuartal", "palette": "YlOrBr", "rot": 0
        })

    tech_col = next((c for c in ["teknik_penangkapan", "jenis_alat_tangkap", "alat_tangkap"] if c in df_model.columns), None)
    if tech_col:
        plot_tasks.append({
            "type": "cat", "x": tech_col, "y": "berat_kg", "data": df_model,
            "title": f"Berat Ikan Per {tech_col.replace('_', ' ').title()}", "xlabel": tech_col.replace('_', ' ').title(), "palette": "Oranges", "rot": 30
        })

    has_sst = "sst" in df_model.columns and df_model["sst"].nunique() > 1
    if has_sst:
        df_sst = df_model.copy()
        try:
            df_sst["sst_bin"] = pd.qcut(df_sst["sst"], q=4, duplicates="drop")
            df_sst["sst_lbl"] = df_sst["sst_bin"].apply(lambda interval: f"{fmt_num(interval.left, 1)}–{fmt_num(interval.right, 1)} °C")
        except Exception:
            df_sst["sst_bin"] = pd.cut(df_sst["sst"], bins=4)
            df_sst["sst_lbl"] = df_sst["sst_bin"].apply(lambda interval: f"{fmt_num(interval.left, 1)}–{fmt_num(interval.right, 1)} °C")
        plot_tasks.append({
            "type": "cat", "x": "sst_lbl", "y": "berat_kg", "data": df_sst,
            "title": "Berat Ikan Per Rentang Suhu (SST)", "xlabel": "Rentang Suhu (°C)", "palette": "Reds", "rot": 30
        })

    has_chl = "chl_a" in df_model.columns and df_model["chl_a"].nunique() > 1
    if has_chl:
        df_chl = df_model.copy()
        try:
            df_chl["chl_bin"] = pd.qcut(df_chl["chl_a"], q=4, duplicates="drop")
            df_chl["chl_lbl"] = df_chl["chl_bin"].apply(lambda interval: f"{fmt_num(interval.left, 2)}–{fmt_num(interval.right, 2)} mg/m³")
        except Exception:
            df_chl["chl_bin"] = pd.cut(df_chl["chl_a"], bins=4)
            df_chl["chl_lbl"] = df_chl["chl_bin"].apply(lambda interval: f"{fmt_num(interval.left, 2)}–{fmt_num(interval.right, 2)} mg/m³")
        plot_tasks.append({
            "type": "cat", "x": "chl_lbl", "y": "berat_kg", "data": df_chl,
            "title": "Berat Ikan Per Rentang Klorofil-a", "xlabel": "Rentang Klorofil-a (mg/m³)", "palette": "Purples", "rot": 30
        })

    other_nums = [c for c in num_list if c != "berat_kg"]
    for c in ["sst", "chl_a"]:
        if c in df_model.columns and c not in other_nums:
            other_nums.append(c)
    for col_n in other_nums:
        plot_tasks.append({
            "type": "num", "y": col_n, "data": df_model,
            "title": f"Boxplot {col_n}", "ylabel": col_n
        })

    if plot_tasks:
        n_plots = len(plot_tasks)
        cols_per_row = 4
        rows_other = int(np.ceil(n_plots / cols_per_row))
        fig_comb, axes_comb = plt.subplots(rows_other, cols_per_row, figsize=(16, 4.2 * rows_other))
        axes_flat = axes_comb.flatten() if n_plots > 1 else [axes_comb]

        for idx, task in enumerate(plot_tasks):
            ax = axes_flat[idx]
            if task["type"] == "cat":
                sns.boxplot(x=task["x"], y=task["y"], data=task["data"], ax=ax, palette=task["palette"])
                ax.set_title(task["title"], fontsize=9, fontweight="bold")
                ax.set_xlabel(task["xlabel"], fontsize=8)
                ax.set_ylabel("Berat (kg)", fontsize=8)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=task["rot"], ha="right" if task["rot"]>0 else "center", fontsize=8)
            elif task["type"] == "num":
                sns.boxplot(y=task["data"][task["y"]], ax=ax, color="#C08B5C")
                ax.set_title(task["title"], fontsize=9, fontweight="bold")
                ax.set_ylabel(task["ylabel"], fontsize=8)

        for i in range(n_plots, len(axes_flat)):
            fig_comb.delaxes(axes_flat[i])

        plt.tight_layout()
        st.pyplot(fig_comb)
        plt.close(fig_comb)


    st.markdown("---")

    st.subheader("1. Uji Normalitas (berat_kg)")
    target_data = df_model["berat_kg"].dropna()

    if len(target_data) > 5000:
        stat_val, p_val = stats.normaltest(target_data)
        test_name = "D'Agostino-Pearson"
    else:
        stat_val, p_val = stats.shapiro(target_data)
        test_name = "Shapiro-Wilk"

    col_norm1, col_norm2 = st.columns(2)
    col_norm1.metric(f"Statistik Test ({test_name})", fmt_num(stat_val, 4))
    col_norm2.metric("p-value", f"{p_val:.4e}".replace(".", ","))

    p_val_str = f"{p_val:.4e}".replace(".", ",")
    if p_val < 0.05:
        norm_kesimpulan = (
            "Data `berat_kg` tidak terdistribusi normal (p-value < 0,05)."
            " Kondisi ini wajar untuk data perikanan dan mendukung penggunaan"
            " GLM/GAM (Poisson, Negative Binomial, Tweedie)."
        )
        st.info(f"**Kesimpulan Normalitas:** {norm_kesimpulan}")
    else:
        norm_kesimpulan = (
            "Data `berat_kg` terdistribusi normal (p-value >= 0,05)."
        )
        st.info(f"**Kesimpulan Normalitas:** {norm_kesimpulan}")

    norm_info = {
        "test_name": test_name,
        "stat_val": stat_val,
        "p_val": p_val_str,
        "kesimpulan": norm_kesimpulan,
    }

    fig_norm, (ax_freq, ax_dens, ax_qq) = plt.subplots(1, 3, figsize=(16, 4.5))

    min_val = np.floor(target_data.min()) if len(target_data) > 0 else 0
    max_val = np.ceil(target_data.max()) if len(target_data) > 0 else 1
    bins_1kg = np.arange(min_val, max_val + 2, 1)

    sns.histplot(
        target_data, bins=bins_1kg, ax=ax_freq, color="#1ABC9C", stat="count"
    )
    ax_freq.set_title("Plot Frekuensi", fontweight="bold")
    ax_freq.set_xlabel("berat_kg")
    ax_freq.set_ylabel("Frekuensi")

    sns.histplot(
        target_data, kde=True, ax=ax_dens, color="#0E4C92", stat="density"
    )
    ax_dens.set_title("Plot Densitas (berat_kg)", fontweight="bold")
    ax_dens.set_xlabel("berat_kg")
    ax_dens.set_ylabel("Density")

    stats.probplot(target_data, dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_color("#0E4C92")
    ax_qq.get_lines()[0].set_markersize(4)
    ax_qq.get_lines()[1].set_color("red")
    ax_qq.set_title("Q-Q Plot (berat_kg)", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig_norm)
    plt.close(fig_norm)

    st.markdown("---")

    st.subheader("2. Uji Heterogenitas Varians (Levene's Test)")
    het_results = []
    for cat in valid_cats:
        groups = [
            group["berat_kg"].values
            for _, group in df_model.groupby(cat)
            if len(group["berat_kg"]) > 1
        ]
        if len(groups) > 1:
            stat_l, p_l = stats.levene(*groups)
            het_results.append({
                "Variabel Independen": cat,
                "Jumlah Kategori": fmt_int(len(groups)),
                "Statistik Levene": fmt_num(stat_l, 4),
                "p-value": f"{p_l:.4e}".replace(".", ","),
                "Status Varians": (
                    "Heterogen (p < 0,05)"
                    if p_l < 0.05
                    else "Homogen (p >= 0,05)"
                ),
            })

    if het_results:
        df_het = pd.DataFrame(het_results)
        col_het, _ = st.columns([2.5, 1])
        with col_het:
            st.dataframe(df_het, use_container_width=False, hide_index=True)
    else:
        st.warning(
            "Tidak ada variabel kategorikal valid untuk diuji heterogenitasnya."
        )

    st.markdown("---")

    st.subheader("3. Uji Multikolinearitas (Variance Inflation Factor - VIF)")
    try:
        rhs_formula = formula_glm.split("~")[1].strip()
        X_mat = dmatrix(rhs_formula, data=df_model, return_type="dataframe")

        vif_data = []
        for i in range(X_mat.shape[1]):
            col_name = X_mat.columns[i]
            if col_name != "Intercept":
                vif_val = variance_inflation_factor(X_mat.values, i)
                vif_data.append({
                    "Prediktor / Term": col_name,
                    "VIF": fmt_num(vif_val, 2),
                    "Keterangan Multikolinearitas": (
                        "Tinggi (VIF > 10)"
                        if vif_val > 10
                        else (
                            "Sedang (VIF 5–10)"
                            if vif_val > 5
                            else "Rendah / Bebas (VIF < 5)"
                        )
                    ),
                })

        df_vif = pd.DataFrame(vif_data)
        col_vif, _ = st.columns([2.5, 1])
        with col_vif:
            st.dataframe(df_vif, use_container_width=False, hide_index=True)
    except Exception as e:
        st.error(f"Gagal menghitung VIF: {e}")

    st.markdown("---")

    st.subheader("4. Heatmap Korelasi Antar Variabel Numerik")
    try:
        corr_cols = list(
            dict.fromkeys(
                ["berat_kg"]
                + valid_nums
                + ([effort_col] if effort_col else [])
            )
        )
        corr_cols = [c for c in corr_cols if c in df_model.columns]
        df_corr = df_model[corr_cols].corr(method="pearson")

        fig_corr, ax_corr = plt.subplots(
            figsize=(max(5, 0.9 * len(corr_cols)), max(4, 0.8 * len(corr_cols)))
        )
        sns.heatmap(
            df_corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Koefisien Korelasi (r)"},
            ax=ax_corr,
        )
        ax_corr.set_title(
            "Matriks Korelasi Pearson Antar Variabel Numerik",
            fontsize=10,
            fontweight="bold",
        )
        plt.tight_layout()

        col_corr, _ = st.columns([2.2, 1])
        with col_corr:
            st.pyplot(fig_corr)
        plt.close(fig_corr)

        corr_pairs = (
            df_corr.where(
                np.triu(np.ones(df_corr.shape), k=1).astype(bool)
            )
            .stack()
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        top_corr_rows = []
        for (var_a, var_b), r_val in corr_pairs.head(5).items():
            top_corr_rows.append({
                "Pasangan Variabel": f"{var_a} — {var_b}",
                "Koefisien Korelasi (r)": fmt_num(r_val, 3),
                "Kekuatan Hubungan": (
                    "Sangat Kuat (|r| > 0,8)"
                    if abs(r_val) > 0.8
                    else (
                        "Kuat (|r| 0,6–0,8)"
                        if abs(r_val) > 0.6
                        else (
                            "Sedang (|r| 0,4–0,6)"
                            if abs(r_val) > 0.4
                            else "Lemah (|r| < 0,4)"
                        )
                    )
                ),
            })
        if top_corr_rows:
            st.markdown("**5 Pasangan Variabel dengan Korelasi Absolut Tertinggi**")
            col_topcorr, _ = st.columns([2.2, 1])
            with col_topcorr:
                st.dataframe(
                    pd.DataFrame(top_corr_rows),
                    use_container_width=False,
                    hide_index=True,
                )

        st.caption(
            "Nilai mendekati **+1** menunjukkan korelasi positif kuat, mendekati **-1** korelasi negatif kuat,"
            " dan mendekati **0** berarti tidak ada hubungan linear yang berarti. Pasangan variabel dengan korelasi"
            " absolut tinggi (|r| > 0,8) sebaiknya dikonfirmasi ulang dengan nilai VIF pada tabel di atas, karena"
            " berpotensi menimbulkan multikolinearitas dalam model."
        )
    except Exception as e:
        st.error(f"Gagal membuat heatmap korelasi: {e}")

# --- TAB 1: EVALUASI MODEL ---
with tab1:
    st.subheader("Seleksi Model (Backward Elimination berbasis AIC)")
    with st.expander(
        "Panduan membaca hasil seleksi model", expanded=False
    ):
        st.markdown("""
        Langkah ini mereplikasi proses `drop1()` / *stepwise regression* pada buku pedoman standarisasi CPUE.
        Model global (seluruh variabel kandidat) dievaluasi menggunakan GLM Poisson, lalu pada setiap iterasi
        dicoba menghapus satu variabel — variabel yang jika dihapus justru **menurunkan AIC** akan dibuang secara
        permanen dari model. Proses berulang sampai tidak ada lagi variabel yang jika dihapus menurunkan AIC.
        Variabel yang tersisa (final) inilah yang dipakai untuk seluruh model (GLM Poisson, Negative Binomial,
        Tweedie, dan GAM) pada tahapan analisis berikutnya.
        """)

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        st.markdown("**Variabel Dipertahankan (Model Final)**")
        if kept_labels:
            st.success(", ".join(kept_labels))
        else:
            st.warning("Tidak ada variabel yang dipertahankan.")
    with col_sel2:
        st.markdown("**Variabel Dihapus (Tidak Signifikan / Menaikkan AIC)**")
        if dropped_labels:
            st.error(", ".join(dropped_labels))
        else:
            st.info("Tidak ada variabel yang dihapus — seluruh variabel kandidat dipertahankan.")

    if not selection_log_df.empty:
        st.markdown("**Rincian Iterasi Seleksi Model**")
        col_sellog, _ = st.columns([3, 1])
        with col_sellog:
            st.dataframe(selection_log_df, use_container_width=False, hide_index=True)

    st.markdown("---")
    st.subheader("Ringkasan Perbandingan Model")

    best_row_info = metrics_df[metrics_df["Model"] == best_model_name].iloc[0]

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">Model Terbaik Terpilih</div>
<div style="font-size: 13px; font-weight: bold; color: #0E4C92; line-height: 1.3; word-wrap: break-word;">{best_model_name}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">Rasio Overdispersi Model Terpilih</div>
<div style="font-size: 18px; font-weight: bold; color: #0E4C92; line-height: 1.3;">{fmt_num(best_row_info["Overdispersion_Ratio"], 2)}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">AIC Model Terpilih</div>
<div style="font-size: 18px; font-weight: bold; color: #0E4C92; line-height: 1.3;">{fmt_num(best_row_info["AIC"], 2)}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">Total Sampel Valid</div>
<div style="font-size: 18px; font-weight: bold; color: #0E4C92; line-height: 1.3;">{fmt_int(len(df_model))} Data</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "**Tabel Perbandingan Kinerja Model (Diurutkan berdasarkan Rasio Overdispersi & AIC)**"
    )
    metrics_display = metrics_df.copy()
    metrics_display["AIC"] = metrics_display["AIC"].apply(
        lambda x: fmt_num(x, 2)
    )
    metrics_display["Deviance"] = metrics_display["Deviance"].apply(
        lambda x: fmt_num(x, 2)
    )
    metrics_display["Null_Deviance"] = metrics_display["Null_Deviance"].apply(
        lambda x: fmt_num(x, 2)
    )
    metrics_display["Pseudo_R2"] = metrics_display["Pseudo_R2"].apply(
        lambda x: fmt_num(x, 4)
    )
    metrics_display["Delta_AIC"] = metrics_display["Delta_AIC"].apply(
        lambda x: fmt_num(x, 2)
    )
    metrics_display["Overdispersion_Ratio"] = metrics_display[
        "Overdispersion_Ratio"
    ].apply(lambda x: fmt_num(x, 2))
    metrics_display["N"] = metrics_display["N"].apply(fmt_int)

    disp_cols = [
        "Model",
        "Overdispersion_Ratio",
        "AIC",
        "Pseudo_R2",
        "Deviance",
        "Null_Deviance",
        "Delta_AIC",
        "N",
    ]
    metrics_display = metrics_display[disp_cols]

    column_cfg = {
        "Model": st.column_config.Column("Model", width=220),
        "Overdispersion_Ratio": st.column_config.Column("Overdispersion_Ratio", width=160),
        "AIC": st.column_config.Column("AIC", width=110),
        "Pseudo_R2": st.column_config.Column("Pseudo_R2", width=110),
        "Deviance": st.column_config.Column("Deviance", width=120),
        "Null_Deviance": st.column_config.Column("Null_Deviance", width=120),
        "Delta_AIC": st.column_config.Column("Delta_AIC", width=100),
        "N": st.column_config.Column("N", width=90),
    }

    st.dataframe(
        metrics_display,
        use_container_width=False,
        hide_index=True,
        column_config=column_cfg,
    )

    st.info(
        f"Model **{best_model_name}** dipilih"
        " sebagai model terbaik berdasarkan kriteria **Rasio Overdispersi terendah**"
        f" ({fmt_num(best_row_info['Overdispersion_Ratio'], 2)}) dan **AIC terendah**"
        f" ({fmt_num(best_row_info['AIC'], 2)})."
    )

    st.markdown("---")

    st.markdown("**Tabel Overdispersion Ratio Seluruh Model**")
    disp_rows = []
    for name, mod in models.items():
        disp_ratio = mod.pearson_chi2 / mod.df_resid
        if disp_ratio > 1.5:
            status = "Overdispersion Tinggi (Rasio > 1,5)"
        elif disp_ratio < 0.8:
            status = "Underdispersion (Rasio < 0,8)"
        else:
            status = "Ideal / Teratasi (Rasio ≈ 1,0)"

        disp_rows.append({
            "Model": name,
            "Pearson Chi2": fmt_num(mod.pearson_chi2, 2),
            "df Resid": fmt_int(mod.df_resid),
            "Overdispersion Ratio": fmt_num(disp_ratio, 2),
            "Status Evaluasi Varians": status,
        })

    df_disp_table = pd.DataFrame(disp_rows)
    col_disp, _ = st.columns([3.5, 1])
    with col_disp:
        st.dataframe(df_disp_table, use_container_width=False, hide_index=True)

    st.markdown("---")
    st.subheader("Residual Plot Model")

    with st.expander("Panduan membaca Residual Plot Model", expanded=False):
        st.markdown("""
        Plot residual digunakan untuk memeriksa keakuratan prediksi dan apakah asumsi model telah terpenuhi.
        
        * **Sumbu X (Fitted Values):** Nilai estimasi atau prediksi hasil tangkapan yang dihasilkan oleh model.
        * **Sumbu Y (Response Residuals):** Sisaan (selisih) antara nilai hasil tangkapan aktual dengan nilai prediksi model.
        * **Garis Putus-putus Merah (Nol):** Titik ideal di mana tidak ada selisih (prediksi sama persis dengan aktual).
        * **Pola yang Baik / Ideal:** Titik-titik data tersebar secara acak dan merata di atas maupun di bawah garis merah, tanpa membentuk pola yang jelas.
        * **Indikasi Masalah Model:** Jika titik-titik membentuk pola tertentu seperti *corong* (melebar atau menyempit searah sumbu X) atau pola *lengkungan*, hal ini menandakan model belum sepenuhnya menangkap varians data secara sempurna (misalnya terdapat efek heteroskedastisitas atau efek non-linear yang tidak terjelaskan).
        """)

    fig_res, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_list = axes.flatten()

    for idx, (name, mod) in enumerate(models.items()):
        if idx >= 4:
            break
        axes_list[idx].scatter(
            mod.fittedvalues,
            mod.resid_response,
            alpha=0.3,
            s=12,
            color="#0E4C92",
        )
        axes_list[idx].axhline(y=0, linestyle="--", linewidth=1, color="red")
        axes_list[idx].set_title(
            f"Residuals: {name}", fontsize=10, fontweight="bold"
        )
        axes_list[idx].set_xlabel("Fitted Values", fontsize=8)
        axes_list[idx].set_ylabel("Response Residuals", fontsize=8)

    for i in range(len(models), 4):
        fig_res.delaxes(axes_list[i])

    plt.tight_layout()
    st.pyplot(fig_res)

    # ---------------------------------------------------------
    # RINCIAN KOEFISIEN DAN UJI DISPERSI SELURUH MODEL
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Rincian Koefisien dan Pengujian Dispersi Seluruh Model")

    for m_name in valid_model_list:
        mod_detail = models[m_name]
        is_best = (m_name == best_model_name)
        expander_title = f"Rincian Model: {m_name} " + ("(Model Terpilih)" if is_best else "")

        with st.expander(expander_title, expanded=is_best):
            # 1. Perhitungan Uji Overdispersi
            disp_ratio_val = mod_detail.pearson_chi2 / mod_detail.df_resid
            p_val_disp = 1 - stats.chi2.cdf(mod_detail.pearson_chi2, mod_detail.df_resid)

            st.markdown(f"**Pengujian Overdispersi ({m_name})**")
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Rasio Dispersi", fmt_num(disp_ratio_val, 3))
            col_d2.metric("Pearson Chi-Squared", fmt_num(mod_detail.pearson_chi2, 2))
            col_d3.metric("p-value Dispersi", f"{p_val_disp:.4e}".replace(".", ","))

            if disp_ratio_val > 1.5:
                st.warning(f"**Overdispersi Terdeteksi:** Rasio dispersi ({fmt_num(disp_ratio_val, 2)}) > 1,5 dengan p-value < 0,05.")
            else:
                st.success(f"**Dispersi Ideal:** Rasio dispersi ({fmt_num(disp_ratio_val, 2)}) berada dalam batas wajar.")

            # 2. Ringkasan Deviance & AIC
            null_df_val = int(mod_detail.df_model + mod_detail.df_resid)
            st.markdown("**Ringkasan Deviance dan AIC Model**")
            st.write(f"- **Null Deviance:** `{fmt_num(mod_detail.null_deviance, 2)}` pada `{null_df_val}` derajat bebas")
            st.write(f"- **Residual Deviance:** `{fmt_num(mod_detail.deviance, 2)}` pada `{fmt_int(mod_detail.df_resid)}` derajat bebas")
            st.write(f"- **AIC:** `{fmt_num(mod_detail.aic, 2)}`")

            # 3. Tampilan Teks Ringkasan Rinci
            st.markdown("**Tabel Koefisien Lengkap**")
            st.text(mod_detail.summary().as_text())

# --- TAB 2: EFEK PARSIAL DINAMIS DENGAN PANDUAN ---
with tab2:
    col_sel_t2, _ = st.columns([2, 1])
    with col_sel_t2:
        selected_model_name_t2 = st.selectbox(
            "Pilih Model untuk Menampilkan Plot Efek Parsial:",
            options=valid_model_list,
            index=0,
            key="select_model_tab2",
        )

    model_tab2 = models[selected_model_name_t2]
    st.subheader(f"Plot Efek Parsial Parameter ({selected_model_name_t2})")

    # PANDUAN MEMBACA PLOT EFEK PARSIAL
    with st.expander("Panduan membaca Plot Efek Parsial", expanded=False):
        st.markdown("""
        Plot efek parsial menggambarkan kontribusi isolasi dari masing-masing variabel terhadap hasil tangkapan (CPUE) dengan mengasumsikan variabel lainnya konstan.
        
        * **Sumbu Y (Partial Effect - Skala Log):**
          * **Nilai > 0:** Variabel memberikan pengaruh positif (meningkatkan CPUE terstandar).
          * **Nilai = 0 (Garis Merah):** Variabel bersifat netral / tidak mengubah CPUE.
          * **Nilai < 0:** Variabel memberikan pengaruh negatif (menurunkan CPUE terstandar).
        * **Grafik Garis (Variabel Numerik):**
          * **Garis Solid (Hitam):** Tren arah pengaruh variabel. Jika melengkung/naik-turun, menandakan pola hubungan non-linear (GAM/Spline).
          * **Garis Putus-putus:** Selang Kepercayaan 95% (Confidence Interval). Semakin sempit rentangnya, semakin pasti estimasi dampaknya.
        * **Grafik Batang (Variabel Kategorikal):**
          * Batang di atas garis merah (0) = Kategori tersebut meningkatkan CPUE.
          * Batang di bawah garis merah (0) = Kategori tersebut menurunkan CPUE.
          * **Error Bar (Garis I):** Rentang variasi estimasi pada kategori tersebut.
        """)

    defaults = {"log_effort": 0.0}
    for c in valid_cats:
        defaults[c] = df_model[c].mode()[0]
    for c in valid_nums:
        defaults[c] = df_model[c].mean()

    def make_dummy(override_col, override_vals):
        d = {k: v for k, v in defaults.items()}
        d[override_col] = override_vals
        return pd.DataFrame(d)

    total_plots = len(valid_nums) + len(valid_cats)
    cols_per_row = 3
    rows = int(np.ceil(total_plots / cols_per_row))

    fig_grid, axes = plt.subplots(
        rows, cols_per_row, figsize=(15, max(4 * rows, 5))
    )
    axes_flat = axes.flatten() if total_plots > 1 else [axes]

    plot_idx = 0
    partial_interp_list = []

    var_label_map = {
        "abk": "Jumlah ABK",
        "panjang_kapal": "Panjang Kapal",
        "kapasitas_mesin": "Kapasitas Mesin",
        "gross_tonnage": "Gross Tonnage (GT)",
        "gt": "Gross Tonnage (GT)",
        "sst": "Suhu Permukaan Laut (SST)",
        "chl_a": "Konsentrasi Klorofil-a",
        "tahun": "Tahun Operasional",
        "bulan": "Bulan Operasional",
        "musim": "Musim Penangkapan",
        "quarter": "Kuartal",
        "teknik_penangkapan": "Teknik Penangkapan",
        "jenis_alat_tangkap": "Jenis Alat Tangkap",
        "daerah_spasial": "Daerah Spasial",
        "daerah": "Daerah Penangkapan",
    }

    for col_name in valid_nums:
        ax = axes_flat[plot_idx]
        grid = np.linspace(
            df_model[col_name].min(), df_model[col_name].max(), 150
        )
        try:
            pred = model_tab2.get_prediction(
                make_dummy(col_name, grid), transform=False
            )
            fit = pred.predicted_mean - pred.predicted_mean.mean()
            se = pred.se_mean
        except Exception:
            pred_vals = model_tab2.predict(make_dummy(col_name, grid))
            fit = np.log(np.maximum(pred_vals, 1e-6))
            fit = fit - fit.mean()
            se = np.abs(fit) * 0.1

        fit = np.asarray(fit)
        se = np.asarray(se)

        ax.plot(grid, fit, "k-", lw=1.2)
        ax.plot(grid, fit + 1.96 * se, "k--", lw=0.8)
        ax.plot(grid, fit - 1.96 * se, "k--", lw=0.8)
        ax.set_title(f"Effect: {col_name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Partial effect (log scale)", fontsize=8)
        plot_idx += 1

        v_label = var_label_map.get(
            col_name, col_name.replace("_", " ").title()
        )
        
        max_i = int(np.argmax(fit))
        min_i = int(np.argmin(fit))
        
        if 5 < max_i < (len(grid) - 5):
            desc = f"Berpengaruh non-linear (berpola cembung) dengan puncak dampak positif tertinggi pada nilai <b>{grid[max_i]:.2f}</b>, lalu menurun kembali."
        elif 5 < min_i < (len(grid) - 5):
            desc = f"Berpengaruh non-linear (berpola cekung) dengan titik terendah pada nilai <b>{grid[min_i]:.2f}</b>, sebelum kembali meningkat."
        else:
            delta_eff = fit[-1] - fit[0]
            if delta_eff > 0:
                desc = f"Berpengaruh positif secara konsisten (peningkatan {v_label.lower()} mendorong kenaikan hasil tangkapan)."
            else:
                desc = f"Berpengaruh negatif secara konsisten (peningkatan {v_label.lower()} berhubungan dengan penurunan hasil tangkapan)."

        partial_interp_list.append(f"• <b>{v_label}</b>: {desc}")

    for cat_col in valid_cats:
        ax = axes_flat[plot_idx]
        uniques = sorted(
            df_model[cat_col].dropna().unique(),
            key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
        )

        if len(uniques) > 12:
            uniques = df_model[cat_col].value_counts().index[:10].tolist()

        try:
            pred = model_tab2.get_prediction(
                make_dummy(cat_col, uniques), transform=False
            )
            fit = pred.predicted_mean - pred.predicted_mean.mean()
            se = pred.se_mean
        except Exception:
            pred_vals = model_tab2.predict(make_dummy(cat_col, uniques))
            fit = np.log(np.maximum(pred_vals, 1e-6))
            fit = fit - fit.mean()
            se = np.abs(fit) * 0.1

        fit = np.asarray(fit)
        se = np.asarray(se)

        x_pos = np.arange(len(uniques))

        ax.bar(
            x_pos,
            fit,
            yerr=1.96 * se,
            color="grey",
            edgecolor="black",
            capsize=4,
            alpha=0.7,
        )
        ax.axhline(0, color="red", ls="--", lw=0.8)
        ax.set_xticks(x_pos)

        if cat_col == "bulan":
            x_labels = [month_map.get(str(u), str(u)) for u in uniques]
        else:
            x_labels = [str(u) for u in uniques]

        ax.set_xticklabels(
            x_labels, rotation=30, ha="right", fontsize=8
        )
        ax.set_title(f"Effect: {cat_col}", fontsize=10, fontweight="bold")
        plot_idx += 1

        v_label = var_label_map.get(
            cat_col, cat_col.replace("_", " ").title()
        )
        max_i = int(np.argmax(fit))
        min_i = int(np.argmin(fit))
        max_c = (
            month_map.get(str(uniques[max_i]), str(uniques[max_i]))
            if cat_col == "bulan"
            else str(uniques[max_i])
        )
        min_c = (
            month_map.get(str(uniques[min_i]), str(uniques[min_i]))
            if cat_col == "bulan"
            else str(uniques[min_i])
        )

        desc = (
            f"Tingkat hasil tangkapan paling tinggi ditemukan pada"
            f" <b>{max_c}</b>, sedangkan yang terendah tercatat pada <b>{min_c}</b>."
        )
        partial_interp_list.append(f"• <b>{v_label}</b>: {desc}")

    for i in range(plot_idx, len(axes_flat)):
        fig_grid.delaxes(axes_flat[i])

    plt.tight_layout()
    st.pyplot(fig_grid)

    partial_interp_html = "<br>".join(partial_interp_list)

    st.markdown("---")
    st.subheader("Interpretasi Detail Efek Parsial Parameter")
    interp_clean_str = "\n\n".join([item.replace("<b>", "**").replace("</b>", "**") for item in partial_interp_list])
    st.info(f"**Rangkuman Pengaruh Parsial Variabel terhadap Hasil Tangkapan:**\n\n{interp_clean_str}")


# --- TAB 3: STANDARISASI CPUE DENGAN PANDUAN ---
with tab3:
    col_sel_t3, _ = st.columns([2, 1])
    with col_sel_t3:
        selected_model_name_t3 = st.selectbox(
            "Pilih Model untuk Hasil Standarisasi CPUE:",
            options=valid_model_list,
            index=0,
            key="select_model_tab3",
        )

    model_tab3 = models[selected_model_name_t3]
    st.subheader(f"Hasil Standarisasi CPUE ({selected_model_name_t3})")

    # PANDUAN MEMBACA STANDARISASI CPUE
    with st.expander("Panduan membaca CPUE Terstandar", expanded=False):
        st.markdown("""
        Hasil standarisasi CPUE (Marginal Means / Emmeans) menunjukkan estimasi rata-rata hasil tangkapan per unit effort yang telah dibersihkan dari efek faktor pengganggu (seperti perbedaan ukuran kapal, mesin, lokasi, dan musim).
        
        * **CPUE Standar (kg/hari):** Nilai estimasi rerata hasil tangkapan per hari memancing. Nilai ini yang digunakan sebagai indeks kelimpahan stok ikan yang valid.
        * **SE (Standard Error):** Tingkat kesalahan standar dari estimasi CPUE.
        * **df (Degrees of Freedom):** Derajat bebas residual dari pemodelan statistik.
        * **Tren Garis & Titik:** Menunjukkan arah perkembangan stok (apakah cenderung naik, stabil, atau mengalami penurunan dari tahun ke tahun/bulan ke bulan).
        * **Pita / Area Transparan (Shading Area):** Menunjukkan batas selang kepercayaan 95% (Lower CI hingga Upper CI). Jika pita menyempit, estimasi CPUE pada periode tersebut memiliki tingkat presisi yang tinggi.
        """)

    grid_yr_display = None
    grid_tm_table = None
    fig_yr = None
    fig_mo = None

    # 1. Standarisasi Tahunan via Emmeans
    if "tahun" in valid_cats:
        grid_yr = calculate_emmeans_proportional(
            model_tab3,
            "tahun",
            df_model,
            valid_cats,
            valid_nums,
            "log_effort",
            1.0,
        )

        grid_yr_display = grid_yr.copy()
        grid_yr_display["CPUE_std (kg/hari)"] = grid_yr_display[
            "CPUE_std (kg/hari)"
        ].apply(lambda x: fmt_num(x, 2))
        grid_yr_display["SE"] = grid_yr_display["SE"].apply(
            lambda x: fmt_num(x, 4)
        )
        grid_yr_display["df"] = grid_yr_display["df"].apply(
            lambda x: fmt_num(x, 2)
        )
        grid_yr_display["Lower CI"] = grid_yr_display["Lower CI"].apply(
            lambda x: fmt_num(x, 2)
        )
        grid_yr_display["Upper CI"] = grid_yr_display["Upper CI"].apply(
            lambda x: fmt_num(x, 2)
        )

        col_t1, col_t2 = st.columns([1, 1.5])
        with col_t1:
            st.markdown("**CPUE Standar Tahunan**")
            st.dataframe(grid_yr_display, use_container_width=False, hide_index=True)

        with col_t2:
            fig_yr, ax_yr = plt.subplots(figsize=(7, 3.5))
            years = list(grid_yr["tahun"])
            x_raw = np.arange(len(years))

            if len(years) > 2:
                x_smooth = np.linspace(x_raw.min(), x_raw.max(), 300)
                k_deg = min(3, len(years) - 1)

                spl_m = make_interp_spline(
                    x_raw, grid_yr["CPUE_std (kg/hari)"], k=k_deg
                )
                spl_l = make_interp_spline(x_raw, grid_yr["Lower CI"], k=k_deg)
                spl_u = make_interp_spline(x_raw, grid_yr["Upper CI"], k=k_deg)

                ax_yr.fill_between(
                    x_smooth,
                    spl_l(x_smooth),
                    spl_u(x_smooth),
                    color="#E67E22",
                    alpha=0.18,
                    edgecolor="none",
                )
                ax_yr.plot(
                    x_smooth, spl_m(x_smooth), color="#E67E22", linewidth=1.5
                )
                ax_yr.scatter(
                    x_raw,
                    grid_yr["CPUE_std (kg/hari)"],
                    color="#D35400",
                    s=20,
                    zorder=5,
                )
            else:
                ax_yr.fill_between(
                    x_raw,
                    grid_yr["Lower CI"],
                    grid_yr["Upper CI"],
                    color="#E67E22",
                    alpha=0.18,
                    edgecolor="none",
                )
                ax_yr.plot(
                    x_raw,
                    grid_yr["CPUE_std (kg/hari)"],
                    color="#E67E22",
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                )

            ax_yr.set_xticks(x_raw)
            ax_yr.set_xticklabels(years)
            ax_yr.set_xlabel("Tahun")
            ax_yr.set_ylabel("CPUE Standar (kg/hari)")
            ax_yr.set_title("Tren CPUE Standar Tahunan", fontweight="bold")
            st.pyplot(fig_yr)

        max_yr_row = grid_yr.loc[grid_yr["CPUE_std (kg/hari)"].idxmax()]
        min_yr_row = grid_yr.loc[grid_yr["CPUE_std (kg/hari)"].idxmin()]
        st.info(
            f"**Interpretasi CPUE Tahunan:** Kelimpahan relatif CPUE terstandarisasi tertinggi terjadi pada tahun **{max_yr_row['tahun']}** "
            f"sebesar **{fmt_num(max_yr_row['CPUE_std (kg/hari)'], 2)} kg/hari**. Sebaliknya, tingkat CPUE terendah berada pada tahun **{min_yr_row['tahun']}** "
            f"sebesar **{fmt_num(min_yr_row['CPUE_std (kg/hari)'], 2)} kg/hari**."
        )

        st.markdown("---")

    # 2. Standarisasi Musiman / Bulanan via Emmeans
    time_cat = next(
        (c for c in ["bulan", "musim", "quarter"] if c in valid_cats), None
    )
    if time_cat:
        grid_tm = calculate_emmeans_proportional(
            model_tab3,
            time_cat,
            df_model,
            valid_cats,
            valid_nums,
            "log_effort",
            1.0,
        )

        grid_tm_display = grid_tm.copy()
        if time_cat == "bulan":
            grid_tm_display["bulan"] = grid_tm_display["bulan"].apply(
                lambda x: month_map.get(str(x), str(x))
            )
            x_labels = [
                month_map.get(str(t), str(t)) for t in grid_tm[time_cat]
            ]
        else:
            x_labels = [str(t) for t in grid_tm[time_cat]]

        grid_tm_table = grid_tm_display[
            [time_cat, "CPUE_std (kg/hari)", "SE", "df", "Lower CI", "Upper CI"]
        ].copy()
        grid_tm_table["CPUE_std (kg/hari)"] = grid_tm_table[
            "CPUE_std (kg/hari)"
        ].apply(lambda x: fmt_num(x, 2))
        grid_tm_table["SE"] = grid_tm_table["SE"].apply(
            lambda x: fmt_num(x, 4)
        )
        grid_tm_table["df"] = grid_tm_table["df"].apply(
            lambda x: fmt_num(x, 2)
        )
        grid_tm_table["Lower CI"] = grid_tm_table["Lower CI"].apply(
            lambda x: fmt_num(x, 2)
        )
        grid_tm_table["Upper CI"] = grid_tm_table["Upper CI"].apply(
            lambda x: fmt_num(x, 2)
        )

        col_b1, col_b2 = st.columns([1, 1.5])
        with col_b1:
            st.markdown(f"**CPUE Standar Berdasarkan ({time_cat.title()})**")
            st.dataframe(grid_tm_table, use_container_width=False, hide_index=True)

        with col_b2:
            fig_mo, ax_mo = plt.subplots(figsize=(7, 3.5))
            time_units = list(grid_tm[time_cat])
            x_raw = np.arange(len(time_units))

            if len(time_units) > 2:
                x_smooth = np.linspace(x_raw.min(), x_raw.max(), 300)
                k_deg = min(3, len(time_units) - 1)

                spl_m = make_interp_spline(
                    x_raw, grid_tm["CPUE_std (kg/hari)"], k=k_deg
                )
                spl_l = make_interp_spline(x_raw, grid_tm["Lower CI"], k=k_deg)
                spl_u = make_interp_spline(x_raw, grid_tm["Upper CI"], k=k_deg)

                ax_mo.fill_between(
                    x_smooth,
                    spl_l(x_smooth),
                    spl_u(x_smooth),
                    color="#E67E22",
                    alpha=0.18,
                    edgecolor="none",
                )
                ax_mo.plot(
                    x_smooth, spl_m(x_smooth), color="#E67E22", linewidth=1.5
                )
                ax_mo.scatter(
                    x_raw,
                    grid_tm["CPUE_std (kg/hari)"],
                    color="#D35400",
                    s=20,
                    zorder=5,
                )
            else:
                ax_mo.fill_between(
                    x_raw,
                    grid_tm["Lower CI"],
                    grid_tm["Upper CI"],
                    color="#E67E22",
                    alpha=0.18,
                    edgecolor="none",
                )
                ax_mo.plot(
                    x_raw,
                    grid_tm["CPUE_std (kg/hari)"],
                    color="#E67E22",
                    marker="s",
                    markersize=4,
                    linewidth=1.5,
                )

            ax_mo.set_xticks(x_raw)
            ax_mo.set_xticklabels(x_labels)
            ax_mo.set_xlabel(time_cat.title())
            ax_mo.set_ylabel("CPUE Standar (kg/hari)")
            ax_mo.set_title(
                f"Pola Standar CPUE Berdasarkan {time_cat.title()}",
                fontweight="bold",
            )
            st.pyplot(fig_mo)

        max_tm_row = grid_tm.loc[grid_tm["CPUE_std (kg/hari)"].idxmax()]
        min_tm_row = grid_tm.loc[grid_tm["CPUE_std (kg/hari)"].idxmin()]
        lbl_max = month_map.get(str(max_tm_row[time_cat]), str(max_tm_row[time_cat])) if time_cat == "bulan" else str(max_tm_row[time_cat])
        lbl_min = month_map.get(str(min_tm_row[time_cat]), str(min_tm_row[time_cat])) if time_cat == "bulan" else str(min_tm_row[time_cat])

        st.info(
            f"**Interpretasi CPUE {time_cat.title()}:** Puncak musim penangkapan terjadi pada **{lbl_max}** "
            f"dengan nilai rata-rata CPUE terstandar sebesar **{fmt_num(max_tm_row['CPUE_std (kg/hari)'], 2)} kg/hari**, "
            f"sedangkan periode terendah berada pada **{lbl_min}** ({fmt_num(min_tm_row['CPUE_std (kg/hari)'], 2)} kg/hari)."
        )

    img_res_b64 = fig_to_base64(fig_res) if fig_res else None
    img_grid_b64 = fig_to_base64(fig_grid) if fig_grid else None
    img_yr_b64 = fig_to_base64(fig_yr) if fig_yr else None
    img_tm_b64 = fig_to_base64(fig_mo) if fig_mo else None

    if fig_res:
        plt.close(fig_res)
    if fig_grid:
        plt.close(fig_grid)
    if fig_yr:
        plt.close(fig_yr)
    if fig_mo:
        plt.close(fig_mo)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if "tahun" in valid_cats and grid_yr_display is not None:
            grid_yr_display.to_excel(
                writer, sheet_name="CPUE_Tahunan", index=False
            )
        if time_cat and grid_tm_table is not None:
            grid_tm_table.to_excel(
                writer, sheet_name=f"CPUE_{time_cat}", index=False
            )

    html_report = generate_html_report(
        best_model_name,
        metrics_df,
        df_disp_table,
        df_stat_summary,
        norm_info,
        df_het,
        df_vif,
        grid_yr_display,
        grid_tm_table if time_cat else None,
        len(df_model),
        time_cat,
        img_res_b64,
        img_grid_b64,
        img_yr_b64,
        img_tm_b64,
        partial_interp_html,
    )

    col_down1, col_down2 = st.columns(2)
    with col_down1:
        st.download_button(
            label="Download Hasil Standarisasi CPUE (Excel)",
            data=output.getvalue(),
            file_name="Hasil_Standarisasi_CPUE_YFT.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )
    with col_down2:
        st.download_button(
            label="Download Laporan Pengujian (HTML / Cetak PDF)",
            data=html_report,
            file_name="Laporan_Pengujian_Standarisasi_CPUE.html",
            mime="text/html",
        )

# =========================================================
# 6. FOOTER APLIKASI
# =========================================================
render_footer()