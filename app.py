
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import BytesIO
import zipfile
import html


st.set_page_config(layout="wide", page_title="PSD Engineer Efficiency Analyzer")
st.title("PSD Engineer Efficiency Analyzer (AMS&NIS)")

# ---------------- Sidebar: Uploader + Metric Mode ----------------
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload Efficiency Excel File", type=["xlsx"])  # keep .xlsx

st.sidebar.subheader("Metric Mode")
metric_mode = st.sidebar.radio(
    "Choose the efficiency metric:",
    ["Calendar", "Timesheet", "Combined"],
    help=(
        "Calendar: uses Start/End elapsed hours.\n"
        "Timesheet: uses *Timesheet effort hours when available, else Calendar.\n"
        "Combined: weighted average of Calendar & Timesheet percentiles (fair + robust)."
    ),
    horizontal=False,
    key="metric_mode_toggle",
)

# --- Advanced Parameters in Sidebar ---
st.sidebar.header("Advanced Parameters")

with st.sidebar.expander("Data Cleaning & Parsing", expanded=False):
    days_threshold = st.number_input(
        "Long task days threshold", 1, 365, value=180,
        help="Tasks open longer than this (and completed) will use Update Time instead of End Time."
    )
    completed_kw = st.text_input(
        "Completed status keyword", value="completed",
        help="Phase text indicating a task is completed. Case-insensitive; substring match."
    )
    update_time_override = st.text_input(
        "Override 'Update Time' column (optional)", value="",
        help="Exact column name to use as Update Time. Leave blank to auto-detect (first column containing 'update' and 'time')."
    )
    activity_split_regex = st.text_input(
        "Activity split regex", value=r"[;,/]",
        help="Regex to split multiple activity types. Changing this affects activity counts and charts."
    )
    owner_regex = st.text_input(
        "Engineer name regex", value=r"^([A-Za-z]+(?:\s+[A-Za-z]+)+)",
        help="Regex to extract full name (first and last) from Owner field. Affects grouping per engineer."
    )

with st.sidebar.expander("Difficulty Classification", expanded=False):

    st.caption("Difficulty is a compound KPI: Task Depth (low weight) + DOC Severity + Time required vs average by Activity Type.")

    use_effort_for_diff = st.checkbox(
        "Effort hours for difficulty calculation?",
        value=True,
        help="Uncheck to use the Calendar duration instead."
    )
    # NEW weights and mode
    depth_weight = st.slider("Depth weight (low impact)", 0.0, 2.0, 0.2, step=0.05)
    doc_weight   = st.slider("DOC weight", 0.0, 3.0, 1.0, step=0.05)
    time_weight  = st.slider("Time-vs-avg weight", 0.0, 3.0, 1.0, step=0.05)
    use_log_time = st.checkbox("Use log2 for Time factors (smooth)", value=True)

    # Cap para el ratio de tiempo (anti-outliers)
    cap_ratio = st.slider("Time ratio cap (robust)", 2.0, 4.0, 2.5, 0.1,
                      help="Top for difference time vs activity; with log2, 2.5 tends to be a proper default.")


with st.sidebar.expander("Composite Difficulty thresholds", expanded=False):
    mode_cohort_thresh = st.radio(
        "Threshold mode",
        ["Fixed", "Cohort-adaptive (percentiles)"],
        index=1,
        help="How to convert Composite Difficulty Score into classes."
    )

    # Fixed (defaults como los actuales)
    fixed_easy_max = st.number_input("Fixed: Easy < ", 0.0, 10.0, value=0.75, step=0.05)
    fixed_med_max  = st.number_input("Fixed: Medium < ", 0.0, 10.0, value=1.50, step=0.05)
    fixed_hard_max = st.number_input("Fixed: Hard < ", 0.0, 10.0, value=2.50, step=0.05)

    # Cohort-adaptive percentiles (usados si eliges el modo cohort)
    pct_easy = st.slider("Cohort percentile for Easy/Medium boundary", 1, 99, 30, step=1)
    pct_med  = st.slider("Cohort percentile for Medium/Hard boundary", 1, 99, 70, step=1)
    pct_hard = st.slider("Cohort percentile for Hard/Very Hard boundary", 1, 99, 90, step=1)

# === Sidebar: Difficulty micro‑gating (post‑classification, ultra‑compact) ===
with st.sidebar.expander("Difficulty micro‑gating (post‑classification)", expanded=False):
    st.caption("Demote si no cumple señales mínimas. Un control global para todo (más simple y consistente).")

    # Switch global
    mg_enable = st.checkbox("Enable micro‑gating", value=True)

    # Severidad global (escala umbrales base)
    # 0.9 = más flexible, 1.0 = balanceado, 1.1–1.2 = más estricto
    strictness = st.slider("Strictness (global)", 0.9, 1.2, 1.0, 0.05)

    # Un único 'min signals' para las tres clases (1–3). Mantiene simplicidad visual.
    min_req_all = st.selectbox("Min signals (VH/Hard/Med)", [1, 2, 3], index=1,
                               help="Se requieren N señales (Tiempo, DOC, Profundidad) para mantener la clase.")

    # Toggles por clase (on/off)
    col_tog = st.columns(3)
    with col_tog[0]:
        vh_enable = st.checkbox("Gate VH", value=True, key="vh_gate_uc")
    with col_tog[1]:
        hard_enable = st.checkbox("Gate Hard", value=True, key="h_gate_uc")
    with col_tog[2]:
        med_enable = st.checkbox("Gate Medium", value=True, key="m_gate_uc")

    # ===== Presets base (no UI): thresholds por clase =====
    # Very Hard base
    _tf_vh_base, _doc_vh_base, _dep_vh_base = 1.5, 0.8, 3.0
    # Hard base
    _tf_h_base,  _doc_h_base,  _dep_h_base  = 0.6, 0.4, 2.0   # ~log2(1.5)=0.585
    # Medium base
    _tf_m_base,  _doc_m_base,  _dep_m_base  = 0.3, 0.2, 1.5   # ~log2(1.25)=0.321

    # Escalado por strictness (aplica a los tres factores)
    tf_vh  = _tf_vh_base  * strictness
    doc_vh = _doc_vh_base * strictness
    dep_vh = _dep_vh_base * strictness

    tf_h   = _tf_h_base   * strictness
    doc_h  = _doc_h_base  * strictness
    dep_h  = _dep_h_base  * strictness

    tf_m   = _tf_m_base   * strictness
    doc_m  = _doc_m_base  * strictness
    dep_m  = _dep_m_base  * strictness

    # Unificamos el "min signals" para las tres clases (compatibles con el bloque de aplicación)
    vh_req   = min_req_all
    hard_req = min_req_all
    med_req  = min_req_all


with st.sidebar.expander("Calendar vs Timesheet (Combined Mode)", expanded=False):
    cal_weight = st.slider("Calendar weight", 0.0, 1.0, 0.5, help="Weight applied to Calendar percentile in Combined score.")
    tim_weight = 1.0 - cal_weight
    scale_0_100 = st.checkbox("Scale Combined to 0–100", value=True)

with st.sidebar.expander("Individual Report Suggestions Engine", expanded=False):
    p75_cut = st.slider("High percentile (p75)", 50, 95, 75, help="Threshold used for 'high' effort/depth/variability.")
    p25_cut = st.slider("Low percentile (p25)", 5, 50, 25, help="Threshold used for 'low' effort.")
    vh_min_tasks = st.number_input("Very Hard gate: min tasks", 0, 50, 3,
                                   help="Minimum count of Very Hard tasks for VH-share suggestion.")
    vh_min_share = st.number_input("Very Hard gate: min share", 0.0, 1.0, 0.20, step=0.05,
                                   help="Minimum share of Very Hard tasks (0–1) for VH-share suggestion.")
    var_pctl = st.slider("Variability high percentile", 50, 95, 75, help="Per-activity duration variability threshold.")
    div_low_pctl = st.slider("Activity diversity low percentile", 5, 50, 25, help="Low diversity trigger threshold.")

# ---- DOC: Documented Task Complexity (name pattern override) ----
with st.sidebar.expander("DOC Task Complexity (name-based)", expanded=False):
    st.caption("Detect bracketed blocks in *Name* like \"[ ... / ... / ... ]\" and use item count × activity weight (1–10 scale) to influence depth and efficiency.")
    doc_enable = st.checkbox("Enable DOC override", value=True)
    doc_block_regex = st.text_input(
        "DOC block regex (match [xxx/yyy/zzz])",
        value=r'(\[((?:\s*[^/\[\]\n]+)(?:\s*/\s*[^/\[\]\n]+){1,})\])',
        help="Requires at least one '/' between non-empty segments, e.g. [OLT/Router/Switch]. Blocks like [SR 123] are ignored."
    )

    doc_item_sep_regex = st.text_input(
        "DOC item separator regex", value=r"\s*/\s*",
        help="Regex used to split items inside the DOC block (default '/')"
    )
    st.caption("Activity weights are loaded from the dataset below once the Excel is uploaded.")
    doc_default_weight = st.number_input("Default activity weight (1–10)", 1, 10, value=1, step=1)
    doc_complexity_to_depth = st.number_input(
        "DOC severity → depth boost factor", 0.0, 10.0, value=0.5, step=0.1,
        help="Depth points to add per severity unit. Severity = (item_count × activity_weight)/10."
    )
    doc_efficiency_gain = st.number_input(
        "DOC efficiency gain factor", 0.0, 5.0, value=0.3, step=0.05,
        help="Higher makes complex tasks count as 'faster': we divide Duration/Effort by (1 + gain × severity)."
    )
    # NEW: allow weight to impact difficulty even if no [ ... ] block is present in *Name*
    doc_weight_apply_no_block = st.checkbox(
        "Apply weight when no DOC block (assume 1 item)", value=True,
        help="If enabled, tasks without a [ ... ] block still get severity as 1 × weight/10."
    )
    st.caption("Severity scale (weight 1–10): 1–3 Low, 4–7 Medium, 8–9 Hard, 10 Very Hard.")

# --- Print CSS ---
st.markdown(
    """
    <style>
    @media print {
        header, footer, .stSidebar, .viewerBadge_container__1QSob, .stAppDeployButton { display: none !important; }
        .block-container { padding: 10px !important; }
        .print-hide { display: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helpers ---
DIFF_ORDER = ["Easy", "Medium", "Hard", "Very Hard"]

# Para tablas lindas con link
def render_interactive_link_table(df_in: pd.DataFrame, *, name_col='Task Name', url_col='Task URL',
                                  keep_cols=None, height=450, filter_key=None, show_filter=True):
    """Tabla ordenable con quick filter y link compacto '🔗' al lado del nombre."""
    df = df_in.copy()

    # Quick filter textual local
    if show_filter:
        q = st.text_input("Quick filter", value="", key=filter_key, help="Filtra por cualquier texto de la tabla.")
        if q:
            mask = df.apply(lambda r: r.astype(str).str.contains(q, case=False, na=False), axis=1).any(axis=1)
            df = df[mask]

    # Detectar URLs válidas; la celda debe contener la URL real
    has_urls = (url_col in df.columns) and df[url_col].astype(str).str.strip().str.startswith(("http://","https://")).any()
    if has_urls:
        df['Open'] = df[url_col].astype(str).str.strip()

    # Columnas base robustas
    base_cols = (keep_cols if keep_cols else list(df.columns))
    base_cols = [c for c in base_cols if c in df.columns or c == 'Open']

    # Insertar 'Open' justo después del nombre si hay URL; si no, quitarla
    if has_urls:
        # quitar duplicados preservando orden
        seen = set(); base_cols = [c for c in base_cols if not (c in seen or seen.add(c))]
        if name_col in base_cols:
            base_cols = [name_col] + [c for c in base_cols if c != name_col]
            if 'Open' not in base_cols:
                base_cols.insert(base_cols.index(name_col) + 1, 'Open')
        else:
            if 'Open' not in base_cols:
                base_cols.insert(0, 'Open')
    else:
        base_cols = [c for c in base_cols if c != 'Open']

    # Config visual: mostrar emoji chiquito (texto custom) pero mantener la URL como valor
    col_config = {}
    if has_urls:
        col_config['Open'] = st.column_config.LinkColumn(
            "Link",                 # label cortito
            display_text="🔗",   # <-- texto mostrado (emoji)
            #width="very-small",
            help="Abrir la tarea en el tracker"
        )

    # Ocultar la columna con la URL cruda
    if url_col in df.columns:
        try:
            df = df.drop(columns=[url_col])
        except Exception:
            pass

    # Selección final segura
    base_cols = [c for c in base_cols if c in df.columns]

    st.dataframe(
        df[base_cols],
        use_container_width=True,
        height=height,
        hide_index=True,
        column_config=col_config
    )


# DOC helpers
def extract_doc_info(name: str, block_regex: str, sep_regex: str):
    if not isinstance(name, str):
        return False, 0
    try:
        m = re.search(block_regex, name)
    except re.error:
        return False, 0
    if not m:
        return False, 0
    block = m.group(0)
    try:
        parts = [p.strip() for p in re.split(sep_regex, block.strip("[]")) if p.strip()]
    except re.error:
        parts = [p.strip() for p in block.strip("[]").split("/") if p.strip()]
    return True, len(parts)

def make_report_figures(eng_df):
    fig_hist = px.histogram(
        eng_df, x="Estimated Difficulty", color="Estimated Difficulty",
        category_orders={"Estimated Difficulty": DIFF_ORDER}, text_auto=True,
        title="Task Difficulty Distribution", template="plotly_white",
    )
    fig_hist.update_layout(font=dict(size=15), title_font=dict(size=20), bargap=0.2, legend_title_text="Difficulty")
    fig_hist.update_xaxes(title_text="Estimated Difficulty")
    fig_hist.update_yaxes(title_text="Task Count")
    fig_hist.update_traces(textfont_size=14, textposition="outside", cliponaxis=False)

    fig_scatter = px.scatter(
        eng_df, x="Start Time", y="Duration (hrs)", color="*Activity Type",
        title="Task Durations Over Time",
        hover_data=["*Name", "*Activity Type", "Duration (hrs)"]
    )
    fig_scatter.update_traces(marker=dict(size=9, opacity=0.75))
    fig_scatter.update_layout(template="plotly_white", font=dict(size=15), title_font=dict(size=20), legend_title_text="Activity Type")
    fig_scatter.update_xaxes(title_text="Start Time", tickformat="%Y-%m-%d")
    fig_scatter.update_yaxes(title_text="Duration (hrs)")
    return fig_hist, fig_scatter

def apply_filters(dataframe, group, product, customer):
    df_filtered = dataframe.copy()
    if group != "All":
        df_filtered = df_filtered[df_filtered["Workbook"] == group]
    if product:
        df_filtered = df_filtered[df_filtered["*Product Line"].isin(product)]
    if customer:
        df_filtered = df_filtered[df_filtered["*Customer Name"].isin(customer)]
    return df_filtered

# --- Cohort‑aware suggestion generator (uses Estimated Difficulty) ---
def generate_suggestions(eng_df: pd.DataFrame, cohort_df: pd.DataFrame, metric_mode: str = "Combined"):
    suggestions = []

    m_effort = eng_df['Effort_adj'].mean() if 'Effort_adj' in eng_df.columns else eng_df['Effort (hrs)'].mean()
    m_depth = eng_df['Effective Depth'].mean()
    very_hard_share = eng_df['Estimated Difficulty'].eq('Very Hard').mean()

    eff_col_sugg = 'Effort_adj' if 'Effort_adj' in cohort_df.columns else 'Effort (hrs)'
    coh_effort = cohort_df.groupby('Engineer')[eff_col_sugg].mean().values
    eff_label = "adjusted effort" if eff_col_sugg == 'Effort_adj' else "effort"
    
    coh_depth  = cohort_df.groupby('Engineer')['Effective Depth'].mean().values
    coh_vh     = (cohort_df.assign(vh=cohort_df['Estimated Difficulty'].eq('Very Hard')).groupby('Engineer')['vh'].mean().values)
    coh_div    = cohort_df.groupby('Engineer')['*Activity Type'].nunique().values

    p75_effort = np.nanpercentile(coh_effort, p75_cut) if coh_effort.size else np.nan
    p25_effort = np.nanpercentile(coh_effort, p25_cut) if coh_effort.size else np.nan
    p75_depth  = np.nanpercentile(coh_depth,  p75_cut) if coh_depth.size  else np.nan
    p75_vh     = np.nanpercentile(coh_vh,     p75_cut) if coh_vh.size     else np.nan
    p25_div    = np.nanpercentile(coh_div,    div_low_pctl) if coh_div.size else np.nan

    # Variability (avoid GroupBy.std(min_count=...))
    per_act_std = eng_df.groupby('*Activity Type')['Duration (hrs)'].agg(lambda s: s.std(ddof=1) if s.count() >= 2 else np.nan)
    dur_var = float(np.nanmean(per_act_std.values)) if getattr(per_act_std, 'size', 0) > 0 else np.nan
    coh_std = cohort_df.groupby(['Engineer','*Activity Type'])['Duration (hrs)'].agg(lambda s: s.std(ddof=1) if s.count() >= 2 else np.nan)
    coh_var = coh_std.groupby('Engineer').mean()
    p75_var = np.nanpercentile(coh_var.values, var_pctl) if getattr(coh_var, 'size', 0) > 0 else np.nan

    # Personalized tips
    if np.isfinite(m_effort) and np.isfinite(p75_effort) and m_effort >= p75_effort:
        suggestions.append(f"- Your average {eff_label} per task is high vs peers (>{p75_cut}th pct); consider time-boxing or pairing.")
    elif np.isfinite(m_effort) and np.isfinite(p25_effort) and m_effort <= p25_effort:
        suggestions.append(f"- Your average {eff_label} per task is low vs peers (<{p25_cut}th pct); consider mentoring or taking on more complex work.")

    if np.isfinite(m_depth) and np.isfinite(p75_depth) and m_depth >= p75_depth:
        suggestions.append("- You frequently manage deep task structures; consider knowledge-sharing sessions.")

    vh_count = int(eng_df['Estimated Difficulty'].eq('Very Hard').sum())
    if (
        np.isfinite(very_hard_share) and np.isfinite(p75_vh)
        and p75_vh > 0
        and very_hard_share > p75_vh
        and ((vh_count >= vh_min_tasks) or (very_hard_share >= vh_min_share))
    ):
        suggestions.append("- You handle a high share of very hard tasks; ensure recognition and check workload balance.")

    if eng_df['*Activity Type'].nunique() <= p25_div:
        suggestions.append("- Your activity mix is narrow; adding variety could broaden impact.")

    if np.isfinite(dur_var) and np.isfinite(p75_var) and dur_var > p75_var:
        suggestions.append("- Duration varies a lot across similar tasks; standardizing processes could help.")

    if not suggestions:
        label = {"Calendar":"speed", "Timesheet":"effort efficiency", "Combined":"balanced performance"}.get(metric_mode, "performance")
        suggestions.append(f"- Keep it up! Your {label} is well-aligned with the cohort.")

    return suggestions

# --- Compute scores (includes fusion weights & DOC adjustments) ---
def compute_scores(base_df: pd.DataFrame, mode: str):
    # Use adjusted duration/effort and effective depth
    agg_cal = base_df.groupby('Engineer').agg(
        Task_Count=('ID', 'count'),
        Avg_Duration=('Duration_adj', 'mean'),
        Avg_Depth=('Effective Depth', 'mean')
    ).reset_index()
    agg_cal["Eff_Calendar"] = agg_cal['Task_Count'] / (agg_cal['Avg_Duration'] * agg_cal['Avg_Depth'] + 1)

    agg_eff = base_df.groupby('Engineer').agg(
        Task_Count=('ID', 'count'),
        Avg_Effort=('Effort_adj', 'mean'),
        Avg_Depth=('Effective Depth', 'mean')
    ).reset_index()
    agg_eff["Eff_Timesheet"] = agg_eff['Task_Count'] / (agg_eff['Avg_Effort'] * agg_eff['Avg_Depth'] + 1)

    metrics = pd.merge(
        agg_cal[['Engineer', 'Task_Count', 'Avg_Duration', 'Avg_Depth', 'Eff_Calendar']],
        agg_eff[['Engineer', 'Avg_Effort', 'Eff_Timesheet']],
        on='Engineer', how='outer'
    )

    metrics["Perc_Calendar"] = metrics["Eff_Calendar"].rank(pct=True)
    metrics["Perc_Timesheet"] = metrics["Eff_Timesheet"].rank(pct=True)
    metrics["Perc_Combined"] = cal_weight * metrics["Perc_Calendar"].fillna(0) + tim_weight * metrics["Perc_Timesheet"].fillna(0)

    if mode == "Calendar":
        metrics["Display_Score"] = metrics["Eff_Calendar"]
    elif mode == "Timesheet":
        metrics["Display_Score"] = metrics["Eff_Timesheet"]
    else:  # Combined
        metrics["Display_Score"] = metrics["Perc_Combined"] * (100 if scale_0_100 else 1)

    metrics["Rank_Calendar"] = metrics["Eff_Calendar"].rank(ascending=False, method="min")
    metrics["Rank_Timesheet"] = metrics["Eff_Timesheet"].rank(ascending=False, method="min")
    metrics["Rank_Combined"] = metrics["Perc_Combined"].rank(ascending=False, method="min")
    return metrics

# ========================= Main Flow =========================
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    sheets_to_load = [s for s in sheet_names if not s.lower().startswith("hidden") and "template" not in s.lower()]

    dfs = {}
    for sheet in sheets_to_load:
        tmp = xls.parse(sheet)
        tmp["Workbook"] = sheet
        dfs[sheet] = tmp
    df = pd.concat(dfs.values(), ignore_index=True)

    # ---------- Preprocess ----------
    df.columns = df.columns.str.strip()

    # Split multi-activity values (parameterized regex)
    try:
        df['*Activity Type'] = df['*Activity Type'].astype(str).str.split(activity_split_regex)
    except re.error:
        st.warning("Invalid activity split regex. Falling back to default '[;,/]'.")
        df['*Activity Type'] = df['*Activity Type'].astype(str).str.split(r"[;,/]")
    df = df.explode('*Activity Type')
    df['*Activity Type'] = df['*Activity Type'].str.strip()

    # Timestamps & end-time correction
    df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')
    df['*End Time'] = pd.to_datetime(df['*End Time'], errors='coerce')

    # Update Time detection (override if provided)
    if update_time_override and update_time_override in df.columns:
        update_col = update_time_override
    else:
        update_col = next((c for c in df.columns if ('update' in c.lower()) and ('time' in c.lower())), None)
    df['Update Time'] = pd.to_datetime(df[update_col], errors='coerce') if update_col else pd.NaT

    df['True End Time'] = df['*End Time']
    mask_long = (df['True End Time'] - df['Start Time']).dt.days > days_threshold
    mask_done = df['Status'].astype(str).str.contains(re.escape(completed_kw), case=False, na=False)
    df.loc[mask_long & mask_done, 'True End Time'] = df.loc[mask_long & mask_done, 'Update Time']

    # Base duration
    df['Duration (hrs)'] = (df['True End Time'] - df['Start Time']).dt.total_seconds() / 3600

    # Depth from L1..L5
    def get_depth(row):
        for i in reversed(range(1, 6)):
            if pd.notna(row.get(f"L{i}")):
                return i
        return 0
    df['Task Depth'] = df.apply(get_depth, axis=1)

    # Engineer name (parameterized regex)
    try:
        df['Engineer'] = df['Owner'].astype(str).str.extract(owner_regex)[0].str.strip()
    except re.error:
        st.warning("Invalid engineer regex. Falling back to default '^([A-Za-z\\s]+)'.")
        df['Engineer'] = df['Owner'].astype(str).str.extract(r"^([A-Za-z\s]+)")[0].str.strip()

    # Effort hours: prefer *Timesheet if numeric, else Duration
    df['Timesheet (hrs)'] = pd.to_numeric(df['*Timesheet'], errors='coerce')
    df['Effort (hrs)'] = np.where(df['Timesheet (hrs)'].notna(), df['Timesheet (hrs)'], df['Duration (hrs)'])

    # ---------------- DOC extraction & adjustments ----------------
    # Build activity weights UI from data
    activity_list = sorted(df['*Activity Type'].dropna().unique().tolist())
    with st.sidebar.expander("DOC Activity Weights (from data)", expanded=False):
        st.caption("Set a weight per Activity Type (1–10); severity is normalized so 10 = maximum impact.")
        # Selector: only adjust chosen activity types; others use default weight
        default_preset = {"CTOPO": 4, "Rectification": 7, "TOPN": 10}
        default_selection = [a for a in ["CTOPO", "Rectification", "TOPN"] if a in activity_list]
        selected_types = st.multiselect(
            "Select Activity Types to adjust",
            options=activity_list,
            default=default_selection,
            help="Only the selected types will appear below; the rest will use the default weight.",
            key="doc_act_select",
        )

        weights_map = {}
        for act in selected_types:
            init_val = int(default_preset.get(act, doc_default_weight))
            w = st.number_input(f"{act}", min_value=1, max_value=10, value=init_val, step=1, key=f"doc_w_{act}")
            weights_map[act] = int(w)

    with st.sidebar.expander("DOC Severity Band Thresholds", expanded=False):
        st.caption("Define weight bands for DOC severity labels shown in summaries.")
        th_medium = st.number_input("Min weight for Medium", 1, 10, value=4, step=1, key="doc_th_med")
        th_hard   = st.number_input("Min weight for Hard",   1, 10, value=8, step=1, key="doc_th_hard")
        th_vhard  = st.number_input("Min weight for Very Hard", 1, 10, value=10, step=1, key="doc_th_vhard")

    with st.sidebar.expander("Task Links (optional)", expanded=False):
        id_col_for_link = st.text_input("ID column for link", value="ID",
            help="Column name with the ID of the task.")
        link_template = st.text_input("Task link template", value="https://wetask.welink.huawei.com/v2/task_detail?task_id={id}",
            help="Link structure using {id} as placeholder, ej: https://wetask.com/tasks/{id}")

    # --- Task URL: siempre desde el template del sidebar con ID ---
    if (id_col_for_link in df.columns) and link_template:
        df['Task URL'] = df[id_col_for_link].astype(str).map(lambda x: link_template.replace("{id}", x)).str.strip()
    else:
        df['Task URL'] = np.nan

    # DOC extraction
    has_doc, item_counts = [], []
    for nm in df.get('*Name', '').astype(str):
        flag, cnt = extract_doc_info(nm, doc_block_regex, doc_item_sep_regex) if doc_enable else (False, 0)
        has_doc.append(flag)
        item_counts.append(cnt)
    df['Has_DOC_Structure'] = has_doc
    df['DOC_Item_Count'] = item_counts

    # Map activity weight per row
    df['DOC_Activity_Weight'] = df['*Activity Type'].map(weights_map).fillna(doc_default_weight)

    # Compute severity scaled 0–(item_count)
    eff_item_count = np.where(df['Has_DOC_Structure'], df['DOC_Item_Count'], np.where(doc_weight_apply_no_block, 1, 0))
    df['DOC_Severity'] = (eff_item_count * df['DOC_Activity_Weight'])/10.0

    # Visual band based on weight
    def _weight_band(w: float) -> str:
        if w >= th_vhard: return "Very Hard"
        if w >= th_hard:  return "Hard"
        if w >= th_medium:return "Medium"
        return "Low"

    df['DOC_Severity_Band'] = df['DOC_Activity_Weight'].apply(_weight_band)

    # Effective depth and adjusted durations/efforts
    df['Effective Depth'] = df['Task Depth'] + np.where(doc_enable, df['DOC_Severity'] * doc_complexity_to_depth, 0.0)

    # Adjust duration/effort to reward handling complex tasks quickly
    adj_factor = 1.0 + (doc_efficiency_gain * df['DOC_Severity'] if doc_enable else 0.0)
    df['Duration_adj'] = df['Duration (hrs)'] / adj_factor.replace(0, 1.0)
    df['Effort_adj'] = df['Effort (hrs)'] / adj_factor.replace(0, 1.0)

    # =================== NEW: Composite Difficulty ===================
    use_eff_flag_for_time = use_effort_for_diff and (metric_mode in ("Timesheet", "Combined"))
    df['Base Hours'] = np.where(use_eff_flag_for_time, df['Effort_adj'], df['Duration_adj'])

    # --- ROBUSTEZ por actividad: mediana + winsorization + cap ---
    grp = df.groupby('*Activity Type')['Base Hours']

    # Mediana por actividad (menos sensible a outliers)
    act_med = grp.transform('median')

    # IQR local por actividad
    q1 = grp.transform(lambda s: np.nanpercentile(s, 25))
    q3 = grp.transform(lambda s: np.nanpercentile(s, 75))
    iqr = (q3 - q1).replace(0, np.nan)

    # Winsorizamos Base Hours dentro de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    low = (q1 - 1.5 * iqr).fillna(q1)
    high = (q3 + 1.5 * iqr).fillna(q3)
    bh_robust = df['Base Hours'].clip(lower=low, upper=high)

    # Ratio robusto contra mediana
    time_ratio = (bh_robust / act_med).replace([np.inf, -np.inf], np.nan)

    # Cap del ratio (definido en sidebar)
    time_ratio_capped = time_ratio.clip(lower=1.0, upper=cap_ratio)

    # Factor final
    if use_log_time:
        time_factor = np.log2(time_ratio_capped)     # 1.0 -> 0; tope en log2(cap_ratio)
    else:
        time_factor = (time_ratio_capped - 1.0)      # 1.0 -> 0


    df['Composite Difficulty Score'] = (
          depth_weight * df['Effective Depth'].fillna(0)
        + doc_weight   * df['DOC_Severity'].fillna(0)
        + time_weight  * time_factor.fillna(0)
    )

    # Guardar factor de tiempo y aportes (para mostrar desglose)
    df['Time_Factor'] = time_factor.fillna(0)
    df['Comp_Depth']  = depth_weight * df['Effective Depth'].fillna(0)
    df['Comp_DOC']    = doc_weight   * df['DOC_Severity'].fillna(0)
    df['Comp_Time']   = time_weight  * df['Time_Factor']

    def reasoning_text(row):
        # use local time_ratio via index: safer with .iloc if aligned
        try:
            tr = float((row['Base Hours']) / (act_mean.loc[row.name] if act_mean.loc[row.name] > 0 else 1.0))
        except Exception:
            tr = 1.0

        parts = []
        if tr >= 1.5:
            parts.append(f"Time {tr:.1f}× above avg")
        elif tr > 1.1:
            parts.append(f"Time slightly above avg ({tr:.1f}×)")
        else:
            parts.append("Time ~avg")

        doc_w = float(row.get('DOC_Activity_Weight', 0))
        doc_sev = float(row.get('DOC_Severity', 0))
        if doc_sev >= 1.0 or doc_w >= 8:
            parts.append(f"High DOC (weight={doc_w:.0f}, sev={doc_sev:.2f})")
        elif doc_sev > 0:
            parts.append(f"Moderate DOC (sev={doc_sev:.2f})")

        ed = float(row.get('Effective Depth', 0))
        if ed >= 4:
            parts.append(f"High Depth (EffDepth={ed:.2f})")
        elif ed >= 3:
            parts.append(f"Mid Depth (EffDepth={ed:.2f})")
        else:
            parts.append(f"Low Depth (EffDepth={ed:.2f})")

        return " | ".join(parts)

    df['Reasoning'] = df.apply(reasoning_text, axis=1)

    # ---------------- Sidebar Filters ----------------
    st.sidebar.subheader("Filters")
    selected_group = st.sidebar.selectbox("Group (Workbook)", ["All"] + sheets_to_load, key="filter_group")
    product_options = sorted(df['*Product Line'].dropna().unique().tolist())
    selected_product = st.sidebar.multiselect("Product Line", product_options, default=None, key="filter_product")
    customer_options = sorted(df['*Customer Name'].dropna().unique().tolist())
    selected_customer = st.sidebar.multiselect("Customer", customer_options, default=None, key="filter_customer")

    # Apply filters once for the "cohort"
    cohort_df = apply_filters(df, selected_group, selected_product, selected_customer)

    # === Cohort-aware baselines (ROBUSTAS) ===
    grp_coh = cohort_df.groupby('*Activity Type')['Base Hours']

    act_med_coh = grp_coh.transform('median')
    q1c = grp_coh.transform(lambda s: np.nanpercentile(s, 25))
    q3c = grp_coh.transform(lambda s: np.nanpercentile(s, 75))
    iqr_c = (q3c - q1c).replace(0, np.nan)

    low_c = (q1c - 1.5 * iqr_c).fillna(q1c)
    high_c = (q3c + 1.5 * iqr_c).fillna(q3c)
    bh_robust_coh = cohort_df['Base Hours'].clip(lower=low_c, upper=high_c)

    time_ratio_coh = (bh_robust_coh / act_med_coh).replace([np.inf, -np.inf], np.nan)
    time_ratio_coh = time_ratio_coh.clip(lower=1.0, upper=cap_ratio)

    if use_log_time:
        time_factor_coh = np.log2(time_ratio_coh)
    else:
        time_factor_coh = (time_ratio_coh - 1.0)

    cohort_df['Time_Ratio']  = time_ratio_coh.fillna(1.0)
    cohort_df['Time_Factor'] = time_factor_coh.fillna(0.0)

    # === Composite por cohorte ===
    cohort_df['Comp_Depth'] = depth_weight * cohort_df['Effective Depth'].fillna(0)
    cohort_df['Comp_DOC']   = doc_weight   * cohort_df['DOC_Severity'].fillna(0)
    cohort_df['Comp_Time']  = time_weight  * cohort_df['Time_Factor']
    cohort_df['Composite Difficulty Score'] = (
        cohort_df['Comp_Depth'] + cohort_df['Comp_DOC'] + cohort_df['Comp_Time']
    )

    # --- Compatibility alias for reasoning_text ---
    if '_ActBaseline' in df.columns:
        act_mean = df['_ActBaseline']           # usa el mismo baseline que el score
    else:
        act_mean = (
            df.groupby('*Activity Type')['Base Hours'].transform('median').fillna(1.0)
        )


    # === Reasoning (cohort-aware) ===
    def reasoning_text_cohort(row):
        tr = float(row.get('Time_Ratio', 1.0))
        parts = []
        if tr >= 1.5:
            parts.append(f"Time {tr:.1f}× above avg")
        elif tr > 1.1:
            parts.append(f"Time slightly above avg ({tr:.1f}×)")
        else:
            parts.append("Time ~avg")

        w = float(row.get('DOC_Activity_Weight', 0))
        sev = float(row.get('DOC_Severity', 0))
        if sev >= 1.0 or w >= 8:
            parts.append(f"High DOC (weight={w:.0f}, sev={sev:.2f})")
        elif sev > 0:
            parts.append(f"Moderate DOC (sev={sev:.2f})")

        ed = float(row.get('Effective Depth', 0))
        if ed >= 4:
            parts.append(f"High Depth (EffDepth={ed:.2f})")
        elif ed >= 3:
            parts.append(f"Mid Depth (EffDepth={ed:.2f})")
        else:
            parts.append(f"Low Depth (EffDepth={ed:.2f})")
        return " | ".join(parts)

    cohort_df['Reasoning'] = cohort_df.apply(reasoning_text_cohort, axis=1)

    # === Re-classify difficulty for the current cohort ===
    scores = cohort_df['Composite Difficulty Score'].dropna()
    if mode_cohort_thresh == "Cohort-adaptive (percentiles)" and len(scores) >= 5:
        # calcular cortes por percentiles del cohort
        easy_cut = np.percentile(scores, pct_easy)
        med_cut  = np.percentile(scores, pct_med)
        hard_cut = np.percentile(scores, pct_hard)
    else:
        # usar fijos
        easy_cut, med_cut, hard_cut = fixed_easy_max, fixed_med_max, fixed_hard_max

    def classify_by_cuts(s, e_cut, m_cut, h_cut):
        if s < e_cut:      return "Easy"
        elif s < m_cut:    return "Medium"
        elif s < h_cut:    return "Hard"
        else:              return "Very Hard"

    cohort_df['Estimated Difficulty'] = cohort_df['Composite Difficulty Score'].apply(
        lambda v: classify_by_cuts(v, easy_cut, med_cut, hard_cut)
    )

    # --- Micro‑gating post-classification: demotion si no se cumplen señales mínimas ---
    if 'mg_enable' in locals() and mg_enable:

        # Very Hard → Hard
        if 'vh_enable' in locals() and vh_enable:
            mask_vh = cohort_df['Estimated Difficulty'].eq('Very Hard')
            sig_vh = (
                (cohort_df['Time_Factor']      >= tf_vh ).astype(int)
            + (cohort_df['DOC_Severity']     >= doc_vh).astype(int)
            + (cohort_df['Effective Depth']  >= dep_vh).astype(int)
            )
            cohort_df.loc[mask_vh & (sig_vh < vh_req), 'Estimated Difficulty'] = 'Hard'

        # Hard → Medium
        if 'hard_enable' in locals() and hard_enable:
            mask_h = cohort_df['Estimated Difficulty'].eq('Hard')
            sig_h = (
                (cohort_df['Time_Factor']      >= tf_h ).astype(int)
            + (cohort_df['DOC_Severity']     >= doc_h).astype(int)
            + (cohort_df['Effective Depth']  >= dep_h).astype(int)
            )
            cohort_df.loc[mask_h & (sig_h < hard_req), 'Estimated Difficulty'] = 'Medium'

        # Medium → Easy
        if 'med_enable' in locals() and med_enable:
            mask_m = cohort_df['Estimated Difficulty'].eq('Medium')
            sig_m = (
                (cohort_df['Time_Factor']      >= tf_m ).astype(int)
            + (cohort_df['DOC_Severity']     >= doc_m).astype(int)
            + (cohort_df['Effective Depth']  >= dep_m).astype(int)
            )
            cohort_df.loc[mask_m & (sig_m < med_req), 'Estimated Difficulty'] = 'Easy'

    # --- Compute metrics (all three so we can show deltas) ---
    metrics_all = compute_scores(cohort_df, metric_mode)

    # Merge display score onto per-task df for convenience in Individual tab
    cohort_df = cohort_df.merge(
        metrics_all[['Engineer','Eff_Calendar','Eff_Timesheet','Perc_Combined','Rank_Calendar','Rank_Timesheet','Rank_Combined']],
        on='Engineer', how='left'
    )

    tab1, tab2 = st.tabs(["General Performance", "Individual Performance"])

    with tab1:
        st.header("General Performance Overview")

        with st.expander("Difficulty thresholds in use", expanded=False):
            mode_txt = "Cohort-adaptive" if mode_cohort_thresh.startswith("Cohort") else "Fixed"
            st.markdown(f"**Mode:** {mode_txt}")
            st.markdown(f"**Cuts:** Easy < `{easy_cut:.2f}`, Medium < `{med_cut:.2f}`, Hard < `{hard_cut:.2f}`, else Very Hard")
                        
            _total = len(cohort_df)
            if _total > 0:
                _doc_block_share  = float(cohort_df['Has_DOC_Structure'].mean())
                _doc_impact_share = float(cohort_df['DOC_Severity'].gt(0).mean())
                _toggle_txt = "ON (assume 1 item when missing)" if doc_weight_apply_no_block else "OFF (no impact if missing)"

                st.markdown(
                    f"**DOC coverage:** {_doc_block_share:.0%} tasks with AMS blocks · "
                    f"**DOC weight impact:** {_doc_impact_share:.0%} of tasks affected · "
                    f"**No-block toggle:** {_toggle_txt}"
                )
            # --- Tareas con bloques DOC (opcionalmente filtrar "AMS") ---

            with st.expander("Tasks with DOC blocks (details)", expanded=False):
                # Checkbox para filtrar solo bloques que contengan "AMS"
                # only_ams = st.checkbox("Show only blocks containing 'AMS'", value=False, key="doc_blocks_ams_filter")

                # Extraer el PRIMER bloque DOC del nombre usando el mismo regex de la app
                # doc_block_regex ya está definido en el sidebar (DOC Task Complexity)
                try:
                    block_series = cohort_df['*Name'].astype(str).str.extract(re.compile(doc_block_regex))[0]
                except Exception:
                    # Fallback defensivo por si el regex no compila
                    block_series = cohort_df['*Name'].astype(str).str.extract(r'(\[.*?\])')[0]

                # Máscara base: tareas con bloque DOC
                mask_doc = block_series.notna()

                # Armamos tabla simple: Task Name, Owner, Activity Type, Status
                cols = ['*Name', 'Owner', '*Activity Type', 'Status']
                # (Por si alguna columna faltara, filtramos a las disponibles)
                cols = [c for c in cols if c in cohort_df.columns]

                table_df = (
                    cohort_df.loc[mask_doc, cols]
                    .rename(columns={'*Name': 'Task Name', '*Activity Type': 'Activity Type'})
                    .sort_values(by=['Activity Type', 'Task Name'])
                    .reset_index(drop=True)
                )

                if 'Task URL' in cohort_df.columns:
                    table_df['Task URL'] = cohort_df.loc[mask_doc, 'Task URL'].values

                render_interactive_link_table(
                    table_df,
                    name_col='Task Name',
                    url_col='Task URL',
                    keep_cols=['Task Name', 'Owner', 'Activity Type', 'Status'],
                    height=350,
                    filter_key="doc_blocks_tbl_filter",
                    show_filter=True
                )


        # Engineer Efficiency chart with view controls
        st.subheader(f"Engineers by Efficiency ({metric_mode})")
        view_option = st.radio("Select view:", ["All", "Top N", "Bottom N"], horizontal=True, key="eff_view")
        n_value = st.number_input("N value", min_value=1, max_value=100, value=10, step=1, key="eff_n")

        if metric_mode == "Calendar":
            eng_scores = metrics_all[['Engineer','Eff_Calendar']].rename(columns={'Eff_Calendar':'Score'})
        elif metric_mode == "Timesheet":
            eng_scores = metrics_all[['Engineer','Eff_Timesheet']].rename(columns={'Eff_Timesheet':'Score'})
        else:
            eng_scores = metrics_all[['Engineer','Perc_Combined']].rename(columns={'Perc_Combined':'Score'})
            if scale_0_100:
                eng_scores['Score'] = eng_scores['Score'] * 100  # show 0-100 scale

        if view_option == "Top N":
            eng_scores = eng_scores.sort_values(by='Score', ascending=False).head(n_value)
        elif view_option == "Bottom N":
            eng_scores = eng_scores.sort_values(by='Score', ascending=True).head(n_value)
        else:
            eng_scores = eng_scores.sort_values(by='Score', ascending=False)

        fig_all_eff = px.bar(eng_scores, x='Score', y='Engineer', orientation='h',
                             title=f'Engineers by Efficiency ({metric_mode})', color='Score', template='plotly_white')
        st.plotly_chart(fig_all_eff, use_container_width=True)


        # DOC coverage banner (data hygiene)
        st.subheader("Task Difficulty Distribution")
        # TAB: Chart vs Table for Difficulty
        _tab_diff_chart, _tab_diff_table = st.tabs(["Chart", "Table"])

        with _tab_diff_chart:
            difficulty_summary = cohort_df.groupby('Estimated Difficulty')['ID'].count().reset_index(name='Task Count')
            st.plotly_chart(
                px.bar(
                    difficulty_summary,
                    x='Estimated Difficulty', y='Task Count', color='Estimated Difficulty',
                    category_orders={'Estimated Difficulty': DIFF_ORDER}, template='plotly_white'
                ),
                use_container_width=True
            )

        with _tab_diff_table:
            _cols = ['*Name', 'Owner', 'Composite Difficulty Score', 'Estimated Difficulty', 'Reasoning', 'Task URL']
            _available = [c for c in _cols if c in cohort_df.columns]

            _tbl = (
                cohort_df[_available]
                .rename(columns={
                    '*Name': 'Task Name',
                    'Composite Difficulty Score': 'Difficulty Score',
                    'Estimated Difficulty': 'Classification'
                })
                .sort_values(by=['Classification', 'Difficulty Score'], ascending=[True, False])
                .reset_index(drop=True)
            )

            render_interactive_link_table(
                _tbl,
                name_col='Task Name',
                url_col='Task URL',
                keep_cols=['Task Name', 'Owner', 'Difficulty Score', 'Classification', 'Reasoning'],  # 'Open' se agrega solo
                height=450,
                filter_key="gp_diff_tbl_filter",
                show_filter=True
            )


        st.subheader("Average Duration by Activity Type")
        activity_summary = cohort_df.groupby('*Activity Type').agg({'Duration (hrs)': 'mean', 'Effective Depth': 'mean'}).reset_index()
        st.plotly_chart(px.bar(activity_summary.sort_values(by='Duration (hrs)', ascending=False),
                                x='Duration (hrs)', y='*Activity Type', orientation='h', template='plotly_white'),
                            use_container_width=True)

        st.subheader("Task Count by Engineer and Difficulty")
        heatmap_data = cohort_df.pivot_table(index='Engineer', columns='Estimated Difficulty', values='ID', aggfunc='count', fill_value=0)
        st.plotly_chart(px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Blues'), use_container_width=True)

        st.subheader("Task Volume by Product Line")
        prod_summary = cohort_df.groupby('*Product Line')['ID'].count().reset_index(name='Task Count')
        st.plotly_chart(px.bar(prod_summary, x='Task Count', y='*Product Line', orientation='h', template='plotly_white'), use_container_width=True)

        st.subheader("Task Volume by Customer")
        cust_summary = cohort_df.groupby('*Customer Name')['ID'].count().reset_index(name='Task Count')
        st.plotly_chart(px.bar(cust_summary, x='Task Count', y='*Customer Name', orientation='h', template='plotly_white'), use_container_width=True)

        st.subheader("Duration vs Effective Depth by Activity Type")
        st.plotly_chart(px.scatter(activity_summary, x='Effective Depth', y='Duration (hrs)', text='*Activity Type', color='Duration (hrs)',
                                    title="Duration vs Effective Depth", template='plotly_white'), use_container_width=True)

            # -------- Batch ZIP export (text summaries) --------
        st.divider()
        if st.button("Download All Engineer Summaries (ZIP)"):
            tmp_zip = BytesIO()
            with zipfile.ZipFile(tmp_zip, "w") as zipf:
                for eng in cohort_df['Engineer'].dropna().unique():
                    eng_df = cohort_df[cohort_df['Engineer'] == eng]
                    suggestions = generate_suggestions(eng_df, cohort_df, metric_mode)

                    rowm = metrics_all.loc[metrics_all['Engineer'] == eng]
                    cal = float(rowm['Eff_Calendar']) if not rowm.empty else float('nan')
                    tim = float(rowm['Eff_Timesheet']) if not rowm.empty else float('nan')
                    comb = float(rowm['Perc_Combined'] * (100 if scale_0_100 else 1)) if not rowm.empty else float('nan')

                    # Band share summary
                    band_share = (eng_df['DOC_Severity_Band']
                                    .value_counts(normalize=True)
                                    .reindex(['Low','Medium','Hard','Very Hard'])
                                    .fillna(0.0))

                    # --- NUEVO: DOC Severity thresholds (para trazabilidad en el ZIP) ---
                    doc_thr_line = (
                        f"DOC Severity thresholds: "
                        f"Medium ≥ {int(th_medium)}, Hard ≥ {int(th_hard)}, Very Hard ≥ {int(th_vhard)}"
                    )

                    content = [
                        f"Engineer: {eng}",
                        f"Total Tasks: {len(eng_df)}",
                        f"Avg Duration (hrs): {eng_df['Duration (hrs)'].mean():.2f}",
                        f"Avg Effort (hrs): {eng_df['Effort (hrs)'].mean():.2f}",
                        f"Avg Effective Depth: {eng_df['Effective Depth'].mean():.2f}",
                        f"Avg DOC Severity: {eng_df['DOC_Severity'].mean():.2f}",
                        doc_thr_line,  # ⬅️ NUEVO: linea con thresholds DOC
                        f"DOC Band Share — Low: {band_share['Low']:.0%}, Medium: {band_share['Medium']:.0%}, Hard: {band_share['Hard']:.0%}, Very Hard: {band_share['Very Hard']:.0%}",
                        f"Efficiency — Calendar: {cal:.4f}",
                        f"Efficiency — Timesheet: {tim:.4f}",
                        f"Efficiency — Combined ({'0–100' if scale_0_100 else '0–1'}): {comb:.2f}",
                        "Suggestions:",
                        *suggestions,
                    ]

                    zipf.writestr(f"{eng}_summary.txt", "\n".join(content))
            tmp_zip.seek(0)
            st.download_button("Download All Summaries ZIP", data=tmp_zip, file_name="Engineer_Summaries.zip")

    with tab2:
        st.header("Individual Engineer Report (Cohort-Aware)")
        engineers_in_cohort = cohort_df['Engineer'].dropna().unique()
        if len(engineers_in_cohort) == 0:
            st.info("No engineers in the current filter selection.")
            st.stop()

        selected_engineer = st.selectbox("Select Engineer", sorted(engineers_in_cohort), key="ind_eng")

        if selected_engineer:
            eng_df = cohort_df[cohort_df['Engineer'] == selected_engineer]

            # --- Aging WIP (≥30 days) for this engineer ---
            AGING_WIP_DAYS = 30  # change if you want a different threshold

            # "Open" = status does NOT contain the completed keyword
            open_mask = ~eng_df['Status'].astype(str).str.contains(re.escape(completed_kw), case=False, na=False)

            # Age in days from Start Time to now
            age_days = (pd.Timestamp.now() - pd.to_datetime(eng_df['Start Time'], errors='coerce')).dt.days

            aging_wip_count = int(((open_mask) & (age_days >= AGING_WIP_DAYS)).sum())

            # --- END OF AGING WIP

            rowm = metrics_all.loc[metrics_all['Engineer'] == selected_engineer].iloc[0]
            rank_cal = int(rowm['Rank_Calendar']) if not pd.isna(rowm['Rank_Calendar']) else None
            rank_tim = int(rowm['Rank_Timesheet']) if not pd.isna(rowm['Rank_Timesheet']) else None
            rank_comb = int(rowm['Rank_Combined']) if not pd.isna(rowm['Rank_Combined']) else None

            if metric_mode == "Calendar":
                sel_score = float(rowm['Eff_Calendar'])
                sel_rank = rank_cal
            elif metric_mode == "Timesheet":
                sel_score = float(rowm['Eff_Timesheet'])
                sel_rank = rank_tim
            else:
                sel_score = float(rowm['Perc_Combined'] * (100 if scale_0_100 else 1))
                sel_rank = rank_comb

            if metric_mode == "Calendar":
                series = metrics_all['Eff_Calendar']
            elif metric_mode == "Timesheet":
                series = metrics_all['Eff_Timesheet']
            else:
                series = metrics_all['Perc_Combined']

            rank_percent = (series.rank(pct=True)[metrics_all['Engineer'] == selected_engineer].iloc[0]) * 100

            if rank_percent >= 90:
                medal = "🏅 Gold Performer"
            elif rank_percent >= 70:
                medal = "🥈 Silver Performer"
            elif rank_percent >= 50:
                medal = "🥉 Bronze Performer"
            else:
                medal = "📈 Keep Growing"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tasks", len(eng_df))
            c2.metric("Avg Duration (hrs)", f"{eng_df['Duration (hrs)'].mean():.2f}")
            c3.metric("Avg Effort (hrs)", f"{eng_df['Effort (hrs)'].mean():.2f}")
            c4.metric(f"Aging WIP (≥{AGING_WIP_DAYS}d)", aging_wip_count)

            # --- KPI Details strip (INCLUDES DOC) ---
            cal_val = float(rowm['Eff_Calendar']) if not pd.isna(rowm['Eff_Calendar']) else float('nan')
            tim_val = float(rowm['Eff_Timesheet']) if not pd.isna(rowm['Eff_Timesheet']) else float('nan')
            comb_val = float(rowm['Perc_Combined'] * (100 if scale_0_100 else 1)) if not pd.isna(rowm['Perc_Combined']) else float('nan')
            avg_depth_val = float(eng_df['Effective Depth'].mean()) if len(eng_df) else float('nan')
            avg_doc_val = float(eng_df['DOC_Severity'].mean()) if len(eng_df) else float('nan')

            st.markdown("**KPI Details**")
            k1, k2, k3, k4 = st.columns(4)

            k1.metric("Eff_Calendar", f"{cal_val:.4f}" if np.isfinite(cal_val) else "—")
            k2.metric("Eff_Timesheet", f"{tim_val:.4f}" if np.isfinite(tim_val) else "—")
            k3.metric(f"Combined ({'0–100' if scale_0_100 else '0–1'})",
                      f"{comb_val:.2f}" if scale_0_100 and np.isfinite(comb_val) else
                      (f"{comb_val:.4f}" if np.isfinite(comb_val) else "—"))
            k4.metric("Avg Effective Depth", f"{avg_depth_val:.2f}" if np.isfinite(avg_depth_val) else "—")
            st.caption(f"Avg DOC Severity: {avg_doc_val:.2f}")

            st.markdown(f"**Efficiency Percentile (within current filters):** Top **{int(rank_percent)}%** {medal}")
            st.progress(min(max(rank_percent, 0), 100) / 100.0)

            colA, colB, colC = st.columns(3)
            colA.metric("Rank — Calendar", rank_cal)
            colB.metric("Rank — Timesheet", rank_tim, delta=(rank_tim - rank_cal) if (rank_tim and rank_cal) else None)
            colC.metric("Rank — Combined", rank_comb, delta=(rank_comb - rank_cal) if (rank_comb and rank_cal) else None)

            fig_hist, fig_scatter = make_report_figures(eng_df)
            # --- Task Difficulty Distribution (Engineer) with Tabs ---
            _tab_eng_diff_chart, _tab_eng_diff_table = st.tabs(["Chart", "Table"])

            with _tab_eng_diff_chart:
                st.plotly_chart(fig_hist, use_container_width=True)

            with _tab_eng_diff_table:
                eng_cols = ['*Name', 'Owner', 'Composite Difficulty Score', 'Estimated Difficulty', 'Reasoning', 'Task URL']
                eng_cols = [c for c in eng_cols if c in eng_df.columns]
                eng_tbl = (
                    eng_df[eng_cols]
                    .rename(columns={
                        '*Name': 'Task Name',
                        'Composite Difficulty Score': 'Difficulty Score',
                        'Estimated Difficulty': 'Classification'
                    })
                    .sort_values(by=['Classification', 'Difficulty Score'], ascending=[True, False])
                    .reset_index(drop=True)
                )
                render_interactive_link_table(
                    eng_tbl,
                    name_col='Task Name',
                    url_col='Task URL',
                    keep_cols=['Task Name', 'Owner', 'Difficulty Score', 'Classification', 'Reasoning'],
                    height=420,
                    filter_key="ind_diff_tbl_filter",
                    show_filter=True
                )

            # El scatter se mantiene igual, debajo:
            st.plotly_chart(fig_scatter, use_container_width=True)


            top_activities = eng_df['*Activity Type'].value_counts().head(5).reset_index()
            st.markdown("**Top 5 Activity Types:**")
            st.table(top_activities.rename(columns={'index': 'Activity Type', '*Activity Type': 'Task Count'}))

            st.subheader("Comments based on your performance this month")
            suggestions = generate_suggestions(eng_df, cohort_df, metric_mode)
            for s in suggestions:
                st.markdown(s)

            csv = eng_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Engineer Tasks CSV", data=csv, file_name=f"{selected_engineer}_tasks.csv", mime="text/csv")

else:
    st.info("Please upload an Excel file from WeTask with the correct task structure to begin. Check the documentation in case of doubts.")

st.sidebar.link_button("Documentation","https://3ms.huawei.com/km/groups/3956599/blogs/details/21549614")
st.sidebar.text("Developed by a00401250 Alejandro Uribe for Argentina PSD. V1.1")