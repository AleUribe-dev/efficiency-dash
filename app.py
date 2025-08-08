import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import BytesIO
import zipfile

st.set_page_config(layout="wide", page_title="Engineer Task Efficiency Analyzer")
st.title("Engineer Task Efficiency Analyzer")

# ---------------- Sidebar: Uploader + Filters ----------------
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload Efficiency Excel File", type=["xlsx"])  # keep .xlsx

# --- Metric mode toggle ---
st.sidebar.subheader("Metric Mode")
metric_mode = st.sidebar.radio(
    "Choose the efficiency metric:",
    ["Calendar", "Timesheet", "Combined"],
    help=(
        "Calendar: uses Start/End elapsed hours.\n"
        "Timesheet: uses *Timesheet effort hours when available, else Calendar.\n"
        "Combined: average of Calendar & Timesheet percentiles (fair + robust)."
    ),
    horizontal=False,
    key="metric_mode_toggle",
)

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

def make_report_figures(eng_df):
    fig_hist = px.histogram(
        eng_df,
        x="Estimated Difficulty",
        color="Estimated Difficulty",
        category_orders={"Estimated Difficulty": DIFF_ORDER},
        text_auto=True,
        title="Task Difficulty Distribution",
        template="plotly_white",
    )
    fig_hist.update_layout(font=dict(size=15), title_font=dict(size=20), bargap=0.2, legend_title_text="Difficulty")
    fig_hist.update_xaxes(title_text="Estimated Difficulty")
    fig_hist.update_yaxes(title_text="Task Count")
    fig_hist.update_traces(textfont_size=14, textposition="outside", cliponaxis=False)

    fig_scatter = px.scatter(
        eng_df,
        x="Start Time",
        y="Duration (hrs)",
        color="*Activity Type",
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

# --- Difficulty classification ---
# Tightened rules: we require BOTH the hour threshold and depth threshold for "Hard" now.
# Selector uses Duration hours in Calendar mode, Effort hours otherwise.

def classify_difficulty(row, use_effort: bool):
    hrs = row["Effort (hrs)"] if use_effort else row["Duration (hrs)"]
    depth = row["Task Depth"]
    if (hrs <= 170) and (depth <= 2):
        return "Easy"
    elif (hrs <= 340) and (depth <= 3):
        return "Medium"
    elif (hrs <= 442) and (depth <= 4):
        return "Hard"
    else:
        return "Very Hard"

# --- Compute scores ---

def compute_scores(base_df: pd.DataFrame, mode: str):
    # Engineer aggregates for Calendar & Effort
    agg_cal = base_df.groupby('Engineer').agg(
        Task_Count=('ID', 'count'),
        Avg_Duration=('Duration (hrs)', 'mean'),
        Avg_Depth=('Task Depth', 'mean')
    ).reset_index()
    agg_cal["Eff_Calendar"] = agg_cal['Task_Count'] / (agg_cal['Avg_Duration'] * agg_cal['Avg_Depth'] + 1)

    agg_eff = base_df.groupby('Engineer').agg(
        Task_Count=('ID', 'count'),
        Avg_Effort=('Effort (hrs)', 'mean'),
        Avg_Depth=('Task Depth', 'mean')
    ).reset_index()
    agg_eff["Eff_Timesheet"] = agg_eff['Task_Count'] / (agg_eff['Avg_Effort'] * agg_eff['Avg_Depth'] + 1)

    # Merge bases
    metrics = pd.merge(
        agg_cal[['Engineer', 'Task_Count', 'Avg_Duration', 'Avg_Depth', 'Eff_Calendar']],
        agg_eff[['Engineer', 'Avg_Effort', 'Eff_Timesheet']],
        on='Engineer', how='outer'
    )

    # Percentiles for fusion
    metrics["Perc_Calendar"] = metrics["Eff_Calendar"].rank(pct=True)
    metrics["Perc_Timesheet"] = metrics["Eff_Timesheet"].rank(pct=True)
    metrics["Perc_Combined"] = 0.5 * metrics["Perc_Calendar"].fillna(0) + 0.5 * metrics["Perc_Timesheet"].fillna(0)

    # Select display score depending on mode
    if mode == "Calendar":
        metrics["Display_Score"] = metrics["Eff_Calendar"]
    elif mode == "Timesheet":
        metrics["Display_Score"] = metrics["Eff_Timesheet"]
    else:  # Combined
        metrics["Display_Score"] = metrics["Perc_Combined"] * 100  # nicer scale for charts

    # Ranks (1 best)
    metrics["Rank_Calendar"] = metrics["Eff_Calendar"].rank(ascending=False, method="min")
    metrics["Rank_Timesheet"] = metrics["Eff_Timesheet"].rank(ascending=False, method="min")
    metrics["Rank_Combined"] = metrics["Perc_Combined"].rank(ascending=False, method="min")
    return metrics


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

    # Split multi-activity values
    df['*Activity Type'] = df['*Activity Type'].astype(str).str.split(r'[;,/]')
    df = df.explode('*Activity Type')
    df['*Activity Type'] = df['*Activity Type'].str.strip()

    # Timestamps & end-time correction
    df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')
    df['*End Time'] = pd.to_datetime(df['*End Time'], errors='coerce')

    update_col = next((c for c in df.columns if 'update' in c.lower() and 'time' in c.lower()), None)
    df['Update Time'] = pd.to_datetime(df[update_col], errors='coerce') if update_col else pd.NaT
    df['*Phase'] = df['*Phase'].astype(str)

    df['True End Time'] = df['*End Time']
    mask_long = (df['True End Time'] - df['Start Time']).dt.days > 180
    mask_done = df['*Phase'].str.lower().str.contains("completed")
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

    # Engineer name (strip IDs)
    df['Engineer'] = df['Owner'].astype(str).str.extract(r"^([A-Za-z\s]+)")[0].str.strip()

    # Effort hours: prefer *Timesheet if numeric, else Duration
    df['Timesheet (hrs)'] = pd.to_numeric(df['*Timesheet'], errors='coerce')
    df['Effort (hrs)'] = np.where(df['Timesheet (hrs)'].notna(), df['Timesheet (hrs)'], df['Duration (hrs)'])

    # Difficulty depends on metric mode (Calendar uses duration hours; others use effort hours)
    use_effort_for_difficulty = metric_mode in ("Timesheet", "Combined")
    df['Estimated Difficulty'] = df.apply(lambda r: classify_difficulty(r, use_effort_for_difficulty), axis=1)

    # ---------------- Sidebar Filters ----------------
    st.sidebar.subheader("Filters")
    selected_group = st.sidebar.selectbox("Group (Workbook)", ["All"] + sheets_to_load, key="filter_group")
    product_options = sorted(df['*Product Line'].dropna().unique().tolist())
    selected_product = st.sidebar.multiselect("Product Line", product_options, default=None, key="filter_product")
    customer_options = sorted(df['*Customer Name'].dropna().unique().tolist())
    selected_customer = st.sidebar.multiselect("Customer", customer_options, default=None, key="filter_customer")

    # Apply filters once for the "cohort"
    cohort_df = apply_filters(df, selected_group, selected_product, selected_customer)

    # --- Compute metrics (all three so we can show deltas) ---
    metrics_all = compute_scores(cohort_df, metric_mode)

    # Merge display score onto per-task df for convenience in Individual tab
    cohort_df = cohort_df.merge(metrics_all[['Engineer','Eff_Calendar','Eff_Timesheet','Perc_Combined','Rank_Calendar','Rank_Timesheet','Rank_Combined']],
                                on='Engineer', how='left')

    tab1, tab2 = st.tabs(["General Performance", "Individual Performance"])

    with tab1:
        st.header("General Performance Overview")

        # Engineer Efficiency chart with view controls
        st.subheader(f"Engineers by Efficiency ({metric_mode})")
        view_option = st.radio("Select view:", ["All", "Top N", "Bottom N"], horizontal=True, key="eff_view")
        n_value = st.number_input("N value", min_value=1, max_value=50, value=10, step=1, key="eff_n")

        if metric_mode == "Calendar":
            eng_scores = metrics_all[['Engineer','Eff_Calendar']].rename(columns={'Eff_Calendar':'Score'})
        elif metric_mode == "Timesheet":
            eng_scores = metrics_all[['Engineer','Eff_Timesheet']].rename(columns={'Eff_Timesheet':'Score'})
        else:
            eng_scores = metrics_all[['Engineer','Perc_Combined']].rename(columns={'Perc_Combined':'Score'})
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

        st.subheader("Task Difficulty Distribution")
        difficulty_summary = cohort_df.groupby('Estimated Difficulty')['ID'].count().reset_index(name='Task Count')
        st.plotly_chart(px.bar(difficulty_summary, x='Estimated Difficulty', y='Task Count', color='Estimated Difficulty',
                               category_orders={'Estimated Difficulty': DIFF_ORDER}, template='plotly_white'),
                        use_container_width=True)

        st.subheader("Average Duration by Activity Type")
        activity_summary = cohort_df.groupby('*Activity Type').agg({'Duration (hrs)': 'mean', 'Task Depth': 'mean'}).reset_index()
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

        st.subheader("Duration vs Task Depth by Activity Type")
        st.plotly_chart(px.scatter(activity_summary, x='Task Depth', y='Duration (hrs)', text='*Activity Type', color='Duration (hrs)',
                                   title="Duration vs Task Depth", template='plotly_white'), use_container_width=True)

        # -------- Batch ZIP export (text summaries) --------
        st.divider()
        if st.button("Download All Engineer Summaries (ZIP)"):
            tmp_zip = BytesIO()
            with zipfile.ZipFile(tmp_zip, "w") as zipf:
                for eng in cohort_df['Engineer'].dropna().unique():
                    eng_df = cohort_df[cohort_df['Engineer'] == eng]
                    # suggestions (simple heuristics)
                    suggestions = []
                    mean_effort = eng_df['Effort (hrs)'].mean()
                    mean_depth = eng_df['Task Depth'].mean()
                    if mean_effort and mean_effort > 340:
                        suggestions.append("- Consider time-boxing or requesting support for long-duration/effort tasks.")
                    if mean_depth and mean_depth > 3:
                        suggestions.append("- You are managing deep task structures — consider mentoring others.")
                    if eng_df.groupby('*Activity Type')['Duration (hrs)'].std().mean() > 50:
                        suggestions.append("- Aim for consistency across similar task types. High variation suggests standardization gaps.")
                    if eng_df['Estimated Difficulty'].value_counts().get("Very Hard", 0) > 5:
                        suggestions.append("- Frequently assigned the most complex tasks — consider workload rebalancing or role recognition.")
                    if eng_df['*Activity Type'].nunique() < 3:
                        suggestions.append("- Consider requesting task variety to broaden skill application.")
                    if not suggestions:
                        suggestions.append("- Keep up the good work! Balanced performance across tasks.")

                    # metrics snapshot
                    rowm = metrics_all.loc[metrics_all['Engineer'] == eng]
                    cal = float(rowm['Eff_Calendar']) if not rowm.empty else float('nan')
                    tim = float(rowm['Eff_Timesheet']) if not rowm.empty else float('nan')
                    comb = float(rowm['Perc_Combined'] * 100) if not rowm.empty else float('nan')

                    content = [
                        f"Engineer: {eng}",
                        f"Total Tasks: {len(eng_df)}",
                        f"Avg Duration (hrs): {eng_df['Duration (hrs)'].mean():.2f}",
                        f"Avg Effort (hrs): {eng_df['Effort (hrs)'].mean():.2f}",
                        f"Avg Task Depth: {eng_df['Task Depth'].mean():.2f}",
                        f"Efficiency — Calendar: {cal:.4f}",
                        f"Efficiency — Timesheet: {tim:.4f}",
                        f"Efficiency — Combined (0–100): {comb:.2f}",
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

            # Current selection rank + deltas
            rowm = metrics_all.loc[metrics_all['Engineer'] == selected_engineer].iloc[0]
            rank_cal = int(rowm['Rank_Calendar']) if not pd.isna(rowm['Rank_Calendar']) else None
            rank_tim = int(rowm['Rank_Timesheet']) if not pd.isna(rowm['Rank_Timesheet']) else None
            rank_comb = int(rowm['Rank_Combined']) if not pd.isna(rowm['Rank_Combined']) else None

            # Choose display score & rank depending on mode
            if metric_mode == "Calendar":
                sel_score = float(rowm['Eff_Calendar'])
                sel_rank = rank_cal
            elif metric_mode == "Timesheet":
                sel_score = float(rowm['Eff_Timesheet'])
                sel_rank = rank_tim
            else:
                sel_score = float(rowm['Perc_Combined'] * 100)
                sel_rank = rank_comb

            # Percentile within the current filters (using chosen metric)
            if metric_mode == "Calendar":
                series = metrics_all['Eff_Calendar']
                sel_val = rowm['Eff_Calendar']
            elif metric_mode == "Timesheet":
                series = metrics_all['Eff_Timesheet']
                sel_val = rowm['Eff_Timesheet']
            else:
                series = metrics_all['Perc_Combined']
                sel_val = rowm['Perc_Combined']

            # Top % (higher is better)
            rank_percent = (series.rank(pct=True)[metrics_all['Engineer'] == selected_engineer].iloc[0]) * 100

            # Medal
            if rank_percent >= 90:
                medal = "🏅 Gold Performer"
            elif rank_percent >= 70:
                medal = "🥈 Silver Performer"
            elif rank_percent >= 50:
                medal = "🥉 Bronze Performer"
            else:
                medal = "📈 Keep Growing"

            # KPI Tiles
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tasks", len(eng_df))
            c2.metric("Avg Duration (hrs)", f"{eng_df['Duration (hrs)'].mean():.2f}")
            c3.metric("Avg Effort (hrs)", f"{eng_df['Effort (hrs)'].mean():.2f}")
            c4.metric(f"Efficiency ({metric_mode})", f"{sel_score:.4f}" if metric_mode != 'Combined' else f"{sel_score:.2f}")

            st.markdown(f"**Efficiency Percentile (within current filters):** Top **{int(rank_percent)}%** {medal}")
            st.progress(min(max(rank_percent, 0), 100) / 100.0)

            # Rank deltas table
            colA, colB, colC = st.columns(3)
            colA.metric("Rank — Calendar", rank_cal)
            colB.metric("Rank — Timesheet", rank_tim, delta=(rank_tim - rank_cal) if (rank_tim and rank_cal) else None)
            colC.metric("Rank — Combined", rank_comb, delta=(rank_comb - rank_cal) if (rank_comb and rank_cal) else None)

            # Charts
            fig_hist, fig_scatter = make_report_figures(eng_df)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Top activities
            top_activities = eng_df['*Activity Type'].value_counts().head(5).reset_index()
            st.markdown("**Top 5 Activity Types:**")
            st.table(top_activities.rename(columns={'index': 'Activity Type', '*Activity Type': 'Task Count'}))

            # Suggestions
            st.subheader("Suggestions for Improvement")
            suggestions = []
            if eng_df['Effort (hrs)'].mean() > 340:
                suggestions.append("- Consider time-boxing or requesting support for high-effort tasks.")
            if eng_df['Task Depth'].mean() > 3:
                suggestions.append("- You are managing deep task structures — consider mentoring others.")
            if eng_df.groupby('*Activity Type')['Duration (hrs)'].std().mean() > 50:
                suggestions.append("- Aim for consistency in tasks of similar type. High variation suggests standardization gaps.")
            if eng_df['Estimated Difficulty'].value_counts().get("Very Hard", 0) > 5:
                suggestions.append("- You're frequently assigned the most complex tasks. Consider workload rebalancing.")
            if eng_df['*Activity Type'].nunique() < 3:
                suggestions.append("- Consider requesting task variety to broaden skill application.")
            if not suggestions:
                suggestions.append("- Keep up the good work! Your performance is well-balanced across tasks.")
            for s in suggestions:
                st.markdown(s)

            # Optional CSV export for this engineer
            csv = eng_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Engineer Tasks CSV", data=csv, file_name=f"{selected_engineer}_tasks.csv", mime="text/csv")

            # -------- Print View --------
            st.subheader("Print View")
            st.caption("Use this to print/export a clean PDF of the Individual Report (hides sidebar & controls).")
            print_btn = """
                <div class="print-hide">
                    <button onclick="window.print()">🖨️ Print / Save as PDF</button>
                </div>
            """
            st.markdown(print_btn, unsafe_allow_html=True)
else:
    st.info("Please upload an Excel file to begin.")
