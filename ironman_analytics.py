"""
Ironman Race Analytics – Streamlit App
Five modules mirroring the React web app:
  1. Race Intelligence Dashboard
  2. Athlete Performance Comparator
  3. Pacing Strategy Analyzer
  4. Predictive Time Model
  5. Race Strategy Builder

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ironman Race Analytics",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

YEARS = list(range(2003, 2026))          # 2003 – 2025
DISCIPLINE_COLORS = {
    "Swim":  "#38bdf8",   # sky blue
    "T1":    "#a78bfa",   # violet
    "Bike":  "#34d399",   # emerald
    "T2":    "#fb923c",   # orange
    "Run":   "#f87171",   # red
}
ATHLETE_PALETTE = ["#10b981", "#3b82f6", "#f97316", "#a855f7"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_time(t) -> float:
    """HH:MM:SS -> seconds. Returns NaN for zeros / invalid / DNF."""
    s = str(t).strip() if not pd.isna(t) else ""
    if s in ("", "00:0:0", "0:0:0", "nan"):
        return np.nan
    try:
        parts = s.split(":")
        if len(parts) == 3:
            total = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            total = int(parts[0]) * 60 + int(parts[1])
        else:
            return np.nan
        return float(total) if total > 0 else np.nan
    except Exception:
        return np.nan


def hms(seconds) -> str:
    """seconds -> H:MM:SS"""
    if pd.isna(seconds):
        return "N/A"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def hm(seconds) -> str:
    """seconds -> H:MM"""
    if pd.isna(seconds):
        return "N/A"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}:{m:02d}"


def pace_per_km(seconds, km) -> str:
    """seconds for a leg -> MM:SS/km pace string"""
    if pd.isna(seconds) or km == 0:
        return "N/A"
    pace = seconds / km
    m = int(pace // 60)
    s = int(round(pace % 60))
    return f"{m}:{s:02d}/km"


def speed_kmh(seconds, km) -> str:
    if pd.isna(seconds) or seconds == 0:
        return "N/A"
    return f"{(km / seconds * 3600):.1f} km/h"


def percentile_rank(value, series) -> float:
    """Percentile of 'value' within 'series' (lower time to higher percentile)."""
    valid = series.dropna()
    if len(valid) == 0:
        return 50.0
    return round(float((valid > value).sum() / len(valid) * 100), 1)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading race data …")
def load_data() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        for code in ("M", "F"):
            path = os.path.join(DATA_DIR, f"IM{year}_{code}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, low_memory=False)
                df["Year"] = year
                df["FileGender"] = code
                frames.append(df)
            except Exception as e:
                st.warning(f"Could not read {path}: {e}")

    if not frames:
        st.error(
            "No CSV files found. Place IM2003_F.csv … IM2026_M.csv "
            f"in: {DATA_DIR}"
        )
        st.stop()

    raw = pd.concat(frames, ignore_index=True)

    # ── Parse times to seconds ──────────────────────────────────────────────
    time_map = {
        "Overall Time":       "Overall_sec",
        "Swim Time":          "Swim_sec",
        "Bike Time":          "Bike_sec",
        "Run Time":           "Run_sec",
        "Transition 1 Time":  "T1_sec",
        "Transition 2 Time":  "T2_sec",
    }
    for col, sec_col in time_map.items():
        if col in raw.columns:
            raw[sec_col] = raw[col].apply(parse_time)

    # ── Keep finishers with valid overall time ───────────────────────────────
    raw = raw[raw["Finish"] == "FIN"].copy()
    raw = raw[raw["Overall_sec"].notna() & (raw["Overall_sec"] > 0)].copy()

    # ── Derive gender from Division prefix (more reliable than Gender col) ───
    def div_gender(div):
        d = str(div).upper()
        if d.startswith("F"):
            return "Female"
        if d.startswith("M"):
            return "Male"
        return "Unknown"

    raw["Gender_clean"] = raw["Division"].apply(div_gender)

    # ── Normalise age-group labels  ──────────────────────────────────────────
    def clean_div(div):
        d = str(div).strip()
        if d in ("MPRO", "FPRO"):
            return "PRO"
        return d.replace("M", "").replace("F", "").strip() if d not in ("Male", "Female") else d

    raw["AgeGroup"] = raw["Division"].apply(clean_div)

    # ── Derived numeric columns ──────────────────────────────────────────────
    raw["Total_min"] = raw["Overall_sec"] / 60
    raw["Total_hr"]  = raw["Overall_sec"] / 3600

    # ── Percentile within year × gender × age-group ─────────────────────────
    # (computed lazily per query to avoid huge upfront cost)

    raw.reset_index(drop=True, inplace=True)
    return raw


df_all = load_data()

# ─── Sidebar – Global Filters ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Global Filters")
    sel_years = st.multiselect(
        "Year(s)",
        options=YEARS,
        default=YEARS,
    )
    sel_gender = st.selectbox(
        "Gender",
        ["All", "Male", "Female"],
        index=0,
    )
    all_ag = sorted(df_all["AgeGroup"].dropna().unique().tolist())
    sel_ag = st.multiselect(
        "Age Group(s)",
        options=all_ag,
        default=[],
        placeholder="All age groups",
    )
    st.markdown("---")
    st.caption("Ironman World Championship • 2003 – 2025")

# Apply global filters
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if sel_years:
        out = out[out["Year"].isin(sel_years)]
    if sel_gender != "All":
        out = out[out["Gender_clean"] == sel_gender]
    if sel_ag:
        out = out[out["AgeGroup"].isin(sel_ag)]
    return out

df = apply_filters(df_all)

# ─── Tabs ──────────���──────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏁 Race Intelligence Dashboard",
    "⚖️ Athlete Comparator",
    "📈 Pacing Strategy Analyzer",
    "🔮 Predictive Time Model",
    "🗺️ Race Strategy Builder",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RACE INTELLIGENCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Race Intelligence Dashboard")

    if df.empty:
        st.warning("No data matches the current filters.")
    else:
        # ── KPI Cards ────────────────────────────────────────────────────────
        n_fin     = len(df)
        avg_total = df["Overall_sec"].mean()
        best_time = df["Overall_sec"].min()
        worst_time= df["Overall_sec"].max()
        avg_swim  = df["Swim_sec"].dropna().mean()
        avg_bike  = df["Bike_sec"].dropna().mean()
        avg_run   = df["Run_sec"].dropna().mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Finishers",  f"{n_fin:,}")
        c2.metric("Average Finish",   hms(avg_total))
        c3.metric("Course Record",    hms(best_time))
        c4.metric("Slowest Finisher", hms(worst_time))

        st.markdown("---")

        # ── Average split composition ─────────────────────────────────────
        st.subheader("Average Split Composition")
        splits = {
            "Swim": avg_swim,
            "T1":   df["T1_sec"].dropna().mean(),
            "Bike": avg_bike,
            "T2":   df["T2_sec"].dropna().mean(),
            "Run":  avg_run,
        }
        total_split = sum(v for v in splits.values() if not np.isnan(v))
        split_pct = {k: (v / total_split * 100 if not np.isnan(v) else 0) for k, v in splits.items()}

        fig_split = go.Figure()
        x_pos = 0
        for seg, pct in split_pct.items():
            fig_split.add_trace(go.Bar(
                name=seg,
                x=[pct],
                y=["Avg Finisher"],
                orientation="h",
                marker_color=DISCIPLINE_COLORS[seg],
                text=f"{seg}<br>{hms(splits[seg])}<br>({pct:.1f}%)",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"<b>{seg}</b><br>Avg: {hms(splits[seg])}<br>{pct:.1f}% of race<extra></extra>",
            ))

        fig_split.update_layout(
            barmode="stack",
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        st.plotly_chart(fig_split, width="stretch")

        st.markdown("---")

        col_hist, col_bench = st.columns([3, 2])

        # ── Finish-Time Histogram ────────────────────────────────────────────
        with col_hist:
            st.subheader("Finish Time Distribution")
            bins = st.slider("Histogram bins", 20, 80, 40, key="hist_bins")
            fig_hist = px.histogram(
                df,
                x="Total_hr",
                nbins=bins,
                labels={"Total_hr": "Finish Time (hours)"},
                color_discrete_sequence=["#10b981"],
            )
            fig_hist.update_traces(
                hovertemplate="Time: %{x:.2f} hrs<br>Count: %{y}<extra></extra>"
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font_color="#9ca3af",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#374151"),
                yaxis=dict(title="Athletes", gridcolor="#374151"),
            )
            st.plotly_chart(fig_hist, width="stretch")

        # ── Age Group Benchmarks ─────────────────────────────────────────────
        with col_bench:
            st.subheader("Age Group Benchmarks")
            ag_stats = (
                df.groupby("AgeGroup")["Overall_sec"]
                .agg(["mean", "median", "min", "count"])
                .reset_index()
                .rename(columns={"mean": "Avg", "median": "Median",
                                  "min": "Best", "count": "N"})
                .sort_values("Avg")
            )
            ag_stats = ag_stats[ag_stats["N"] >= 5]  # At least 5 finishers

            fig_ag = go.Figure()
            fig_ag.add_trace(go.Bar(
                y=ag_stats["AgeGroup"],
                x=ag_stats["Avg"] / 3600,
                orientation="h",
                name="Average",
                marker_color="#10b981",
                text=[hm(v) for v in ag_stats["Avg"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Avg: %{text}<br>N=%{customdata}<extra></extra>",
                customdata=ag_stats["N"],
            ))
            fig_ag.add_trace(go.Scatter(
                y=ag_stats["AgeGroup"],
                x=ag_stats["Best"] / 3600,
                mode="markers",
                name="Best",
                marker=dict(color="#fbbf24", size=8, symbol="diamond"),
                hovertemplate="<b>%{y}</b> Best: %{text}<extra></extra>",
                text=[hm(v) for v in ag_stats["Best"]],
            ))
            fig_ag.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font_color="#9ca3af",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Hours", gridcolor="#374151"),
                yaxis=dict(gridcolor="#374151"),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_ag, width="stretch")

        st.markdown("---")

        # ── Year-over-Year Trends ────────────────────────────────────────────
        st.subheader("Year-over-Year Performance Trends")
        yearly = (
            df.groupby("Year")
            .agg(
                Avg_Total=("Overall_sec", "mean"),
                Avg_Swim=("Swim_sec", "mean"),
                Avg_Bike=("Bike_sec", "mean"),
                Avg_Run=("Run_sec", "mean"),
                Finishers=("Overall_sec", "count"),
            )
            .reset_index()
        )

        col_trend_l, col_trend_r = st.columns([3, 1])
        with col_trend_l:
            fig_trend = go.Figure()
            for seg, col_key in [("Swim", "Avg_Swim"), ("Bike", "Avg_Bike"), ("Run", "Avg_Run")]:
                fig_trend.add_trace(go.Scatter(
                    x=yearly["Year"],
                    y=yearly[col_key] / 3600,
                    mode="lines+markers",
                    name=seg,
                    line=dict(color=DISCIPLINE_COLORS[seg], width=2),
                    hovertemplate=f"<b>{seg}</b> %{{x}}: %{{text}}<extra></extra>",
                    text=[hm(v) for v in yearly[col_key]],
                ))
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font_color="#9ca3af",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Year", dtick=1, gridcolor="#374151"),
                yaxis=dict(title="Hours (avg)", gridcolor="#374151"),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_trend, width="stretch")

        with col_trend_r:
            st.markdown("#### Field Size")
            for _, row in yearly.iterrows():
                st.metric(str(int(row["Year"])), f"{int(row['Finishers']):,} finishers")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ATHLETE PERFORMANCE COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Athlete Performance Comparator")

    # Athlete search
    all_names = sorted(df["Name"].dropna().unique().tolist())
    sel_athletes = st.multiselect(
        "Search and select up to 4 athletes",
        options=all_names,
        max_selections=4,
        placeholder="Type a name …",
    )

    if not sel_athletes:
        st.info("Select at least one athlete from the list above to begin comparison.")
    else:
        athlete_df = df[df["Name"].isin(sel_athletes)].copy()

        # If an athlete raced multiple years, show latest
        athlete_rows = (
            athlete_df
            .sort_values("Year", ascending=False)
            .groupby("Name", as_index=False)
            .first()
        )

        # ── Summary cards ──────────────────────────────────────────────────
        cols = st.columns(len(athlete_rows))
        for i, (_, row) in enumerate(athlete_rows.iterrows()):
            with cols[i]:
                color = ATHLETE_PALETTE[i % len(ATHLETE_PALETTE)]
                st.markdown(
                    f"<div style='border:1px solid {color};border-radius:8px;"
                    f"padding:12px;background:rgba(0,0,0,0.3)'>"
                    f"<b style='color:{color}'>{row['Name']}</b><br>"
                    f"<span style='font-size:0.8em;color:#9ca3af'>"
                    f"{row.get('Division','?')} · {int(row['Year'])} · {row.get('Country','?')}</span><br>"
                    f"<span style='font-size:1.4em;color:#e5e7eb'>{hms(row['Overall_sec'])}</span><br>"
                    f"<span style='font-size:0.75em;color:#6b7280'>"
                    f"Rank #{int(row['Overall Rank']) if not pd.isna(row.get('Overall Rank')) else '?'}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Score normalisation for radar  ─────────────────────────────────
        # Score each discipline 0–100: 100 = fastest finisher, 0 = slowest
        def score_discipline(val, col_ref):
            valid = df[col_ref].dropna()
            if valid.empty or pd.isna(val):
                return 50.0
            worst = valid.quantile(0.95)
            best  = valid.quantile(0.05)
            if worst == best:
                return 50.0
            return float(np.clip((worst - val) / (worst - best) * 100, 0, 100))

        radar_cats = ["Swim", "Bike", "Run", "T1 Eff.", "T2 Eff."]
        col_map    = ["Swim_sec", "Bike_sec", "Run_sec", "T1_sec", "T2_sec"]

        fig_radar = go.Figure()
        for i, (_, row) in enumerate(athlete_rows.iterrows()):
            scores = [
                score_discipline(row["Swim_sec"], "Swim_sec"),
                score_discipline(row["Bike_sec"], "Bike_sec"),
                score_discipline(row["Run_sec"],  "Run_sec"),
                # Transitions: lower is better, score inverted
                score_discipline(row["T1_sec"],   "T1_sec"),
                score_discipline(row["T2_sec"],   "T2_sec"),
            ]
            # Prendiamo il colore base dell'atleta
        base_color = ATHLETE_PALETTE[i % len(ATHLETE_PALETTE)]
        
        # Convertiamo l'esadecimale (es. #10b981) in formato rgba(R, G, B, 0.15)
        h = base_color.lstrip('#')
        r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
        fill_rgba = f"rgba({r}, {g}, {b}, 0.15)"

        fig_radar.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            name=row["Name"],
            line_color=base_color,
            fillcolor=fill_rgba,
            opacity=0.85,
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], showticklabels=True, gridcolor="#374151"),
                angularaxis=dict(gridcolor="#374151"),
                bgcolor="rgba(17,24,39,0.6)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af",
            showlegend=True,
            height=420,
            legend=dict(orientation="h", y=-0.1),
        )

        # ── Split comparison bars ────────────────────────────────────────────
        bar_segs = [
            ("Swim", "Swim_sec"),
            ("T1",   "T1_sec"),
            ("Bike", "Bike_sec"),
            ("T2",   "T2_sec"),
            ("Run",  "Run_sec"),
        ]

        fig_bar = go.Figure()
        for seg, sec_col in bar_segs:
            fig_bar.add_trace(go.Bar(
                name=seg,
                x=[r["Name"] for _, r in athlete_rows.iterrows()],
                y=[r[sec_col] / 60 if not pd.isna(r[sec_col]) else 0
                   for _, r in athlete_rows.iterrows()],
                marker_color=DISCIPLINE_COLORS[seg],
                text=[hm(r[sec_col]) for _, r in athlete_rows.iterrows()],
                textposition="inside",
                hovertemplate=f"<b>{seg}</b><br>%{{x}}: %{{text}}<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.6)",
            font_color="#9ca3af",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#374151"),
            yaxis=dict(title="Minutes", gridcolor="#374151"),
            legend=dict(orientation="h", y=1.05),
            height=350,
        )

        col_r, col_b = st.columns(2)
        with col_r:
            st.subheader("Performance Radar")
            st.plotly_chart(fig_radar, width="stretch")
        with col_b:
            st.subheader("Split Breakdown")
            st.plotly_chart(fig_bar, width="stretch")

        # ── Detail table ────────────────────────────────────────────────────
        st.subheader("Split Details")
        detail_rows = []
        for _, row in athlete_rows.iterrows():
            detail_rows.append({
                "Athlete":      row["Name"],
                "Year":         int(row["Year"]),
                "Division":     row.get("Division", "?"),
                "Country":      row.get("Country", "?"),
                "Finish Time":  hms(row["Overall_sec"]),
                "Overall Rank": int(row["Overall Rank"]) if not pd.isna(row.get("Overall Rank")) else "?",
                "Swim":         hms(row["Swim_sec"]),
                "T1":           hms(row["T1_sec"]),
                "Bike":         hms(row["Bike_sec"]),
                "T2":           hms(row["T2_sec"]),
                "Run":          hms(row["Run_sec"]),
            })
        st.dataframe(
            pd.DataFrame(detail_rows),
            width="stretch",
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PACING STRATEGY ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Pacing Strategy Analyzer")
    st.caption(
        "Visualise bike-vs-run pacing trade-offs. "
        "Overbiker: fast bike to slow run. Underrunner: fast run relative to bike."
    )

    pa_df = df.dropna(subset=["Bike_sec", "Run_sec"]).copy()
    if pa_df.empty:
        st.warning("No valid Bike/Run data for current filters.")
    else:
        col_pa_l, col_pa_r = st.columns([1, 3])

        with col_pa_l:
            st.markdown("#### Filters")
            ag_options = ["All"] + sorted(pa_df["AgeGroup"].dropna().unique().tolist())
            pa_ag = st.selectbox("Age Group", ag_options, key="pa_ag")
            pa_year = st.selectbox(
                "Year",
                ["All"] + [str(y) for y in sorted(pa_df["Year"].unique())],
                key="pa_year",
            )
            sample_n = st.slider("Max points plotted", 200, 2000, 800, 100, key="pa_n")

            if pa_ag != "All":
                pa_df = pa_df[pa_df["AgeGroup"] == pa_ag]
            if pa_year != "All":
                pa_df = pa_df[pa_df["Year"] == int(pa_year)]

            bike_med = pa_df["Bike_sec"].median()
            run_med  = pa_df["Run_sec"].median()

            def pacing_label(row):
                b_fast = row["Bike_sec"] < bike_med
                r_fast = row["Run_sec"]  < run_med
                if b_fast and r_fast:
                    return "Elite"
                if b_fast and not r_fast:
                    return "Overbiker"
                if not b_fast and r_fast:
                    return "Strong Runner"
                return "Conservative"

            pa_df["PacingType"] = pa_df.apply(pacing_label, axis=1)

            label_counts = pa_df["PacingType"].value_counts()
            st.markdown("#### Pacing Breakdown")
            pacing_colors = {
                "Elite":         "#fbbf24",
                "Overbiker":     "#ef4444",
                "Strong Runner": "#10b981",
                "Conservative":  "#6b7280",
            }
            for label, cnt in label_counts.items():
                pct = cnt / len(pa_df) * 100
                color = pacing_colors.get(label, "#9ca3af")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"border-left:3px solid {color};padding-left:8px;margin-bottom:6px'>"
                    f"<span style='color:{color}'>{label}</span>"
                    f"<span style='color:#9ca3af'>{pct:.0f}%</span></div>",
                    unsafe_allow_html=True,
                )

        with col_pa_r:
            sample = pa_df.sample(min(sample_n, len(pa_df)), random_state=42)

            fig_sc = go.Figure()
            for label, grp in sample.groupby("PacingType"):
                fig_sc.add_trace(go.Scatter(
                    x=grp["Bike_sec"] / 3600,
                    y=grp["Run_sec"]  / 3600,
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=pacing_colors.get(label, "#9ca3af"),
                        size=6,
                        opacity=0.7,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Bike: %{customdata[0]}<br>"
                        "Run: %{customdata[1]}<extra></extra>"
                    ),
                    text=grp["Name"].fillna("Unknown"),
                    customdata=list(zip(
                        [hm(v) for v in grp["Bike_sec"]],
                        [hm(v) for v in grp["Run_sec"]],
                    )),
                ))

            # Quadrant lines
            fig_sc.add_vline(x=bike_med / 3600, line_dash="dash", line_color="#4b5563",
                             annotation_text="Bike Median", annotation_font_color="#6b7280")
            fig_sc.add_hline(y=run_med / 3600, line_dash="dash", line_color="#4b5563",
                             annotation_text="Run Median", annotation_font_color="#6b7280")

            fig_sc.update_layout(
                title="Bike vs Run Scatter – Pacing Quadrants",
                xaxis_title="Bike Time (hours)",
                yaxis_title="Run Time (hours)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font_color="#9ca3af",
                xaxis=dict(gridcolor="#374151"),
                yaxis=dict(gridcolor="#374151"),
                legend=dict(orientation="h", y=1.08),
                height=500,
            )
            st.plotly_chart(fig_sc, width="stretch")

        # ── Swim-to-Bike-to-Run pacing funnel ─────────────────────────────
        st.subheader("Leg-Time Correlation: Does a fast swim lead to a faster finish?")
        corr_seg = st.selectbox(
            "Select discipline to correlate with finish time",
            ["Swim_sec", "Bike_sec", "Run_sec"],
            format_func=lambda x: x.replace("_sec", ""),
            key="corr_seg",
        )
        corr_df = pa_df.dropna(subset=[corr_seg, "Overall_sec"])
        corr_df = corr_df.sample(min(1000, len(corr_df)), random_state=1)

        fig_corr = px.scatter(
            corr_df,
            x=corr_df[corr_seg] / 3600,
            y=corr_df["Overall_sec"] / 3600,
            color="AgeGroup",
            hover_name="Name",
            trendline="ols",
            labels={
                "x": f"{corr_seg.replace('_sec','')} Time (hrs)",
                "y": "Finish Time (hrs)",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.6)",
            font_color="#9ca3af",
            height=380,
            xaxis=dict(gridcolor="#374151"),
            yaxis=dict(gridcolor="#374151"),
        )
        st.plotly_chart(fig_corr, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE TIME MODEL
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Predictive Time Model")
    st.caption(
        "Enter your target splits to estimate your finish time and "
        "how you compare against the historical field."
    )

    col_inp, col_out = st.columns([1, 2])

    with col_inp:
        st.markdown("#### Your Profile")
        pred_gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
        pred_ag_opts = sorted(
            df_all[df_all["Gender_clean"] == pred_gender]["AgeGroup"].dropna().unique().tolist()
        )
        pred_ag = st.selectbox("Age Group", pred_ag_opts if pred_ag_opts else ["PRO"], key="pred_ag")

        st.markdown("#### Enter Target Times")
        st.caption("Format: HH:MM:SS")

        def time_input(label, default):
            raw = st.text_input(label, default, key=f"pred_{label}")
            secs = parse_time(raw)
            if pd.isna(secs):
                st.warning(f"Invalid time for {label}")
            return secs

        pred_swim = time_input("Swim (3.8 km)", "01:10:00")
        pred_t1   = time_input("Transition 1",  "00:03:30")
        pred_bike = time_input("Bike (180 km)", "05:30:00")
        pred_t2   = time_input("Transition 2",  "00:02:30")
        pred_run  = time_input("Run (42.2 km)", "04:00:00")

        valid_inputs = all(not pd.isna(v) for v in [pred_swim, pred_t1, pred_bike, pred_t2, pred_run])
        pred_total = sum([v for v in [pred_swim, pred_t1, pred_bike, pred_t2, pred_run] if not pd.isna(v)])

    with col_out:
        if not valid_inputs:
            st.info("Fix the time inputs on the left to see predictions.")
        else:
            # Reference group
            ref = df_all[
                (df_all["Gender_clean"] == pred_gender) &
                (df_all["AgeGroup"] == pred_ag)
            ].dropna(subset=["Overall_sec"])

            ref_all = df_all.dropna(subset=["Overall_sec"])

            # ── Percentile metrics ─────────────────────────────────────
            pct_overall = percentile_rank(pred_total, ref_all["Overall_sec"])
            pct_ag      = percentile_rank(pred_total, ref["Overall_sec"]) if not ref.empty else 50.0
            pct_swim    = percentile_rank(pred_swim,  df_all["Swim_sec"].dropna()) if not pd.isna(pred_swim) else 50.0
            pct_bike    = percentile_rank(pred_bike,  df_all["Bike_sec"].dropna()) if not pd.isna(pred_bike) else 50.0
            pct_run     = percentile_rank(pred_run,   df_all["Run_sec"].dropna())  if not pd.isna(pred_run)  else 50.0

            st.markdown("#### Predicted Finish")
            st.markdown(
                f"<div style='font-size:3rem;font-weight:700;color:#10b981;text-align:center'>"
                f"{hms(pred_total)}</div>",
                unsafe_allow_html=True,
            )

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Overall Percentile",    f"Top {100 - pct_overall:.0f}%")
            mc2.metric(f"{pred_ag} Percentile", f"Top {100 - pct_ag:.0f}%")
            mc3.metric("Field Size (ref)",       f"{len(ref):,}" if not ref.empty else "N/A")

            # ── Your splits vs. age group benchmarks ──────────────────
            st.markdown("#### Your Splits vs. Age Group Benchmarks")
            if not ref.empty:
                bench_swim = ref["Swim_sec"].median()
                bench_bike = ref["Bike_sec"].median()
                bench_run  = ref["Run_sec"].median()
                bench_t1   = ref["T1_sec"].median()
                bench_t2   = ref["T2_sec"].median()

                comp_data = pd.DataFrame({
                    "Segment":   ["Swim", "T1", "Bike", "T2", "Run"],
                    "You (min)": [
                        (pred_swim or 0) / 60,
                        (pred_t1   or 0) / 60,
                        (pred_bike or 0) / 60,
                        (pred_t2   or 0) / 60,
                        (pred_run  or 0) / 60,
                    ],
                    "Median (min)": [
                        bench_swim / 60 if not np.isnan(bench_swim) else 0,
                        bench_t1   / 60 if not np.isnan(bench_t1)   else 0,
                        bench_bike / 60 if not np.isnan(bench_bike) else 0,
                        bench_t2   / 60 if not np.isnan(bench_t2)   else 0,
                        bench_run  / 60 if not np.isnan(bench_run)  else 0,
                    ],
                    "Color": ["#38bdf8", "#a78bfa", "#34d399", "#fb923c", "#f87171"],
                })

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name="Median of Age Group",
                    x=comp_data["Segment"],
                    y=comp_data["Median (min)"],
                    marker_color="#4b5563",
                    hovertemplate="Median %{x}: %{y:.1f} min<extra></extra>",
                ))
                fig_comp.add_trace(go.Bar(
                    name="Your Target",
                    x=comp_data["Segment"],
                    y=comp_data["You (min)"],
                    marker_color=comp_data["Color"].tolist(),
                    hovertemplate="Your %{x}: %{y:.1f} min<extra></extra>",
                ))
                fig_comp.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,24,39,0.6)",
                    font_color="#9ca3af",
                    height=320,
                    xaxis=dict(gridcolor="#374151"),
                    yaxis=dict(title="Minutes", gridcolor="#374151"),
                    legend=dict(orientation="h", y=1.05),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_comp, width="stretch")

            # ── Finish time distribution with marker ───────────────────
            if not ref.empty:
                st.markdown(f"#### Where You'd Finish Among {pred_ag} ({pred_gender})")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=ref["Overall_sec"] / 3600,
                    nbinsx=40,
                    marker_color="#10b981",
                    opacity=0.7,
                    name="Age Group",
                ))
                fig_dist.add_vline(
                    x=pred_total / 3600,
                    line_color="#fbbf24",
                    line_width=2,
                    line_dash="dash",
                    annotation_text=f"You: {hms(pred_total)}",
                    annotation_font_color="#fbbf24",
                )
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,24,39,0.6)",
                    font_color="#9ca3af",
                    height=280,
                    xaxis=dict(title="Finish Time (hrs)", gridcolor="#374151"),
                    yaxis=dict(title="Athletes", gridcolor="#374151"),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_dist, width="stretch")

            # ── Per-discipline percentiles ─────────────────────────────
            st.markdown("#### Discipline Percentiles")
            dp1, dp2, dp3 = st.columns(3)
            dp1.metric("Swim Percentile", f"Top {100 - pct_swim:.0f}%",
                       help="vs. all finishers in filtered dataset")
            dp2.metric("Bike Percentile", f"Top {100 - pct_bike:.0f}%")
            dp3.metric("Run Percentile",  f"Top {100 - pct_run:.0f}%")


# ═══════════════════════════════════════════════════════════════════��══════════
# TAB 5 — RACE STRATEGY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Race Strategy Builder")
    st.caption(
        "Set a target finish time and let the builder reverse-engineer "
        "optimal splits from real race data."
    )

    col_sb_l, col_sb_r = st.columns([1, 2])

    with col_sb_l:
        st.markdown("#### Athlete Profile")
        sb_gender = st.selectbox("Gender", ["Male", "Female"], key="sb_gender")
        sb_ag_opts = sorted(
            df_all[df_all["Gender_clean"] == sb_gender]["AgeGroup"].dropna().unique().tolist()
        )
        sb_ag = st.selectbox("Age Group", sb_ag_opts if sb_ag_opts else ["PRO"], key="sb_ag")
        target_str = st.text_input("Target Finish Time (HH:MM:SS)", "10:00:00", key="sb_target")
        target_sec = parse_time(target_str)

        sb_weight = st.slider("Athlete weight (kg) – for power estimates", 50, 110, 75, key="sb_weight")
        st.markdown("---")
        st.markdown("#### Strategy Bias")
        swim_bias = st.slider("Swim effort vs. median (%)", -20, 20, 0, key="sb_swim_bias",
                              help="Negative = faster than median for this segment's proportion")
        bike_bias = st.slider("Bike effort vs. median (%)", -20, 20, 0, key="sb_bike_bias")
        run_bias  = st.slider("Run effort vs. median (%)",  -20, 20, 0, key="sb_run_bias")

    with col_sb_r:
        if pd.isna(target_sec) or target_sec <= 0:
            st.warning("Enter a valid target time (HH:MM:SS) on the left.")
        else:
            ref_sb = df_all[
                (df_all["Gender_clean"] == sb_gender) &
                (df_all["AgeGroup"] == sb_ag)
            ].dropna(subset=["Overall_sec", "Swim_sec", "Bike_sec", "Run_sec"])

            # ── Calculate split targets ───────────────────────────────
            # Use median proportions from reference group, then scale to target
            if ref_sb.empty:
                st.warning("No reference data for this gender/age group combination.")
            else:
                # Median segment proportions (excluding transitions in proportion calc)
                med_swim = ref_sb["Swim_sec"].median()
                med_bike = ref_sb["Bike_sec"].median()
                med_run  = ref_sb["Run_sec"].median()
                med_t1   = ref_sb["T1_sec"].dropna().median()
                med_t2   = ref_sb["T2_sec"].dropna().median()

                # Fix transitions (not scaled with speed)
                t1_target = med_t1 if not np.isnan(med_t1) else 210
                t2_target = med_t2 if not np.isnan(med_t2) else 150

                # Remaining time for SBR after transitions
                sbr_target = target_sec - t1_target - t2_target

                # Proportions with bias
                denom = med_swim + med_bike + med_run
                p_swim = (med_swim / denom) * (1 + swim_bias / 100)
                p_bike = (med_bike / denom) * (1 + bike_bias / 100)
                p_run  = (med_run  / denom) * (1 + run_bias  / 100)
                total_p = p_swim + p_bike + p_run

                swim_target = sbr_target * (p_swim / total_p)
                bike_target = sbr_target * (p_bike / total_p)
                run_target  = sbr_target * (p_run  / total_p)

                calc_total = swim_target + t1_target + bike_target + t2_target + run_target

                # ── Feasibility vs. reference group ──────────────────────
                pct_vs_ag = percentile_rank(target_sec, ref_sb["Overall_sec"])
                achievers = (ref_sb["Overall_sec"] <= target_sec).sum()
                pct_achievers = achievers / len(ref_sb) * 100

                st.markdown("#### Projected Split Plan")
                st.markdown(
                    f"<div style='font-size:2.5rem;font-weight:700;color:#10b981;text-align:center'>"
                    f"{hms(calc_total)}</div>"
                    f"<div style='text-align:center;color:#6b7280;font-size:0.85em'>"
                    f"Projected finish · Target was {hms(target_sec)}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # ── Strategy table ────────────────────────────────────────
                swim_speed_mh = (3.8 / swim_target * 3600) * 1000  # m/h
                bike_speed    = 180 / bike_target * 3600
                run_pace_km   = run_target / 42.195

                # Rough power estimate
                v_ms = bike_speed / 3.6
                power_w = int(0.3 * v_ms ** 3 + 35 + (sb_weight - 70) * 0.5)

                strategy_df = pd.DataFrame([
                    {
                        "Segment":      "🏊 Swim",
                        "Target Time":  hms(swim_target),
                        "Metric":       f"{pace_per_km(swim_target, 3.8)}",
                        "vs. Median":   f"{((swim_target - med_swim) / med_swim * 100):+.1f}%",
                        "Percentile":   f"Top {100 - percentile_rank(swim_target, ref_sb['Swim_sec']):.0f}%",
                    },
                    {
                        "Segment":      "⚡ T1",
                        "Target Time":  hms(t1_target),
                        "Metric":       "—",
                        "vs. Median":   f"{((t1_target - med_t1) / med_t1 * 100 if not np.isnan(med_t1) else 0):+.1f}%",
                        "Percentile":   "—",
                    },
                    {
                        "Segment":      "🚴 Bike",
                        "Target Time":  hms(bike_target),
                        "Metric":       f"{speed_kmh(bike_target, 180)} · ~{power_w}W",
                        "vs. Median":   f"{((bike_target - med_bike) / med_bike * 100):+.1f}%",
                        "Percentile":   f"Top {100 - percentile_rank(bike_target, ref_sb['Bike_sec']):.0f}%",
                    },
                    {
                        "Segment":      "⚡ T2",
                        "Target Time":  hms(t2_target),
                        "Metric":       "—",
                        "vs. Median":   f"{((t2_target - med_t2) / med_t2 * 100 if not np.isnan(med_t2) else 0):+.1f}%",
                        "Percentile":   "—",
                    },
                    {
                        "Segment":      "🏃 Run",
                        "Target Time":  hms(run_target),
                        "Metric":       f"{pace_per_km(run_target, 42.195)}",
                        "vs. Median":   f"{((run_target - med_run) / med_run * 100):+.1f}%",
                        "Percentile":   f"Top {100 - percentile_rank(run_target, ref_sb['Run_sec']):.0f}%",
                    },
                ])
                st.dataframe(strategy_df, width="stretch", hide_index=True)

                # ── Feasibility ───────────────────────────────────────────
                st.markdown("#### Feasibility Analysis")
                fa1, fa2, fa3 = st.columns(3)
                fa1.metric(
                    "Athletes who hit this target",
                    f"{achievers:,} / {len(ref_sb):,}",
                    delta=f"{pct_achievers:.1f}% of {sb_ag}",
                )
                fa2.metric("Field Percentile", f"Top {100 - pct_vs_ag:.0f}%")
                fa3.metric("Reference Field", f"{len(ref_sb):,} athletes")

                # ── Visual split bar ──────────────────────────────────────
                fig_plan = go.Figure()
                plan_segs = [
                    ("Swim", swim_target, "#38bdf8"),
                    ("T1",   t1_target,   "#a78bfa"),
                    ("Bike", bike_target, "#34d399"),
                    ("T2",   t2_target,   "#fb923c"),
                    ("Run",  run_target,  "#f87171"),
                ]
                plan_med = [
                    ("Swim", med_swim, "#38bdf8"),
                    ("T1",   med_t1 if not np.isnan(med_t1) else 0, "#a78bfa"),
                    ("Bike", med_bike, "#34d399"),
                    ("T2",   med_t2 if not np.isnan(med_t2) else 0, "#fb923c"),
                    ("Run",  med_run,  "#f87171"),
                ]

                for (seg, val, col) in plan_segs:
                    fig_plan.add_trace(go.Bar(
                        name=seg,
                        x=["Your Plan"],
                        y=[val / 60],
                        marker_color=col,
                        text=hm(val),
                        textposition="inside",
                        hovertemplate=f"<b>{seg}</b>: {hm(val)}<extra></extra>",
                    ))
                for (seg, val, col) in plan_med:
                    fig_plan.add_trace(go.Bar(
                        name=f"{seg} (Median)",
                        x=["Age Group Median"],
                        y=[val / 60 if not pd.isna(val) else 0],
                        marker_color=col,
                        marker_pattern_shape="/",
                        text=hm(val),
                        textposition="inside",
                        hovertemplate=f"<b>{seg} median</b>: {hm(val)}<extra></extra>",
                        showlegend=False,
                    ))

                fig_plan.update_layout(
                    barmode="stack",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,24,39,0.6)",
                    font_color="#9ca3af",
                    height=320,
                    xaxis=dict(gridcolor="#374151"),
                    yaxis=dict(title="Minutes", gridcolor="#374151"),
                    legend=dict(orientation="h", y=1.08),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_plan, width="stretch")

                # ── Coaching insights ──────────────────────────────────────
                st.markdown("#### Coaching Insights")
                insights = []
                if swim_target < med_swim:
                    insights.append(("✅", "Swim", "You're targeting a swim faster than your age group median. Conserve energy for the bike and run."))
                else:
                    insights.append(("⚠️", "Swim", "Your swim target is slower than the median. A faster swim could improve your position early in the race."))
                if bike_target < med_bike * 0.95:
                    insights.append(("⚠️", "Bike", "You're targeting a significantly faster bike than median. Be cautious of overbiker syndrome – a hard bike often leads to a slow run."))
                elif bike_target < med_bike:
                    insights.append(("✅", "Bike", "Bike target is slightly faster than median – a solid position to hold."))
                else:
                    insights.append(("ℹ️", "Bike", "Conservative bike target leaves energy for a strong run."))
                if run_target < med_run:
                    insights.append(("✅", "Run", "Strong run target – if your legs are fresh off the bike, this is achievable."))
                else:
                    insights.append(("ℹ️", "Run", "Run target is at or above median. Focus on nutrition and pacing on the bike to maximise run potential."))

                for emoji, seg, text in insights:
                    st.markdown(
                        f"<div style='border-left:3px solid #374151;padding:8px 12px;"
                        f"margin-bottom:8px;color:#d1d5db'>"
                        f"<b>{emoji} {seg}:</b> {text}</div>",
                        unsafe_allow_html=True,
                    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Ironman Race Analytics · Data: Ironman World Championship 2003–2025 · "
    "Built with Streamlit & Plotly"
)
