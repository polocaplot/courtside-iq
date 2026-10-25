"""COURTSIDE / custom NBA simulation and fantasy research dashboard.

Run: streamlit run app.py
Data-source failures never silently become fabricated live NBA data.
"""
from __future__ import annotations

import html
import time
import os
import json
import re
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import (actual_boxscore, current_roster, demo_dataset, injury_status, injuries,
                          normalize_name, normalize_players, normalize_teams, previous_season_players,
                          reconcile_matchup_rosters, schedule, season_stats)
from fantasy import FantasyScoring, explain_fantasy_line, fantasy_values, leaderboard
from model import Config, prop_probability, run
from ensemble import blended_inputs, role_usage_adjustments, prediction_deltas
from tracking import grade_snapshot, load_history, store_snapshot
from autosync import make_snapshot, record as record_feed, history as sync_history, last_check as sync_last_check, reconcile_editor
from streamlit_autorefresh import st_autorefresh
from training import DEFAULT_FOLDER as TRAINING_FOLDER
from training import fetch_historical, train as train_models, forecast as forecast_ml, load as load_trained
from retrain import update_models
from model_bootstrap import bootstrap_model, DEFAULT_SEASONS
from markets import fetch_events as fetch_market_events, fetch_event_odds, parse_props, compare_props, match_event
from shot_intelligence import (ZONES, fetch_team_shots, fetch_player_shots, historical_seasons,
                               merge_historical_profiles, validate_profiles, validate_defense, player_zone_model)
from lineup_engine import validate_lineups
from defender_intelligence import validate_defenders
from generalization import sample_reliability
from calibration import evaluate_calibration
from advanced_feeds import read_cache as read_advanced_cache, fetch_public_feeds, should_refresh as advanced_refresh_due
from automatic_audit import save_if_due, grade_completed, grade_summary, calibration_frame
from diagnostic_bundle import make_report
from nba_archive import (DB_PATH as ARCHIVE_PATH, connect as archive_connect, overview as archive_overview,
                         apply_archive_priors, team_context as archive_team_context,
                         historical_context as archive_player_context)
import subprocess
import sys

st.set_page_config(page_title="COURTSIDE IQ — NBA Intelligence", page_icon="🏀",
                   layout="wide", initial_sidebar_state="expanded")

CLOUD_MODE = os.getenv("COURTSIDE_IQ_CLOUD", "0") == "1"
APP_DIR = Path.home() / ".courtsideiq"
SETTINGS_PATH = APP_DIR / "settings.json"
COLORS = {"Electric violet": "#9575FF", "Ocean teal": "#00CFA6", "Cobalt": "#4C93FF",
          "Hot coral": "#FF6B7B", "Amber": "#E7AD54", "Ice blue": "#75D6FF", "Custom": "#9575FF"}
NAV = ["Command Center", "Fantasy Lab", "Box Score", "Player Lab", "Shot Lab", "Prop Lab",
       "Scenario Lab", "Live Sync", "NBA Database", "COURTSIDE VISION", "Accuracy Lab", "ML Studio", "Market Lab", "Appearance"]
DEFAULTS = {"theme": "Midnight", "accent": "Electric violet", "density": "Comfortable",
            "corners": "Rounded", "chart": "Bars", "show_bands": True, "show_insights": True,
            "show_distribution": True, "n": 150_000, "seed": 42,
            "auto_sync": True, "sync_minutes": 5, "auto_rerun": True,
            "watch_scope": "Selected matchup", "custom_hex": "#9575FF",
            "font_size": "Standard", "auto_first_run": False, "ml_active": True, "ml_auto_update": True,
            "shot_mode": True, "possession_mode": True, "defense_strength": 1.0, "advanced_auto": True,
            "audit_auto": True, "grade_auto": True, "smart_mode": False, "smart_role": True,
            "archive_auto": True, "archive_backfill": True}


def settings_read() -> dict:
    try:
        loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return {**DEFAULTS, **{k: v for k, v in loaded.items() if k in DEFAULTS}}
    except (OSError, ValueError):
        return DEFAULTS.copy()


def settings_write(d: dict):
    if CLOUD_MODE:
        return  # Session-only preferences; do not leak across cloud visitors.
    APP_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")


if "settings" not in st.session_state:
    if CLOUD_MODE:
        # Shared Streamlit hosting has one filesystem for all visitors. Never
        # read another visitor's saved appearance / simulation preferences.
        cloud_defaults = DEFAULTS.copy()
        cloud_defaults.update({
            "auto_sync": False, "auto_rerun": False, "auto_first_run": False,
            "advanced_auto": False, "audit_auto": False, "grade_auto": False,
            "archive_auto": False, "archive_backfill": False,
            "ml_auto_update": False,
        })
        st.session_state.settings = cloud_defaults
    else:
        st.session_state.settings = settings_read()
S = st.session_state.settings


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def apply_theme():
    dark = S["theme"] == "Midnight"
    custom = str(S.get("custom_hex", "#9575FF"))
    accent = custom if S["accent"] == "Custom" and re.fullmatch(r"#[0-9a-fA-F]{6}", custom) else COLORS.get(S["accent"], "#9575FF")
    radius = "18px" if S["corners"] == "Rounded" else "7px"
    bg = "#090E19" if dark else "#F3F6FB"
    surface = "#11192A" if dark else "#FFFFFF"
    surface2 = "#172238" if dark else "#E9EEF7"
    fg = "#ECF1FA" if dark else "#182236"
    muted = "#9FAFC8" if dark else "#60718E"
    border = "#243149" if dark else "#D9E1ED"
    pad = "0.65rem" if S["density"] == "Compact" else "1.05rem"
    base_font = {"Small": "14px", "Standard": "16px", "Large": "18px"}.get(S.get("font_size"), "16px")
    st.markdown(f"""
    <style>
    :root {{ --cs-accent:{accent}; --cs-base-font:{base_font}; --cs-bg:{bg}; --cs-surface:{surface};
      --cs-surface-2:{surface2}; --cs-text:{fg}; --cs-muted:{muted};
      --cs-border:{border}; --cs-radius:{radius}; --cs-pad:{pad}; }}
    html, body, [data-testid="stAppViewContainer"], .stApp {{ background:var(--cs-bg); color:var(--cs-text); font-size:var(--cs-base-font); }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{ background:var(--cs-surface) !important; }}
    .block-container {{ padding-top:1.2rem; max-width:1700px; }}
    h1,h2,h3,h4,p,small {{ color:var(--cs-text); }}
    .stMarkdown p {{ color:var(--cs-text); }}
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {{ color:var(--cs-muted) !important; }}
    .brand {{ font-weight:900; font-size:1.6rem; letter-spacing:-.065em; color:var(--cs-text); }}
    .brand em {{ color:var(--cs-accent); font-style:normal; }}
    .eyebrow {{ font-size:.73rem; text-transform:uppercase; letter-spacing:.17em; color:var(--cs-accent); font-weight:800; }}
    .hero {{ border:1px solid var(--cs-border); border-radius:var(--cs-radius); padding:1.35rem 1.6rem;
       background:linear-gradient(116deg,{surface}, {surface2}); margin-bottom:1rem; }}
    .hero h1 {{ margin:.25rem 0 .4rem; font-size:2.25rem; letter-spacing:-.055em; line-height:1.11; font-weight:850; }}
    .hero p {{ color:var(--cs-muted); margin:0; font-size:.94rem; }}
    .glass {{ border:1px solid var(--cs-border); border-radius:var(--cs-radius);
        background:var(--cs-surface); padding:var(--cs-pad); margin:.12rem 0; }}
    .kicker {{ color:var(--cs-muted); font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.065em; }}
    .stat {{ font-size:1.9rem; font-weight:860; letter-spacing:-.05em; margin:.25rem 0; color:var(--cs-text); }}
    .micro {{ font-size:.75rem; color:var(--cs-muted); }}
    .pill {{ display:inline-block; font-size:.72rem; padding:.23rem .55rem; margin:.2rem .3rem .2rem 0;
       border:1px solid var(--cs-border); border-radius:50px; color:var(--cs-muted); font-weight:700; }}
    .pill.live {{ border-color:var(--cs-accent); color:var(--cs-accent); }}
    .section-title {{ font-size:1.13rem; font-weight:800; letter-spacing:-.035em; margin:1.1rem 0 .5rem; }}
    .stButton>button[kind="primary"] {{ background:var(--cs-accent); color:#08101C; border:0;
       font-weight:800; border-radius:11px; }}
    .stButton>button {{ border-radius:11px; border-color:var(--cs-border); font-weight:650; }}
    div[data-testid="stMetric"] {{ background:var(--cs-surface); border:1px solid var(--cs-border);
       padding:.9rem 1.1rem; border-radius:var(--cs-radius); }}
    div[data-testid="stMetricLabel"] p {{ color:var(--cs-muted) !important; }}
    [data-testid="stDataFrame"], div[data-testid="stExpander"] {{ border-color:var(--cs-border); border-radius:var(--cs-radius); }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:var(--cs-radius); }}
    .stTabs [data-baseweb="tab"] {{ color:var(--cs-muted); font-weight:700; }}
    .stTabs [aria-selected="true"] {{ color:var(--cs-accent); }}
    [data-testid="stSidebarNav"] {{ display:none; }}
    .footer {{ color:var(--cs-muted); font-size:.74rem; padding-top:1.5rem; border-top:1px solid var(--cs-border); margin-top:2rem; }}

    /* COURTSIDE IQ 2.0 — visual redesign only; simulation code untouched */
    .stApp {{ background: radial-gradient(ellipse 55% 30% at 88% 0%, color-mix(in srgb,var(--cs-accent) 9%, transparent),transparent 75%),var(--cs-bg) !important; }}
    .block-container {{ max-width:1580px !important; padding:2.2rem 2.4rem 5rem !important; }}
    [data-testid="stSidebar"] {{ border-right:1px solid var(--cs-border) !important; }}
    [data-testid="stSidebar"] > div:first-child {{ padding:1.1rem .65rem 2rem !important; }}
    [data-testid="stSidebar"] .brand {{ font-size:1.8rem; letter-spacing:-.075em; padding:.4rem .4rem .05rem; }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ padding:0 .4rem; font-size:.68rem; letter-spacing:.10em; }}
    [data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.2rem; padding:.6rem 0; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label {{ border-radius:11px; padding:.55rem .7rem; margin:0; transition:background .15s ease, transform .15s ease; border:1px solid transparent; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background:var(--cs-surface-2); transform:translateX(2px); }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background:color-mix(in srgb,var(--cs-accent) 14%,var(--cs-surface)); border-color:color-mix(in srgb,var(--cs-accent) 30%,transparent); }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color:var(--cs-accent) !important; font-weight:800; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size:.88rem; font-weight:630; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] {{ margin-left:.2rem; }}
    .hero {{ position:relative; overflow:hidden; padding:2.1rem 2.3rem !important; border-radius:24px !important;
       background:linear-gradient(118deg,color-mix(in srgb,var(--cs-accent) 12%,var(--cs-surface)) 0%,var(--cs-surface) 56%,var(--cs-surface-2) 100%) !important;
       box-shadow:0 16px 42px rgba(0,0,0,.12); border-color:color-mix(in srgb,var(--cs-accent) 25%,var(--cs-border)) !important; }}
    .hero:after {{ content:'IQ'; position:absolute; font-size:12rem; font-weight:1000; letter-spacing:-.12em; line-height:1;
       right:3rem; top:-1.5rem; color:var(--cs-accent); opacity:.055; pointer-events:none; }}
    .hero h1 {{ font-size:clamp(2rem,3.2vw,3.25rem) !important; letter-spacing:-.065em !important; max-width:950px; }}
    .hero p {{ font-size:1.01rem !important; max-width:830px; line-height:1.65; }}
    .eyebrow {{ font-size:.70rem !important; letter-spacing:.19em !important; margin-bottom:.7rem; }}
    .glass {{ min-height:116px; border-radius:18px !important; padding:1.15rem 1.3rem !important;
      box-shadow:0 5px 24px rgba(0,0,0,.07); transition:transform .18s ease,border-color .18s ease; }}
    .glass:hover {{ transform:translateY(-2px); border-color:color-mix(in srgb,var(--cs-accent) 36%,var(--cs-border)) !important; }}
    .stat {{ font-size:2.2rem !important; font-variant-numeric:tabular-nums; }}
    .section-title {{ font-size:1.24rem !important; margin:1.65rem 0 .75rem !important; letter-spacing:-.045em !important; }}
    div[data-testid="stMetric"] {{ border-radius:17px !important; padding:1.2rem 1.3rem !important; box-shadow:0 6px 22px rgba(0,0,0,.06); }}
    div[data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; letter-spacing:-.045em; }}
    [data-testid="stDataFrame"] {{ border:1px solid var(--cs-border); border-radius:15px !important; overflow:hidden; }}
    [data-testid="stExpander"] {{ border-radius:14px !important; overflow:hidden; }}
    [data-testid="stExpander"] summary:hover {{ background:color-mix(in srgb,var(--cs-accent) 5%,transparent); }}
    .stButton>button {{ min-height:2.65rem; border-radius:12px !important; transition:filter .15s ease,transform .15s ease; }}
    .stButton>button:hover {{ filter:brightness(1.1); transform:translateY(-1px); }}
    .stButton>button[kind="primary"] {{ box-shadow:0 7px 18px color-mix(in srgb,var(--cs-accent) 25%,transparent); }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid var(--cs-border); }}
    .stTabs [data-baseweb="tab"] {{ padding:.65rem 1.1rem; border-radius:10px 10px 0 0; }}
    .stTabs [aria-selected="true"] {{ background:color-mix(in srgb,var(--cs-accent) 9%,transparent); }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div {{ border-radius:11px !important; }}
    .js-plotly-plot {{ border-radius:16px; overflow:hidden; }}
    @media (max-width:850px) {{
      .block-container {{ padding:.75rem .7rem 5rem !important; }}
      .hero {{ padding:1.1rem !important; border-radius:16px !important; }}
      .hero h1 {{ font-size:1.75rem !important; }}
      .hero:after {{ display:none; }}
      .glass {{ min-height:0 !important; padding:.8rem !important; }}
      .stat {{ font-size:1.65rem !important; }}
      div[data-testid="stMetric"] {{ padding:.7rem !important; }}
      div[data-testid="stMetricValue"] {{ font-size:1.5rem !important; }}
      .stButton>button {{ min-height:2.9rem; font-size:.85rem; }}
      .stTabs [data-baseweb="tab-list"] {{ overflow-x:auto; flex-wrap:nowrap; scrollbar-width:thin; }}
      .stTabs [data-baseweb="tab"] {{ flex-shrink:0; padding:.55rem .75rem; }}
      [data-testid="stDataFrame"] {{ max-width:100%; }}
      .js-plotly-plot {{ max-width:100%; }}
      [data-testid="stSidebar"] > div:first-child {{ padding:.7rem .5rem 1rem !important; }}
    }}
    </style>
    """, unsafe_allow_html=True)
    return {"background": bg, "paper": surface, "font": fg, "muted": muted, "accent": accent, "border": border}


PALETTE = apply_theme()


def hero(title: str, subtitle: str, tag: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">{escape(tag)}</div>'
                f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def metric_card(label: str, main: str, sub: str = ""):
    st.markdown(f'<div class="glass"><div class="kicker">{escape(label)}</div>'
                f'<div class="stat">{escape(main)}</div><div class="micro">{escape(sub)}</div></div>', unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def plot_hist(values: np.ndarray, title: str, threshold: float | None = None, bins: int = 40):
    # Bin on the server so a 150,000-draw chart sends just 40 plotting points
    # to the browser, rather than 150,000 raw observations.
    arr = np.asarray(values, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    x = (edges[1:] + edges[:-1]) / 2
    if S["chart"] == "Curve":
        fig = go.Figure(go.Scatter(x=x, y=counts, mode="lines", fill="tozeroy",
                                   line=dict(color=PALETTE["accent"], width=3), name=title))
    else:
        fig = go.Figure(go.Bar(x=x, y=counts, width=np.diff(edges) * .93,
                               marker_color=PALETTE["accent"], opacity=.9, name=title))
    if threshold is not None:
        fig.add_vline(x=float(threshold), line_dash="dash", line_color="#F59E7D")
    fig.update_layout(title=title, template="plotly_dark" if S["theme"] == "Midnight" else "plotly_white",
                      height=305, margin=dict(l=8, r=8, t=48, b=8),
                      paper_bgcolor=PALETTE["paper"], plot_bgcolor=PALETTE["paper"],
                      font=dict(color=PALETTE["font"]), showlegend=False,
                      xaxis=dict(showgrid=False, title=None), yaxis=dict(showgrid=False, title="Runs"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def fmt_t(datetime_iso: str) -> str:
    try:
        x = datetime.fromisoformat(datetime_iso.replace("Z", "+00:00"))
        return x.astimezone().strftime("%a %b %d · %I:%M %p")
    except Exception:
        return datetime_iso


@st.cache_data(ttl=300, show_spinner=False)
def fetch_games(day_iso: str):
    return schedule(date.fromisoformat(day_iso))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_season(season: str):
    return season_stats(season)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def fetch_injuries(team_id: str):
    return injuries(team_id)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_actual(game_id: str):
    return actual_boxscore(game_id)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def fetch_roster(team_id: str):
    return current_roster(team_id)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_prior_player_stats(season: str):
    return previous_season_players(season)


def nba_season(d: date) -> str:
    start = d.year if d.month >= 9 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def game_alias(abbr: str) -> str:
    return {"NY": "NYK", "GS": "GSW", "SA": "SAS", "NO": "NOP", "UTAH": "UTA",
            "WSH": "WAS", "PHO": "PHX", "BRK": "BKN"}.get(abbr, abbr)


def injury_dict(team_info: dict) -> tuple[dict[str, str], list[dict], str | None]:
    """Hold last-known good report on a temporary outage; mark it STALE."""
    key = str(team_info["id"])
    cache = st.session_state.setdefault("last_good_injury_feed", {})
    try:
        items, _provenance = fetch_injuries(key)
        cache[key] = {"items": items, "updated": datetime.now(timezone.utc).isoformat()}
        return {normalize_name(x["player"]): injury_status(x["status"]) for x in items}, items, None
    except Exception as exc:
        old = cache.get(key)
        if old is not None:
            items = old["items"]
            note = (f'{team_info["abbr"]}: LIVE injury fetch unavailable ({type(exc).__name__}); '
                    f'using last-good report from {old["updated"][:16]} UTC (STALE)')
            return {normalize_name(x["player"]): injury_status(x["status"]) for x in items}, items, note
        return {}, [], f'{team_info["abbr"]}: injury source unavailable ({type(exc).__name__}); injury states unverified'


def roster_with_fallback(team_info: dict) -> tuple[list[dict], str | None]:
    """Use cached roster on a temporary outage and flag it as potentially stale."""
    key = str(team_info["id"])
    cache = st.session_state.setdefault("last_good_team_roster", {})
    try:
        athletes = fetch_roster(key)
        cache[key] = {"items": athletes, "updated": datetime.now(timezone.utc).isoformat()}
        return athletes, None
    except Exception as exc:
        old = cache.get(key)
        if old is not None:
            return old["items"], (f'{team_info["abbr"]}: live roster fetch unavailable '
                                  f'({type(exc).__name__}); using {old["updated"][:16]} UTC snapshot (STALE)')
        raise


def fantasy_config():
    with st.expander("Fantasy scoring · your league", expanded=False):
        st.caption("Defaults match your 1 / 2 / 2 / 3 / 3 / −1 scoring.")
        pts = st.number_input("Points", -5.0, 10.0, 1.0, .5, key="w_pts")
        reb = st.number_input("Rebounds", -5.0, 10.0, 2.0, .5, key="w_reb")
        ast = st.number_input("Assists", -5.0, 10.0, 2.0, .5, key="w_ast")
        stl = st.number_input("Steals", -5.0, 10.0, 3.0, .5, key="w_stl")
        blk = st.number_input("Blocks", -5.0, 10.0, 3.0, .5, key="w_blk")
        tov = st.number_input("Turnovers", -5.0, 10.0, -1.0, .5, key="w_tov")
        threes = st.number_input("Made threes bonus", -5.0, 10.0, 0.0, .5, key="w_3pm")
        return FantasyScoring(pts, reb, ast, stl, blk, tov, threes)


# --- Sidebar and data sourcing ---
st.sidebar.markdown('<div class="brand">COURTSIDE<em>IQ</em></div>', unsafe_allow_html=True)
st.sidebar.caption("SIMULATION + FANTASY INTELLIGENCE")
if CLOUD_MODE:
    st.sidebar.caption("📱 Mobile + cloud edition · same simulation engine")
page = st.sidebar.radio("Workspace", NAV, label_visibility="collapsed", key="page")
# Navigation is also visible on every screen. Button callbacks run before Streamlit
# recreates the sidebar radio, so changing its keyed state is safe.
def navigate_to(destination: str):
    st.session_state["page"] = destination

nav_cols = st.columns([1.35, 1.0, 1.0, 1.0, 1.0])
for col, title, destination in zip(nav_cols,
        ["⌂ Home / Simulator", "Box Score", "Fantasy Lab", "Shot Lab", "More ▾"],
        ["Command Center", "Box Score", "Fantasy Lab", "Shot Lab", "Player Lab"]):
    with col:
        st.button(title, key="topnav_" + destination, use_container_width=True,
                  type="primary" if page == destination else "secondary",
                  on_click=navigate_to, args=(destination,))
if page != "Command Center":
    st.caption("Navigation: use Home / Simulator to return to the matchup. Your last simulation remains available.")

st.sidebar.divider()
mode = st.selectbox("📡 Data source — verify this before simulating", ["Live NBA data", "Import CSV files", "Synthetic demo"],
                             index=2 if os.getenv("COURTSIDE_IQ_DEMO") == "1" else 0,
                             help="Synthetic demo is never passed off as current NBA data.")
scoring = fantasy_config()
with st.sidebar.expander("Live Sync · automatic updates", expanded=True):
    S["auto_sync"] = st.toggle("Automatically check for changes", value=bool(S.get("auto_sync", True)),
                               help="Polls as long as this dashboard is open in your browser.")
    S["sync_minutes"] = st.selectbox("Check every", [2, 5, 10, 15, 30],
                                     index=[2, 5, 10, 15, 30].index(int(S.get("sync_minutes", 5))),
                                     format_func=lambda n: f"{n} minutes")
    S["auto_rerun"] = st.checkbox("Automatically rerun projections when inputs change",
                                 value=bool(S.get("auto_rerun", True)))
    S["auto_first_run"] = st.checkbox("Auto-generate the initial matchup forecast",
                                     value=bool(S.get("auto_first_run", True)))
    S["advanced_auto"] = st.checkbox("Update public lineup/defender aggregates daily", value=bool(S.get("advanced_auto", True)))
    S["audit_auto"] = st.checkbox("Automatically save genuine pregame forecasts", value=bool(S.get("audit_auto", True)))
    S["grade_auto"] = st.checkbox("Automatically grade completed games", value=bool(S.get("grade_auto", True)))
    S["watch_scope"] = st.selectbox("Watch for injuries and roster moves", ["Selected matchup", "Today's NBA slate"],
                                    index=["Selected matchup", "Today's NBA slate"].index(S.get("watch_scope", "Selected matchup")),
                                    help="Slate mode checks more teams and may hit API limits.")
    st.caption("Runs while this browser tab is open. Sources can lag or omit official injury updates.")
if mode == "Live NBA data" and S["auto_sync"]:
    st_autorefresh(interval=int(S["sync_minutes"]) * 60_000, key="courtside_auto_poll")
# Core controls belong in the main workspace; no sidebar hunting required.
st.markdown("### 🏀 Simulation Control Room")
st.caption("Detailed possessions now couple foul trouble, team bonus, live substitutions, lineup defense, shot outcomes and actual on-court minutes. Historical baselines inform roles; these remain experimental predictions, not NBA tracking replays.")
engine_choice = st.radio("Simulation engine", ["Detailed possessions (recommended)", "Fast statistical baseline"],
    index=0 if S.get("possession_mode", True) else 1, horizontal=True,
    help="Detailed mode links possessions, personal fouls, bonus free throws, rotations, individual minutes and lineup-conditioned defense. It does not use optical player tracking or exact play-by-play.")
S["possession_mode"] = engine_choice.startswith("Detailed")
engine_a, engine_b, engine_c = st.columns([2, 1, 1])
with engine_a:
    num_sims = st.select_slider("Simulations per matchup", options=[1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000], value=10_000 if CLOUD_MODE else 150_000)
    if CLOUD_MODE and num_sims >= 100_000:
        st.caption("⚠️ Large simulations may exceed free hosting memory/time limits. Start with 10,000 and increase if stable.")
with engine_b:
    S["shot_mode"] = st.checkbox("Shot-zone model", value=bool(S.get("shot_mode", True)),
        help="Uses actual NBA.com shot profiles only if loaded. Otherwise falls back to league zone priors.")
with engine_c:
    S["lineup_mode"] = st.checkbox("New lineup combinations", value=bool(S.get("lineup_mode", True)))
S["auto_shot_fetch"] = st.checkbox("Automatically fetch real player shot charts before detailed simulations",
    value=bool(S.get("auto_shot_fetch", True)),
    help="NBA.com may be slow, rate-limited or unavailable. Missing profiles are explicitly flagged; they are never described as verified.")
with st.expander("Advanced simulation controls", expanded=False):
    seed = st.number_input("Reproducible random seed", min_value=0, max_value=2_000_000, value=int(S["seed"]), step=1)
    home_adv = st.slider("Home-court advantage (points)", 0.0, 5.0, 2.2, .1)
    pace_delta = st.slider("Pace adjustment", -8.0, 8.0, 0.0, .5)
    uncertainty = st.slider("Statistical variance multiplier", .6, 1.8, 1.0, .05)
    fantasy_threshold = st.number_input("Fantasy target (points)", 1.0, 120.0, 40.0, 1.0)
    S["rotation_interval"] = st.slider("Resample rotations every N possessions", 4, 20, int(S.get("rotation_interval", 8)))
    S["defense_strength"] = st.slider("Opponent-zone defense influence", 0., 1.5, float(S.get("defense_strength", 1.0)), .1)
    S["defensive_events_strength"] = st.slider("Lineup defensive event influence", 0., 1.5, float(S.get("defensive_events_strength", 1.0)), .1)
    st.caption("Defensive inputs are proxies unless verified tracking data is supplied.")

with st.expander("Pregame intelligence · optional", expanded=False):
    S["smart_mode"] = st.toggle("Additional experimental historical blend", value=bool(S.get("smart_mode", False)),
        help="Current-season data already blends last season automatically. This optional second blend is experimental and may double-count historical information; leave off for standard projections.")
    S["smart_role"] = st.toggle("Injury-aware role opportunity", value=bool(S.get("smart_role", True)),
        help="Conservative missing-teammate usage proxy. Manual usage edits take priority.")
    st.caption("Heuristic input estimates, not a validated trained ensemble. Compare against original below.")
# Shot and defense controls are in the main Control Room above.

with st.sidebar.expander("Display preferences", expanded=False):
    S["theme"] = st.selectbox("Theme", ["Midnight", "Daylight"], index=["Midnight", "Daylight"].index(S["theme"]))
    S["accent"] = st.selectbox("Accent", list(COLORS), index=list(COLORS).index(S["accent"]))
    if S["accent"] == "Custom":
        S["custom_hex"] = st.color_picker("Custom accent hex", value=S.get("custom_hex", "#9575FF"))
    S["font_size"] = st.selectbox("Typography size", ["Small", "Standard", "Large"],
                                  index=["Small", "Standard", "Large"].index(S.get("font_size", "Standard")))
    S["density"] = st.selectbox("Density", ["Comfortable", "Compact"], index=["Comfortable", "Compact"].index(S["density"]))
    S["chart"] = st.selectbox("Charts", ["Bars", "Curve"], index=["Bars", "Curve"].index(S["chart"]))
    S["corners"] = st.selectbox("Card corners", ["Rounded", "Sharp"], index=["Rounded", "Sharp"].index(S["corners"]))
    S["show_bands"] = st.checkbox("Show uncertainty bands", value=bool(S["show_bands"]))
    S["show_insights"] = st.checkbox("Show model context", value=bool(S["show_insights"]))
    S["show_distribution"] = st.checkbox("Show distributions", value=bool(S["show_distribution"]))
    if st.button("Save preferences" if CLOUD_MODE else "Save all preferences on this Mac", use_container_width=True):
        S["seed"] = int(seed)
        settings_write(S)
        st.success("Preferences saved for this browser session" if CLOUD_MODE else "Appearance, Auto Sync and simulation preferences saved locally")
with st.sidebar.expander("Trained prediction engine", expanded=False):
    S["ml_active"] = st.toggle("Use validated historical ML", value=bool(S.get("ml_active", True)),
        help="Uses a fitted model only for stats whose holdout accuracy beat rolling historical averages. Needs training first.")
    S["ml_auto_update"] = st.toggle("Check for new model-training games daily", value=bool(S.get("ml_auto_update", True)),
        help="When the dashboard is open, fetch new completed NBA games once daily and retrain after enough new logs.")
    if (TRAINING_FOLDER / "manifest.json").exists():
        try:
            meta_preview = json.loads((TRAINING_FOLDER / "manifest.json").read_text())
            st.caption(f"Trained: {meta_preview['created_utc'][:10]} · latest games: {meta_preview['latest_game']}")
        except (OSError, ValueError, KeyError):
            st.caption("Model metadata unreadable. Open ML Studio.")
    else:
        st.warning("Historical ML not trained; the simulator will use verified statistical baselines until training succeeds.")
        if st.button("Retry historical download + training", key="bootstrap_ml_sidebar", use_container_width=True,
                     help="Downloads official historical NBA player game logs and runs chronological holdout validation. Can take several minutes; NBA.com may block requests."):
            try:
                with st.spinner("Downloading historical NBA games and validating model; please wait…"):
                    report = bootstrap_model(TRAINING_FOLDER, DEFAULT_SEASONS)
                st.success(f"Historical model trained: {report.get('all_games', '?')} player games. Review holdout scores in ML Studio.")
                st.rerun()
            except Exception as exc:
                st.session_state["training_last_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                st.error(f"Historical training failed: {st.session_state['training_last_error']}. NBA.com may be unavailable; the simulation will still run without ML.")
    if st.session_state.get("training_last_error"):
        st.caption("Last training error: " + st.session_state["training_last_error"])
st.sidebar.caption("Data depends on third-party services. No betting guarantees.")

# File-based daily throttling prevents repeated downloads on browser reload.
if (mode == "Live NBA data" and S.get("ml_active") and S.get("ml_auto_update")
        and (TRAINING_FOLDER / "manifest.json").exists()):
    stamp_file = TRAINING_FOLDER / "last_daily_check.json"
    today_utc = datetime.now(timezone.utc).date().isoformat()
    try:
        checked = json.loads(stamp_file.read_text()).get("date") if stamp_file.exists() else ""
        if checked != today_utc:
            stamp_file.write_text(json.dumps({"date":today_utc, "status":"checking"}), encoding="utf-8")
            try:
                stamp_report = update_models(TRAINING_FOLDER, (f"{date.today().year if date.today().month >= 9 else date.today().year-1}-{str((date.today().year if date.today().month >= 9 else date.today().year-1)+1)[-2:]}"))
                st.session_state["model_sync_message"] = stamp_report.get("message", "Model update complete")
                if stamp_report.get("status") == "retrained":
                    st.session_state.pop("result", None)
                    st.session_state.pop("slate_result", None)
                    st.session_state["needs_autorun"] = True
            except Exception as exc:
                st.session_state["model_sync_message"] = (f"Daily model update unavailable ({type(exc).__name__}); "
                    "the last saved model is unchanged.")
    except OSError:
        pass

now = datetime.now(timezone.utc)
today = date.today()
schedule_date = st.date_input("📅 NBA game date", value=today)
season_str = nba_season(schedule_date)

teams: pd.DataFrame | None = None
players: pd.DataFrame | None = None
feed_issue = None
source_description = ""
if mode == "Live NBA data":
    with st.spinner(f"Loading NBA.com season stats ({season_str})…"):
        try:
            td, pd0 = fetch_season(season_str)
            teams, players = normalize_teams(td), normalize_players(pd0)
            source_description = f"NBA.com Stats · {season_str} · checked {now.strftime('%Y-%m-%d %H:%M UTC')} (cached up to 1h)"
        except Exception as exc:
            feed_issue = f"NBA season statistics could not be loaded: {type(exc).__name__}: {str(exc)[:180]}"
elif mode == "Import CSV files":
    st.caption("Upload both NBA-style CSV tables, with explicit source provenance.")
    up_teams = st.file_uploader("Team stats CSV", type=["csv"], key="t_csv")
    up_players = st.file_uploader("Player stats CSV", type=["csv"], key="p_csv")
    if up_teams and up_players:
        try:
            teams = normalize_teams(pd.read_csv(up_teams))
            players = normalize_players(pd.read_csv(up_players))
            source_description = f"Imported CSVs · provided by user · {now.strftime('%H:%M UTC')}"
        except Exception as exc:
            feed_issue = f"Invalid uploaded CSVs: {exc}"
    else:
        feed_issue = "Upload the team and player CSV files to run an imported-data projection."
else:
    teams, players = map(lambda x: x.copy(), demo_dataset())
    teams, players = normalize_teams(teams), normalize_players(players)
    source_description = "SYNTHETIC DEMO — fictional player names and illustrative numbers, NOT predictions"

# Archive collection runs in a separate process so data backfills never block
# the UI or the simulation. The process uses an atomic lock to prevent overlap.
if not CLOUD_MODE and mode == "Live NBA data" and S.get("archive_auto", True) and not st.session_state.get("archive_started"):
    st.session_state["archive_started"] = True
    try:
        worker = Path(__file__).with_name("archive_worker.py")
        subprocess.Popen([sys.executable, str(worker), "--recent-days", "3", "--max-days",
                          "7" if S.get("archive_backfill", True) else "0"],
                         cwd=str(Path(__file__).parent),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception as exc:
        st.session_state["archive_start_error"] = str(exc)

all_games = []
games_issue = None
try:
    all_games = fetch_games(str(schedule_date))
except Exception as exc:
    games_issue = f"Live schedule unavailable: {type(exc).__name__}. Pick teams manually."
if st.button("↻ Refresh live feeds", use_container_width=True):
    fetch_games.clear()
    fetch_injuries.clear()
    fetch_roster.clear()
    fetch_prior_player_stats.clear()
    fetch_season.clear()
    fetch_actual.clear()
    st.session_state.pop("result", None)
    st.rerun()

if feed_issue:
    st.error(feed_issue)
    if mode == "Live NBA data":
        st.info("NBA.com may throttle requests. You can import verified current-season CSVs or explicitly choose Synthetic demo to explore the UI. We do not silently substitute demo numbers.")
if games_issue:
    st.warning(games_issue)
if mode == "Synthetic demo":
    st.warning("**DEMO MODE:** Teams and player stats here are invented examples. Do not use these predictions for fantasy or betting decisions.")
if teams is not None and players is not None:
    st.caption(f"**Data provenance:** {source_description} | {len(teams)} teams · {len(players)} player records")

selected_game: dict | None = None
home: str | None = None
away: str | None = None
if teams is not None and players is not None:
    available = set(teams["TEAM_ABBREVIATION"]) & set(players["TEAM_ABBREVIATION"])
    fixtures = [g for g in all_games if game_alias(g["home"]["abbr"]) in available
                and game_alias(g["away"]["abbr"]) in available]
    g_opts = [f'{game_alias(g["away"]["abbr"])} @ {game_alias(g["home"]["abbr"])}  ·  {fmt_t(g["tipoff"])}  ·  {g["state"]}' for g in fixtures]
    choices = g_opts + ["Custom matchup (manual team selection)"]
    selection = st.selectbox("🏀 Select NBA matchup", choices, index=0)
    if selection != choices[-1]:
        selected_game = fixtures[g_opts.index(selection)]
        home = game_alias(selected_game["home"]["abbr"])
        away = game_alias(selected_game["away"]["abbr"])
    else:
        available = sorted(available)
        if len(available) > 1:
            away = st.selectbox("Away team", available, index=0)
            home = st.selectbox("Home team", [x for x in available if x != away], index=0)
        elif available:
            feed_issue = "Only one eligible team is present in the dataset."

# --- Injury data and scenario modifications ---
active_roster = pd.DataFrame()
automatic_injuries: dict[str, str] = {}
raw_injuries: list[dict] = []
injury_errors: list[str] = []
roster_note = ""
verified_rosters: dict[str, list[dict]] = {}
injury_feeds: dict[str, list[dict]] = {}
if home and away and players is not None:
    # Live roster is different from the player-stat leaderboard: injured athletes
    # may have zero current games, and traded players should follow their new team.
    if mode == "Live NBA data" and selected_game is not None:
        try:
            for role in ("home", "away"):
                team_key = game_alias(selected_game[role]["abbr"])
                roster, roster_warning = roster_with_fallback(selected_game[role])
                verified_rosters[team_key] = roster
                if roster_warning:
                    injury_errors.append(roster_warning)
            prior_year = int(season_str[:4]) - 1
            prior_season = f"{prior_year}-{str(prior_year+1)[-2:]}"
            try:
                past_stats = fetch_prior_player_stats(prior_season)
                st.session_state["smart_prior"] = past_stats
                st.session_state["smart_prior_season"] = prior_season
            except Exception:
                past_stats = None
                st.session_state["smart_prior"] = None
                st.session_state["smart_prior_season"] = None
            roster_players, unmatched = reconcile_matchup_rosters(players, verified_rosters, past_stats)
            # Reconcile only the selected matchup; keep league-wide stats for the slate preview.
            players_for_selected = players[~players["TEAM_ABBREVIATION"].isin([home, away])].copy()
            players_for_selected = pd.concat([players_for_selected, roster_players], ignore_index=True)
            roster_note = (f"Verified current roster against ESPN for {away} / {home}. "
                           f"{len(unmatched)} athletes lacking historical NBA rate data.")
            if unmatched:
                injury_errors.append("No statistical baseline for: " + ", ".join(unmatched[:8]))
            for team_roster in verified_rosters.values():
                for entry in team_roster:
                    if entry["injury"]:
                        automatic_injuries[entry["player"]] = injury_status(entry["injury"])
        except Exception as exc:
            players_for_selected = players
            injury_errors.append(f"Could not verify current roster: {type(exc).__name__}; using NBA stats membership")
    else:
        players_for_selected = players
    if mode == 'Live NBA data':
        hist = players_for_selected
        if 'HISTORICAL_PRIOR_USED' in hist:
            matched = int(hist['HISTORICAL_PRIOR_USED'].fillna(False).astype(bool).sum())
            st.caption(f"Historical NBA baseline: {matched}/{len(hist)} player records matched to last season; current-season rates are sample-size weighted. Previous-team membership is not reused.")
            if matched == 0:
                st.warning('No prior-season player evidence was matched. Early-season predictions are unreliable; check the diagnostic report.')
        warnings = hist.get('HISTORICAL_WARNING')
        if warnings is not None and warnings.astype(str).str.len().gt(0).any():
            st.warning('Historical data could not be fully loaded: ' + str(warnings.iloc[0]) + '. Current-season-only predictions are less reliable.')
    # Historical archive is an independent, timestamp-bounded role/pace signal.
    # Existing NBA.com multi-season blending stays primary. Only real NBA mode.
    if mode == "Live NBA data":
        try:
            teams, players_for_selected, archive_evidence = apply_archive_priors(
                teams, players_for_selected, str(schedule_date))
            st.session_state["archive_evidence"] = archive_evidence
        except Exception as exc:
            st.session_state["archive_evidence"] = {"status": "error", "message": str(exc)[:160]}
    active_roster = players_for_selected[players_for_selected["TEAM_ABBREVIATION"].isin([home, away])].copy()
    active_roster = active_roster.sort_values("MIN", ascending=False).groupby("TEAM_ABBREVIATION", group_keys=False).head(13)
    if selected_game is not None and mode != "Synthetic demo":
        for role in ["away", "home"]:
            mapping, items, issue = injury_dict(selected_game[role])
            if issue:
                injury_errors.append(issue)
            raw_injuries.extend(items)
            if not issue:
                injury_feeds[game_alias(selected_game[role]["abbr"])] = items
            for p in active_roster["PLAYER_NAME"]:
                n = normalize_name(p)
                if n in mapping:
                    automatic_injuries[p] = mapping[n]
    elif mode != "Synthetic demo":
        injury_errors.append("Manual matchup selected: automatic injury matching requires a scheduled game.")

st.sidebar.caption("**Injury verification:** " + ("Source unavailable / manual review required" if injury_errors else
                   "Feed queried; absence from injury list ≠ confirmed active" if selected_game else "Not attached to game"))

# Poll a second group of teams when the user opts into slate-wide watching.
# Do not use second-team feed failures to imply roster departures or cleared injuries.
if mode == "Live NBA data" and S.get("watch_scope") == "Today's NBA slate" and all_games:
    slate_teams = {}
    for game in all_games:
        for role in ("away", "home"):
            t = game[role]
            abbr = game_alias(t["abbr"])
            if abbr not in slate_teams:
                slate_teams[abbr] = t
    for abbr, t in slate_teams.items():
        if abbr not in verified_rosters:
            try:
                roster, roster_warning = roster_with_fallback(t)
                verified_rosters[abbr] = roster
                if roster_warning:
                    injury_errors.append(roster_warning)
            except Exception as exc:
                injury_errors.append(f"{abbr} roster watch unavailable ({type(exc).__name__})")
        if abbr not in injury_feeds:
            try:
                _mapping, injury_entries, injury_warning = injury_dict(t)
                if injury_warning:
                    injury_errors.append(injury_warning)
                else:
                    injury_feeds[abbr] = injury_entries
            except Exception as exc:
                injury_errors.append(f"{abbr} injury watch unavailable ({type(exc).__name__})")

# Persist changes between polls. Failed sources are omitted, not classified as mass cuts.
latest_changes: list[dict] = []
if mode == "Live NBA data" and selected_game:
    baselines = {}
    if not active_roster.empty:
        for abbr in (home, away):
            rdf = active_roster[active_roster["TEAM_ABBREVIATION"] == abbr]
            baselines[abbr] = {
                str(r["PLAYER_NAME"]): {"MIN": r["MIN"], "PTS": r["PTS"],
                                        "AST": r["AST"], "REB": r["REB"]}
                for _, r in rdf.iterrows()}
    try:
        stale_team_names = {abbr for abbr in verified_rosters
                            if any(f"{abbr}: live roster fetch unavailable" in err for err in injury_errors)}
        fresh_rosters = {t: r for t, r in verified_rosters.items() if t not in stale_team_names}
        snapshot = make_snapshot(fresh_rosters, injury_feeds, selected_game, baselines)
        latest_changes = record_feed(snapshot)
    except OSError as exc:
        injury_errors.append(f"Change log could not be saved ({type(exc).__name__})")
    if latest_changes:
        st.session_state.pop("slate_result", None)
        st.session_state.pop("slate_date", None)
        st.session_state["latest_feed_changes"] = latest_changes
        st.session_state["feed_change_notice"] = True

# Build and incrementally update the editor. Auto updates affect only *unlocked*
# availability statuses. Users can explicitly lock a player to preserve overrides.
if not active_roster.empty:
    scenario_id = f"{mode}|{schedule_date}|{away}|{home}"
    prior_id = st.session_state.get("scenario_id")
    # Signature includes current roster AND auto injury statuses AND player baselines.
    signature_rows = sorted((str(r["TEAM_ABBREVIATION"]), str(r["PLAYER_NAME"]),
                             round(float(r["MIN"]), 2), round(float(r["PTS"]), 2),
                             automatic_injuries.get(str(r["PLAYER_NAME"]), "Healthy"))
                            for _, r in active_roster.iterrows())
    new_signature = json.dumps(signature_rows)
    if prior_id != scenario_id:
        st.session_state["scenario_data"] = reconcile_editor(None, active_roster, automatic_injuries)
        st.session_state["scenario_id"] = scenario_id
        st.session_state["scenario_auto_signature"] = new_signature
        st.session_state["scenario_version"] = st.session_state.get("scenario_version", 0) + 1
        st.session_state.pop("result", None)
    elif st.session_state.get("scenario_auto_signature") != new_signature:
        previous_table = st.session_state.get("scenario_data")
        revised = reconcile_editor(previous_table, active_roster, automatic_injuries)
        st.session_state["scenario_data"] = revised
        st.session_state["scenario_auto_signature"] = new_signature
        st.session_state["scenario_version"] = st.session_state.get("scenario_version", 0) + 1
        if st.session_state.get("result") is not None:
            st.session_state["needs_autorun"] = True
        st.session_state.pop("result", None)

# Shot charts are intentionally not silently fetched during each browser refresh.
# Persisted profiles are explicitly user-loaded and tagged with their source.
SHOT_DIR = APP_DIR / "shot_intelligence"
# NBA.com aggregate lineup and defended-shot feeds. These are proxies, not tracking.
if st.session_state.get("advanced_loaded_season") != season_str:
    lp, dp, meta = read_advanced_cache(season_str)
    st.session_state.lineup_priors = lp
    st.session_state.defender_profiles = dp
    st.session_state.advanced_feed_meta = meta
    st.session_state.advanced_loaded_season = season_str
if (S.get("advanced_auto", True) and mode == "Live NBA data" and teams is not None
        and players is not None and advanced_refresh_due(season_str, hours=24)):
    # Avoid blocking the initial dashboard load: the user can fetch immediately
    # in Shot Lab. Automatic attempts happen only on refresh after data loads.
    last_attempt = st.session_state.get("advanced_last_attempt")
    now_ts = time.time()
    # On an updated feed parser, retry once immediately instead of reusing
    # the prior version's failed 1-hour cache/session throttle.
    feed_version = __import__('advanced_feeds').FEED_SCHEMA_VERSION
    last_version = st.session_state.get('advanced_attempt_schema_version')
    if (last_version != feed_version or not isinstance(last_attempt, tuple)
            or last_attempt[0] != season_str or now_ts - last_attempt[1] >= 3600):
        st.session_state.advanced_attempt_schema_version = feed_version
        st.session_state.advanced_last_attempt = (season_str, now_ts)
        try:
            st.session_state.advanced_feed_meta = fetch_public_feeds(season_str, players, teams)
            st.session_state.lineup_priors, st.session_state.defender_profiles, _ = read_advanced_cache(season_str)
        except Exception as exc:
            st.session_state.advanced_feed_meta = {"errors": [f"Feed refresh failed: {type(exc).__name__}: {exc}"]}
if "lineup_priors" not in st.session_state:
    st.session_state.lineup_priors = pd.DataFrame(columns=list(__import__('lineup_engine').LINEUP_COLUMNS))
if "defender_profiles" not in st.session_state:
    st.session_state.defender_profiles = pd.DataFrame(columns=list(__import__('defender_intelligence').COLS))
if "shot_profiles" not in st.session_state:
    st.session_state.shot_profiles = pd.DataFrame(columns=["TEAM_ABBREVIATION", "PLAYER_NAME", "ZONE", "FGA", "FGM"])
if "defense_profiles" not in st.session_state:
    st.session_state.defense_profiles = pd.DataFrame(columns=["TEAM_ABBREVIATION", "ZONE", "FGA_ALLOWED", "FGM_ALLOWED"])
if "shot_source" not in st.session_state:
    st.session_state.shot_source = "None — league priors"
if "shot_cache_season" not in st.session_state or st.session_state.shot_cache_season != season_str:
    st.session_state.shot_cache_season = season_str
    cache_file = SHOT_DIR / f"shots_{season_str}.csv"
    if cache_file.exists():
        try:
            st.session_state.shot_profiles = validate_profiles(pd.read_csv(cache_file))
            st.session_state.shot_source = f"Cached NBA.com shot charts · {season_str}"
        except Exception:
            st.session_state.shot_profiles = pd.DataFrame(columns=["TEAM_ABBREVIATION", "PLAYER_NAME", "ZONE", "FGA", "FGM"])
            st.session_state.shot_source = "Cache unreadable — league priors"
    else:
        st.session_state.shot_profiles = pd.DataFrame(columns=["TEAM_ABBREVIATION", "PLAYER_NAME", "ZONE", "FGA", "FGM"])
        st.session_state.shot_source = "None — league priors"

# --- Running the simulation: keep explicit and reproducible ---
result = st.session_state.get("result")
matching = bool(result is not None and result.get("home") == home and result.get("away") == away)


def read_scenario() -> tuple[dict, dict, dict]:
    table = st.session_state["scenario_data"]
    injury_states = {r["Player"]: r["Availability"] for _, r in table.iterrows()}
    minute_caps = {r["Player"]: float(r["Minute cap"]) for _, r in table.iterrows()}
    usages = {r["Player"]: float(r["Usage x"]) for _, r in table.iterrows()}
    return injury_states, minute_caps, usages


def make_config(n: int, seed_value: int) -> Config:
    injury_states, minute_caps, usages = read_scenario()
    return Config(iterations=n, seed=int(seed_value), home_advantage=home_adv,
                  pace_adjustment=pace_delta, volatility=uncertainty,
                  injury_overrides=injury_states, minutes_overrides=minute_caps,
                  usage_overrides=usages,
                  possession_mode=bool(S.get("possession_mode", False)),
                  lineup_priors=st.session_state.get("lineup_priors") if S.get("lineup_mode", True) else None,
                  defender_profiles=st.session_state.get("defender_profiles"),
                  rotation_interval=int(S.get("rotation_interval", 8)),
                  shot_mode=bool(S.get("shot_mode", False)),
                  shot_profiles=st.session_state.shot_profiles if (S.get("shot_mode") or S.get("possession_mode")) else None,
                  defense_profiles=st.session_state.defense_profiles if (S.get("shot_mode") or S.get("possession_mode")) else None,
                  defense_strength=float(S.get("defense_strength", 1.)),
                  defensive_events_strength=float(S.get("defensive_events_strength", 1.)))


def smart_pregame_inputs(source: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, Config, pd.DataFrame, pd.DataFrame]:
    from dataclasses import replace
    prev_year = int(season_str[:4]) - 1
    expected_prior = f"{prev_year}-{str(prev_year+1)[-2:]}"
    prior = (st.session_state.get("smart_prior") if mode == "Live NBA data" and
             st.session_state.get("smart_prior_season") == expected_prior else None)
    adjusted, audit = blended_inputs(source, prior)
    if S.get("smart_role", True):
        cfg = replace(cfg)
        cfg.usage_overrides, role_audit = role_usage_adjustments(
            adjusted, cfg.injury_overrides, cfg.usage_overrides)
    else:
        role_audit = pd.DataFrame()
    return adjusted, cfg, audit, role_audit


def ensure_selected_shot_profiles() -> tuple[int, int, list[str]]:
    """Retrieve current shots plus cached, player-ID-matched historical evidence.

    Up to 12 historical requests per run for the selected matchup only.
    Work resumes on subsequent runs; failures retry after 24 hours.
    Missing attempts are never reported as observed charts.
    """
    if mode != "Live NBA data" or teams is None or players is None or not home or not away:
        return 0, 0, []
    errors = []
    coverage = st.session_state.shot_profiles
    # The in-memory frame may already contain historical weighted evidence.
    # Reload the strictly current-season cache so repeated simulations cannot
    # multiply historical attempts on every run.
    current_file = SHOT_DIR / f'shots_{season_str}.csv'
    if current_file.exists():
        try:
            current = validate_profiles(pd.read_csv(current_file))
        except (ValueError, OSError):
            current = pd.DataFrame(columns=['TEAM_ABBREVIATION','PLAYER_NAME','ZONE','FGA','FGM'])
    else:
        current = (coverage.copy() if st.session_state.get('shot_source') == 'User-imported shot-zone CSV'
                   else pd.DataFrame(columns=['TEAM_ABBREVIATION','PLAYER_NAME','ZONE','FGA','FGM']))
    now = time.time()
    attempts = st.session_state.setdefault('shot_fetch_attempts', {})
    for abbr in (away, home):
        roster_names = set(players_for_selected.loc[
            players_for_selected['TEAM_ABBREVIATION'] == abbr, 'PLAYER_NAME'].astype(str).str.casefold())
        present = set(current.loc[current['TEAM_ABBREVIATION'] == abbr, 'PLAYER_NAME'].astype(str).str.casefold())
        if roster_names and roster_names.issubset(present):
            continue
        key = f'current:{season_str}:{abbr}'
        if now - float(attempts.get(key, 0)) < 24*3600:
            continue
        attempts[key] = now
        try:
            tid = teams.loc[teams['TEAM_ABBREVIATION'] == abbr, 'TEAM_ID'].iloc[0]
            with st.spinner(f'Fetching {abbr} current-season shot locations…'):
                incoming = fetch_team_shots(int(tid), season_str, abbr)
            remaining = current.loc[current['TEAM_ABBREVIATION'] != abbr]
            current = validate_profiles(pd.concat([remaining, incoming], ignore_index=True))
            SHOT_DIR.mkdir(parents=True, exist_ok=True)
            current.to_csv(SHOT_DIR / f'shots_{season_str}.csv', index=False)
        except Exception as exc:
            errors.append(f'{abbr} current-season shots: {type(exc).__name__}: {str(exc)[:90]}')
    # Historical caches use stable PLAYER_ID and current roster identity. This
    # works across trades, and does not depend on a player's current FGA count.
    history_dir = SHOT_DIR / 'history'
    history_dir.mkdir(parents=True, exist_ok=True)
    older_frames = []
    requests_left = 12
    outstanding = 0
    # CRITICAL: players_for_selected also contains the other ~380 NBA players
    # for the slate. Only download historical shots for this matchup.
    selected_roster = players_for_selected.loc[
        players_for_selected['TEAM_ABBREVIATION'].isin((away, home)),
        ['TEAM_ABBREVIATION', 'PLAYER_NAME', 'PLAYER_ID']
    ].drop_duplicates('PLAYER_ID')
    seasons = historical_seasons(season_str, 5)
    # Prioritize players with no current OR cached historical shot evidence.
    # Previously a whole season of downloads was processed before trying a
    # second season for players still missing charts. That can leave players
    # with empty recent-season responses uncovered despite older observations.
    covered_cached = {
        (str(r.TEAM_ABBREVIATION), str(r.PLAYER_NAME).casefold())
        for r in current.itertuples()
        if float(r.FGA) > 0
    }
    for row in selected_roster.itertuples(index=False):
        key = (str(row.TEAM_ABBREVIATION), str(row.PLAYER_NAME).casefold())
        if key in covered_cached:
            continue
        try:
            pid = int(row.PLAYER_ID)
        except (ValueError, TypeError):
            continue
        for season in seasons:
            cached_file = history_dir / f'{season}_{pid}.csv'
            if not cached_file.exists():
                continue
            try:
                cached = pd.read_csv(cached_file)
                if 'FGA' in cached.columns and pd.to_numeric(cached['FGA'], errors='coerce').fillna(0).sum() > 0:
                    covered_cached.add(key)
                    break
            except (ValueError, OSError, pd.errors.EmptyDataError):
                continue
    selected_roster = selected_roster.copy()
    selected_roster['_missing_evidence'] = [
        (str(r.TEAM_ABBREVIATION), str(r.PLAYER_NAME).casefold()) not in covered_cached
        for r in selected_roster.itertuples(index=False)
    ]
    selected_roster = selected_roster.sort_values('_missing_evidence', ascending=False, kind='stable')
    roster_rows = list(selected_roster.itertuples(index=False))
    # Exhaust the newest-to-oldest history for uncovered players first, then
    # spend remaining requests deepening evidence for already-covered players.
    for missing_first in (True, False):
        for idx, prior_season in enumerate(seasons):
            for row in roster_rows:
                if ((str(row.TEAM_ABBREVIATION), str(row.PLAYER_NAME).casefold()) not in covered_cached) != missing_first:
                    continue
                try:
                    pid = int(row.PLAYER_ID)
                except (ValueError, TypeError):
                    continue
                file = history_dir / f'{prior_season}_{pid}.csv'
                frame = None
                if file.exists():
                    try:
                        frame = pd.read_csv(file)
                        if not {'ZONE','FGA','FGM'}.issubset(frame.columns):
                            raise ValueError('invalid cached chart')
                    except (ValueError, OSError):
                        file.unlink(missing_ok=True)
                if frame is None:
                    key = f'history:{prior_season}:{pid}'
                    if requests_left <= 0 or now - float(attempts.get(key, 0)) < 24*3600:
                        outstanding += 1
                        continue
                    attempts[key] = now
                    requests_left -= 1
                    try:
                        frame = fetch_player_shots(pid, prior_season, str(row.TEAM_ABBREVIATION), str(row.PLAYER_NAME))
                        # Empty frames are cached as confirmed zero-attempt seasons.
                        frame.to_csv(file, index=False)
                    except Exception as exc:
                        errors.append(f'{row.PLAYER_NAME} {prior_season}: {type(exc).__name__}: {str(exc)[:75]}')
                        outstanding += 1
                        continue
                if not frame.empty:
                    frame = frame.copy()
                    frame['TEAM_ABBREVIATION'] = str(row.TEAM_ABBREVIATION)
                    frame['PLAYER_NAME'] = str(row.PLAYER_NAME)
                    frame[['FGA','FGM']] = frame[['FGA','FGM']].astype(float) * (.40 * (.55 ** idx))
                    older_frames.append(frame)
    combined = validate_profiles(pd.concat([current] + older_frames, ignore_index=True))
    st.session_state.shot_profiles = combined
    st.session_state.shot_source = f'NBA.com current {season_str} + player-ID historical shot charts (weighted, cached)'
    st.session_state.shot_history_pending = outstanding
    if outstanding:
        errors.append(f'{outstanding} historical player-season downloads pending; up to 12 fetched per simulation')
    covered_names = {(str(r.TEAM_ABBREVIATION), str(r.PLAYER_NAME).casefold()) for r in combined.itertuples()}
    included = selected_roster[['TEAM_ABBREVIATION', 'PLAYER_NAME']].drop_duplicates()
    count = sum((str(r.TEAM_ABBREVIATION), str(r.PLAYER_NAME).casefold()) in covered_names
                for r in included.itertuples())
    return count, len(included), errors


def launch_sim(n: int | None = None) -> bool:
    if teams is None or players is None or not home or not away:
        st.error("Load two team rosters before simulating.")
        return False
    try:
        shot_check = None
        if S.get("possession_mode", False) and mode == "Live NBA data" and S.get("auto_shot_fetch", True):
            shot_check = ensure_selected_shot_profiles()
            covered, total, issues = shot_check
            if issues:
                st.warning("Shot-data fetch issues: " + " · ".join(issues))
            if covered < total:
                st.error(f"Only {covered}/{total} selected player shot charts available. Missing profiles will use league priors. This run is NOT fully data-verified.")
        with st.spinner(f"Simulating {n or num_sims:,} game outcomes, minutes and player box scores…"):
            adjusted_players = players_for_selected
            st.session_state["model_used"] = False
            st.session_state["model_issue"] = ""
            if not S.get("ml_active"):
                st.session_state["model_issue"] = "Historical ML disabled in settings; statistical baselines used."
            elif mode != "Live NBA data":
                st.session_state["model_issue"] = "Historical ML requires Live NBA data mode."
            elif selected_game is not None and selected_game.get("state") != "pre":
                st.session_state["model_issue"] = "Historical ML is intentionally unavailable for a started/finished game. Select an upcoming game to apply a pregame-trained model; past games require a date-cutoff replay model."
            elif not (TRAINING_FOLDER / "manifest.json").exists():
                # One automatic attempt per local day; never block subsequent runs on a failed provider.
                # Cached real NBA game logs are reused. Failed training never creates synthetic observations.
                attempt_path = TRAINING_FOLDER / "last_bootstrap_attempt.json"
                today = datetime.now(timezone.utc).date().isoformat()
                try:
                    previous = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
                except (OSError, ValueError):
                    previous = {}
                if previous.get("date") != today:
                    try:
                        TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)
                        attempt_path.write_text(json.dumps({"date": today, "status": "started"}))
                        with st.spinner("First run: downloading real historical NBA logs and training ML (may take several minutes)…"):
                            bootstrap_model(TRAINING_FOLDER, DEFAULT_SEASONS)
                        attempt_path.write_text(json.dumps({"date": today, "status": "trained"}))
                    except Exception as exc:
                        detail = f"{type(exc).__name__}: {str(exc)[:180]}"
                        st.session_state["training_last_error"] = detail
                        attempt_path.write_text(json.dumps({"date": today, "status": "failed", "error": detail}))
                if not (TRAINING_FOLDER / "manifest.json").exists():
                    st.session_state["model_issue"] = ("Historical ML could not be trained automatically: "
                        + str(st.session_state.get("training_last_error") or previous.get("error") or "NBA history not available")
                        + ". Baseline simulation remains available; retry from the sidebar.")
            # Custom future matchups can also use historical ML. The trainer checks
            # tipoff is strictly after its last archived game, preventing future leakage.
            ml_tipoff = (selected_game.get("tipoff") if selected_game is not None else
                          datetime.now(timezone.utc).isoformat())
            if (S.get("ml_active") and mode == "Live NBA data"
                    and (selected_game is None or selected_game.get("state") == "pre")
                    and (TRAINING_FOLDER / "manifest.json").exists()):
                try:
                    adjusted_players, player_audit, manifest = forecast_ml(
                        players_for_selected, home, away, ml_tipoff, TRAINING_FOLDER)
                    st.session_state["model_audit"] = player_audit
                    st.session_state["model_manifest"] = manifest
                    st.session_state["model_used"] = True
                    st.session_state["model_issue"] = ""
                except Exception as exc:
                    st.session_state["model_issue"] = (f"Trained model was not used ({type(exc).__name__}: "
                        f"{str(exc)[:130]}). Predictions instead use season-rate baselines.")
            run_cfg = make_config(n or num_sims, int(seed))
            if S.get("smart_mode", False):
                adjusted_players, run_cfg, smart_audit, role_audit = smart_pregame_inputs(adjusted_players, run_cfg)
                st.session_state["smart_audit"] = smart_audit
                st.session_state["role_audit"] = role_audit
            start_clock = time.perf_counter()
            st.session_state['last_simulation_inputs'] = adjusted_players.copy()
            run_result = run(teams, adjusted_players, home, away, run_cfg)
            run_result["elapsed_seconds"] = round(time.perf_counter() - start_clock, 3)
            run_result["completed_draws"] = int(len(run_result["home_score"]))
            run_result["engine_verified"] = ("Detailed possession-event" if run_cfg.possession_mode else "Fast statistical baseline")
            run_result["shot_data_check"] = shot_check
            run_result["shot_history_pending"] = st.session_state.get("shot_history_pending", 0)
            run_result["shot_data_source_at_run"] = st.session_state.get("shot_source", "Unspecified")
        st.session_state.result = run_result
        if S.get("audit_auto", True) and mode == "Live NBA data" and selected_game is not None:
            try:
                saved = save_if_due(run_result, selected_game, scoring, source_description)
                if saved:
                    st.session_state["audit_save_message"] = f"Pregame forecast {saved['id']} saved automatically."
            except (ValueError, OSError, KeyError, TypeError) as exc:
                st.session_state["audit_save_message"] = f"Automatic forecast save unavailable: {exc}"
        st.session_state.result_meta = {"source": source_description,
            "shot_mode": "possession-event simulation" if S.get("possession_mode") else "zone-attempt sampling" if S.get("shot_mode") else "season-rate baseline",
            "shot_source": st.session_state.shot_source if (S.get("shot_mode") or S.get("possession_mode")) else "not used",
            "model": ("smart pregame input blend (unvalidated) + " if S.get("smart_mode") else "") +
                     ("trained + rolling (holdout-gated)" if st.session_state.get("model_used") else "season-average statistical baseline"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fixture_id": selected_game["id"] if selected_game else None}
        return True
    except Exception as exc:
        st.error(f"Simulation did not run: {type(exc).__name__}: {exc}")
        return False


# Automatically recompute from fresh rosters/injuries/baselines after a feed change.
# Never automatically claim a midgame output is a live win-probability model.
run_changed = st.session_state.pop("needs_autorun", False)
initial_preview = (result is None and st.session_state.get("scenario_id")
                   and S.get("auto_first_run", True) and mode == "Live NBA data"
                   and selected_game is not None and selected_game.get("state") == "pre")
if (run_changed and S.get("auto_rerun", True) or initial_preview) and mode == "Live NBA data":
    if selected_game is not None and selected_game.get("state") == "pre":
        if launch_sim():
            result = st.session_state.get("result")
            matching = bool(result is not None)

# Background-like audit while the dashboard is open; checks are rate limited to hourly.
if S.get("grade_auto", True) and mode == "Live NBA data":
    last = st.session_state.get("last_auto_grade_at")
    if last is None or (datetime.now(timezone.utc) - last).total_seconds() >= 3600:
        st.session_state["last_auto_grade_at"] = datetime.now(timezone.utc)
        try:
            graded, errors = grade_completed(fetch_games, fetch_actual, max_games=3)
            st.session_state["auto_grades"] = graded
            st.session_state["auto_grade_errors"] = errors
        except Exception as exc:
            st.session_state["auto_grade_errors"] = [str(exc)]

if page == "Command Center":
    hero("See the whole game before tipoff.",
         "Matchup outlook, probability distributions and calibrated-by-testing research tools in one workspace.",
         "01 / MATCHUP INTELLIGENCE")
    if latest_changes:
        st.success(f"Auto Sync detected {len(latest_changes)} change(s). Projections were invalidated and are recalculated automatically when enabled.")
    if home and away:
        st.markdown(f'<span class="pill live">{escape(away)} @ {escape(home)}</span>'
                    f'<span class="pill">{escape(fmt_t(selected_game["tipoff"]) if selected_game else "Custom matchup")}</span>'
                    f'<span class="pill">{escape(mode)}</span>', unsafe_allow_html=True)
        if selected_game and selected_game["state"] != "pre":
            st.warning("This game is underway or finished. The model is a **pregame baseline**, not a score-conditional live win probability model.")
        if roster_note:
            st.caption(roster_note)
        if injury_errors:
            st.warning(" · ".join(injury_errors))
        if st.button(f"▶ Run {num_sims:,} simulations", type="primary", use_container_width=True):
            if launch_sim():
                result = st.session_state.result
                matching = True
    if matching:
        section("SIMULATION VERIFICATION · ACTUAL EXECUTION")
        n_done = int(result.get("completed_draws", len(result["home_score"])))
        actual_n = len(result["away_score"])
        detail = bool(result["config"].possession_mode)
        sim_columns = st.columns(4)
        sim_columns[0].metric("Completed draws", f"{n_done:,}")
        sim_columns[1].metric("Elapsed compute time", f"{result.get('elapsed_seconds', float('nan')):.2f} s")
        sim_columns[2].metric("Engine actually used", "Detailed events" if detail else "Fast baseline")
        sim_columns[3].metric("Score arrays verified", "PASS" if n_done == actual_n == result["config"].iterations else "CHECK")
        st.caption("Vectorized NumPy operations can process many scenarios quickly. Runtime alone cannot prove that each event is physically realistic.")
        if detail:
            audits = result.get("possession_audits") or {}
            if all(team in audits and "POSS" in audits[team] and len(audits[team]["POSS"]) == n_done for team in (home, away)):
                st.success("Detailed event audit present for every simulated game for both teams.")
            else:
                st.error("Detailed event audit is incomplete. Do not treat this run as verified possession-event simulation.")
        else:
            st.warning("This run used the fast baseline, NOT the possession-event simulator. Select Detailed possessions at the top and rerun.")
        with st.expander("Inspect the actual inputs and simulated outcomes", expanded=True):
            st.write({"Data source": st.session_state.get("result_meta", {}).get("source", "Unspecified"),
                      "Engine": result.get("engine_verified", "Unspecified"),
                      "Completed outcomes": n_done,
                      "NBA season": season_str,
                      "Player records loaded": len(players) if players is not None else 0,
                      "Player shot chart source": st.session_state.get("shot_source", "Not available"),
                      "Observed lineup rows": len(st.session_state.get("lineup_priors", [])),
                      "Injury feed warnings": len(injury_errors),
                      "Saved at UTC": st.session_state.get("result_meta", {}).get("timestamp", "Unknown")})
            preview = pd.DataFrame({"Run": np.arange(1, min(10, n_done) + 1),
                                    home + " points": result["home_score"][:10],
                                    away + " points": result["away_score"][:10],
                                    "Pace": np.round(result["pace"][:10], 1)})
            st.dataframe(preview, hide_index=True, use_container_width=True)
            if detail and result.get("possession_audits"):
                for team_abbr in (away, home):
                    audit = result["possession_audits"][team_abbr]
                    st.caption(f"{team_abbr}: actual per-game event arrays (first 10 draws)")
                    st.dataframe(pd.DataFrame({key: values[:10] for key, values in audit.items()}),
                                 hide_index=True, use_container_width=True)
            if mode == "Synthetic demo":
                st.error("SYNTHETIC DEMO INPUTS — not current NBA data.")
            elif mode == "Live NBA data":
                st.warning("Live data mode does not prove all feeds are fresh. Check injury warnings, roster records, shot-chart coverage and the timestamp. Cached season stats may be up to an hour old.")
        st.divider()
        if st.session_state.get("model_used"):
            st.success("Historical ML active: per-stat usage is gated by chronological holdout validation. This does not establish sportsbook-level accuracy.")
        elif st.session_state.get("model_issue"):
            st.warning(st.session_state["model_issue"])
        else:
            st.caption("Statistical baseline simulation; train and enable ML Studio for historical-model adjustments.")
        if S.get("smart_mode"):
            st.info("Smart pregame inputs enabled. Same underlying simulator, with prior-season shrinkage and optional injury role adjustments. These heuristics are NOT yet historically validated.")
            with st.expander("Inspect smart-model evidence"):
                if st.session_state.get("smart_prior") is None:
                    st.warning("Prior-season feed unavailable: historical blending skipped. Current player inputs remain intact.")
                audit = st.session_state.get("smart_audit", pd.DataFrame())
                if not audit.empty:
                    st.dataframe(audit, hide_index=True, use_container_width=True)
                roles = st.session_state.get("role_audit", pd.DataFrame())
                if not roles.empty:
                    st.dataframe(roles, hide_index=True, use_container_width=True)
        with st.expander("Compare original simulator vs smart inputs (same engine)"):
            st.caption("Runs two fresh 3,000-game simulations with the same seed and settings. Original is unchanged; the smart forecast is an input adjustment, not a replacement. Differences are not evidence of improved accuracy.")
            if st.button("Run original vs smart comparison · 3,000 each", key="smart_compare"):
                try:
                    with st.spinner("Running two comparable forecasts…"):
                        baseline_cfg = make_config(3000, int(seed))
                        baseline_result = run(teams, players_for_selected, home, away, baseline_cfg)
                        smart_players, smart_cfg, smart_audit, smart_roles = smart_pregame_inputs(players_for_selected, make_config(3000, int(seed)))
                        smart_result = run(teams, smart_players, home, away, smart_cfg)
                        st.session_state["smart_comparison"] = {
                            "scenario": st.session_state.get("scenario_id"),
                            "original": baseline_result, "smart": smart_result,
                            "audit": smart_audit, "roles": smart_roles}
                except Exception as exc:
                    st.error(f"Comparison unavailable: {type(exc).__name__}: {exc}")
            comparison = st.session_state.get("smart_comparison")
            if comparison and comparison.get("scenario") == st.session_state.get("scenario_id"):
                orig, smart = comparison["original"], comparison["smart"]
                st.write(f"**Home win:** original {orig['home_win']:.1%} · smart {smart['home_win']:.1%}")
                st.write(f"**Game total:** original {np.mean(orig['total']):.1f} · smart {np.mean(smart['total']):.1f}")
                delta = prediction_deltas(orig, smart, scoring)
                st.dataframe(delta.round(2), hide_index=True, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card(f"{away} expected score", f'{result["away_mean"]:.1f}', "All simulation outcomes")
        with c2: metric_card(f"{home} expected score", f'{result["home_mean"]:.1f}', "Includes home-court input")
        with c3: metric_card(f"{home} win chance", f'{result["home_win"]*100:.1f}%', "Not sportsbook-calibrated")
        with c4: metric_card("Expected game total", f'{float(np.mean(result["total"])):.1f}', f'{result["config"].iterations:,} draws · seed {result["config"].seed}')
        if S["show_distribution"]:
            left, right = st.columns(2)
            with left: plot_hist(result["margin"], f"Margin distribution: {home} minus {away}", 0)
            with right: plot_hist(result["total"], "Combined points distribution")
        # Quarter-level statistics are recorded from the same simulated events,
        # including individual players. No quarter scores are estimated by
        # dividing the final score by four. Overtime is shown separately.
        qsummary = result.get('quarter_summary')
        if qsummary:
            section("QUARTER-BY-QUARTER · SIMULATED SCORE & STATS")
            st.caption("Each quarter is recorded from the same possession simulations as the full-game forecast. "
                       "End-of-quarter scores are cumulative averages across all games. "
                       "Overtime is separate; quarter timing is approximate, not official play-by-play.")
            quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
            rows_q = []
            for qi, label in enumerate(quarter_labels):
                rows_q.append({
                    'Period': label,
                    f'{away} quarter PTS': round(qsummary['teams'][away]['quarter_mean_points'][qi], 1),
                    f'{home} quarter PTS': round(qsummary['teams'][home]['quarter_mean_points'][qi], 1),
                    f'{away} end score': round(qsummary['teams'][away]['end_of_quarter_mean_score'][qi], 1),
                    f'{home} end score': round(qsummary['teams'][home]['end_of_quarter_mean_score'][qi], 1),
                })
            st.dataframe(pd.DataFrame(rows_q), hide_index=True, use_container_width=True)
            st.caption(f"Average overtime contribution (across all games): "
                       f"{away} {qsummary['teams'][away]['overtime_mean_points_all_games']:.2f} · "
                       f"{home} {qsummary['teams'][home]['overtime_mean_points_all_games']:.2f} points.")
            selected_q = st.selectbox('Inspect quarter', quarter_labels, key='quarter_stat_select')
            qi = quarter_labels.index(selected_q)
            selected_team = st.radio('Quarter team', (away, home), horizontal=True, key='quarter_team_select')
            tq = qsummary['teams'][selected_team]
            qstats = {'PTS': tq['quarter_mean_points'][qi]}
            qstats.update({stat: vals[qi] for stat, vals in tq['quarter_event_means'].items()
                           if stat in ('POSS', 'FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'DREB',
                                       'PERSONAL_FOULS', 'SHOOTING_FOULS', 'OFFENSIVE_FOULS')})
            st.markdown(f"**{selected_team} · {selected_q} team averages**")
            st.dataframe(pd.DataFrame([qstats]).round(2), hide_index=True, use_container_width=True)
            player_q = pd.DataFrame({'Player': tq['player_names']})
            for stat in ('MIN', 'PTS', 'REB', 'AST', '3PM', 'FGA', 'FGM', '3PA', 'FTA', 'FTM',
                         'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF'):
                if stat in tq['quarter_player_means']:
                    player_q[stat] = tq['quarter_player_means'][stat][qi]
            st.markdown(f"**{selected_team} · {selected_q} player averages**")
            st.dataframe(player_q.round(2), hide_index=True, use_container_width=True)
            st.download_button('↓ Export selected quarter player stats',
                               player_q.to_csv(index=False),
                               file_name=f'quarter_{selected_q}_{selected_team}.csv',
                               mime='text/csv', key='quarter_player_export')
        else:
            st.caption('Quarter-level event stats are available in Detailed possessions mode.')
        section("Top fantasy projections · your scoring")
        ranked = leaderboard(result, scoring, fantasy_threshold)
        st.dataframe(ranked[["Player", "Team", "Status", "Fantasy AVG", "Fantasy P10", "Fantasy P90", f"P(≥{fantasy_threshold:g})", "MIN"]].head(10),
                     hide_index=True, use_container_width=True)
        section("FULL PROJECTED BOX SCORES · BOTH TEAMS")
        st.caption("These are the averages of the same simulation outcomes shown above, not season averages.")
        for team_abbr in (away, home):
            st.markdown(f"**{team_abbr} · expected score {result['away_mean' if team_abbr == away else 'home_mean']:.1f}**")
            table = result["box"].loc[result["box"]["Team"] == team_abbr].copy()
            table["FANTASY"] = [round(float(fantasy_values(result["samples"][r["ID"]], scoring).mean()), 1)
                                 for _, r in table.iterrows()]
            shown = ["Player", "Status", "MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV", "FANTASY"]
            if result["config"].possession_mode:
                shown += ["FGA", "FGM", "3PA", "FTA", "FTM", "OREB", "DREB"]
            st.dataframe(table[shown].round(1),
                         use_container_width=True, hide_index=True)
        st.download_button("↓ Export complete projected box scores", result["box"].to_csv(index=False),
                           file_name=f"projections_{away}_{home}.csv", mime="text/csv", key="command_export_box")
        if result["config"].shot_mode or result["config"].possession_mode:
            st.warning("Experimental shot-zone / possession-event sampling enabled. Shot-chart coverage: " +
                str(len(st.session_state.shot_profiles['PLAYER_NAME'].unique())) +
                " players · " + st.session_state.shot_source +
                ". Without real charts, player shot locations are league-prior estimates. Defender locations and play sequences are NOT simulated.")
        shot_check = result.get("shot_data_check")
        if shot_check is not None:
            c, total, issues = shot_check
            if c < total:
                st.error(f"DATA COVERAGE FAILURE: {c}/{total} players had observed shot charts when this simulation ran. League priors filled the gaps.")
            else:
                st.success(f"Shot-chart coverage at run time: {c}/{total} selected players.")
        # Diagnostic export contains real input provenance, event-array samples and
        # outcome summaries, not a fabricated play-by-play or every raw event.
        diagnostic = {
            "game": f"{away} @ {home}", "season": season_str,
            "data_source": st.session_state.get("result_meta", {}).get("source", "Unspecified"),
            "engine": result.get("engine_verified"),
            "draws": int(result.get("completed_draws", len(result["home_score"]))),
            "runtime_seconds": result.get("elapsed_seconds"),
            "shot_data_source_at_run": result.get("shot_data_source_at_run"),
            "shot_data_check": result.get("shot_data_check"),
            "injury_feed_warnings": list(map(str, injury_errors)),
            "historical_ml_used": bool(st.session_state.get("model_used")),
            "lineup_rows_loaded": len(st.session_state.get("lineup_priors", [])),
            "shot_calibration": result.get("shot_calibration", {}),
            "sample_game_scores": [
                {"home": int(h), "away": int(a)} for h, a in zip(result["home_score"][:100], result["away_score"][:100])],
            "event_audit_first_100": {
                team: {k: np.asarray(v[:100]).tolist() for k, v in result.get("possession_audits", {}).get(team, {}).items()}
                for team in (home, away)},
            "limitations": "Event arrays are heuristic Monte Carlo outputs, not optical tracking or actual play-by-play. Only the first 100 games are included in this compact diagnostic."
        }
        st.download_button("↓ Download simulation diagnostic report (JSON)",
                           json.dumps(diagnostic, indent=2, default=str),
                           file_name=f"courtside_diagnostics_{away}_{home}.json", mime="application/json",
                           key="diagnostics_download")
        with st.expander("Model integrity · data coverage and limitations"):
            coverage = st.session_state.shot_profiles
            included = result["box"]
            prof_names = set()
            if not coverage.empty:
                prof_names = {(str(x["TEAM_ABBREVIATION"]), str(x["PLAYER_NAME"]).casefold())
                              for _, x in coverage.iterrows()}
            shot_covered = sum((str(x["Team"]), str(x["Player"]).casefold()) in prof_names
                               for _, x in included.iterrows())
            st.write({"Data mode": mode,
                      "Selected game": f"{away} @ {home}",
                      "Injury feed issues": len(injury_errors),
                      "Player shot charts available": f"{shot_covered}/{len(included)}",
                      "Historical ML used": bool(st.session_state.get("model_used")),
                      "Observed lineup rows": len(st.session_state.get("lineup_priors", [])),
                      "Observed lineups blended, not enforced": bool(result["config"].lineup_priors is not None),
                      "Possession-event mode": bool(result["config"].possession_mode),
                      "Simulations": result["config"].iterations,
                      "Random seed": result["config"].seed})
            if mode == "Live NBA data" and players is not None:
                gp = pd.to_numeric(players.loc[players["TEAM_ABBREVIATION"].isin([home, away]), "GP"], errors="coerce").dropna()
                if len(gp):
                    med_gp = float(gp.median())
                    st.write({"Current-season player games (median)": round(med_gp, 1),
                              "Estimated current-season statistical weight": f"{100 * float(sample_reliability([med_gp])[0]):.0f}%"})
                    if med_gp < 12:
                        st.warning("Early-season uncertainty: limited current-season games. Player rates are blended with previous-season information when available; roster membership follows current data. New roles, rookies and recent trades can remain difficult to estimate.")
            if injury_errors:
                st.warning("Some injury or roster information could not be verified; projections may be stale.")
            if result["config"].possession_mode and shot_covered < len(included):
                st.warning("Missing player shot charts are replaced with league-prior zones. Those players do not have individually observed zone distributions.")
            st.caption("These are coverage diagnostics, NOT statistical confidence or validated predictive accuracy.")
        if result["config"].possession_mode and result.get("possession_audits"):
            section("POSSESSION EVENT AUDIT · SIMULATED GAME AVERAGES")
            st.caption("Approximate possessions, shot attempts, makes, free throws and rebounds. All events use heuristic probabilities; this is NOT an official play-by-play or tracking-based model.")
            event_rows = []
            for tm in (away, home):
                audit = result["possession_audits"][tm]
                event_rows.append({"Team": tm, **{k: round(float(v.mean()), 1) for k, v in audit.items()},
                                   "PTS": round(float(result["away_mean" if tm == away else "home_mean"]), 1)})
            st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
            if result.get("possession_calibration"):
                with st.expander("How this model adjusts shooting for team offense and defense"):
                    st.json(result["possession_calibration"])
                    st.caption("A small pregame pilot adjusts shot make probabilities to team offense/opponent-defense baselines. This is a heuristic, NOT trained expected-field-goal modeling or market calibration.")
        if S["show_insights"]:
            st.info("Scores and player points reconcile per simulated game. Possession mode samples exactly five eligible players on court per side, but does NOT know actual coach substitutions, real defender assignments, defensive schemes, shot-clock states or player coordinates. No predictive accuracy guarantee.")
    elif home and away:
        st.info("Select the matchup in the sidebar and run a simulation to generate projections. All stats and fantasy numbers will share the same simulation run.")

elif page == "Scenario Lab":
    hero("Control the assumptions.", "Adjust injury statuses, playing-time ceilings and offensive usage. Then rerun the matchup.", "06 / SCENARIO STUDIO")
    if home and away and not active_roster.empty:
        if injury_errors:
            st.warning("Automatic injuries may be incomplete: " + " · ".join(injury_errors))
        if raw_injuries:
            with st.expander(f"View {len(raw_injuries)} entries from ESPN injury feed"):
                st.dataframe(pd.DataFrame(raw_injuries), use_container_width=True, hide_index=True)
        st.caption("**Availability** is simulated probabilistically (Questionable 55%, Doubtful 14%, Probable 92%). **Minute cap** limits playing time; it is not a target. **Usage x** changes share of scoring opportunities. **Manual lock** protects a player’s manual injury status against incoming feed updates. All editable values are assumptions, not verified injury facts.")
        modified = st.data_editor(st.session_state["scenario_data"], key=f"scenario_editor_{st.session_state.get('scenario_version', 0)}", use_container_width=True, hide_index=True,
            disabled=["Player", "Team", "Base min"],
            column_config={"Availability": st.column_config.SelectboxColumn(options=["Healthy", "Probable", "Questionable", "Doubtful", "Out"], required=True),
                           "Minute cap": st.column_config.NumberColumn(min_value=0., max_value=48., step=1., format="%.0f"),
                           "Usage x": st.column_config.NumberColumn(min_value=.25, max_value=2.5, step=.05, format="%.2f"),
                           "Manual lock": st.column_config.CheckboxColumn(help="Keep my availability override despite live updates")})
        st.session_state["scenario_data"] = modified
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Rerun {num_sims:,} scenarios", type="primary", use_container_width=True):
                if launch_sim():
                    st.success("Scenario applied. Open Box Score or Fantasy Lab to inspect results.")
        with c2:
            if st.button("Reset to detected injuries", use_container_width=True):
                st.session_state["scenario_id"] = "reset"
                st.session_state["scenario_version"] = st.session_state.get("scenario_version", 0) + 1
                st.rerun()
        st.caption("An out designation must leave enough available players to fill a legal rotation. Extreme caps are rejected, rather than inventing extra minutes.")
    else:
        st.info("Load a matchup to edit the rotation.")

elif page == "Fantasy Lab":
    hero("Fantasy projections, built into the engine.",
         "Your scoring: +1 PTS · +2 REB · +2 AST · +3 STL · +3 BLK · −1 TO. Fully configurable in the sidebar.",
         "02 / FANTASY INTELLIGENCE")
    section("Fantasy scoring system")
    labels = [("PTS", scoring.PTS), ("REB", scoring.REB), ("AST", scoring.AST),
              ("STL", scoring.STL), ("BLK", scoring.BLK), ("TO", scoring.TOV)]
    cols = st.columns(6)
    for col, (label, w) in zip(cols, labels):
        with col: metric_card(label, f"{w:+g}")
    if matching:
        rank = leaderboard(result, scoring, fantasy_threshold)
        section(f"Projected fantasy leaders — {away} @ {home}")
        st.dataframe(rank[["Player", "Team", "Status", "Fantasy AVG", "Fantasy P10", "Fantasy P50", "Fantasy P90",
                           f"P(≥{fantasy_threshold:g})", "Boom %", "Bust %", "MIN"]], use_container_width=True,
                     hide_index=True, height=450)
        a, b, c = st.columns([2, 1, 1])
        with a:
            player_name = st.selectbox("Analyze a player", rank["Player"].tolist(), key="fantasy_focus")
        chosen_row = rank[rank["Player"] == player_name].iloc[0]
        vals = fantasy_values(result["samples"][chosen_row["ID"]], scoring)
        with b: metric_card("Projected fantasy points", f'{np.mean(vals):.1f}', "Mean of full distribution")
        with c: metric_card(f"Chance of {fantasy_threshold:g}+", f'{np.mean(vals >= fantasy_threshold)*100:.1f}%', "Conditional on scenario")
        if S["show_distribution"]:
            plot_hist(vals, f"{player_name} · Fantasy-point outcomes", fantasy_threshold)
        section("Where the fantasy points come from")
        st.dataframe(explain_fantasy_line(result["box"].set_index("ID").loc[chosen_row["ID"]], scoring),
                     hide_index=True, use_container_width=True)
        st.download_button("↓ Export fantasy rankings CSV", rank.to_csv(index=False).encode(),
                           file_name=f"fantasy_{away}_{home}.csv", mime="text/csv")
    else:
        st.info("Run a matchup simulation from Command Center to see full fantasy projections.")
    section("Full slate preview")
    st.caption("Quick slate preview uses **2,000 iterations per matchup** for speed, not the 150,000 of a selected-game full simulation. For live data, current rosters are checked game by game, and unknown players are flagged. Injury coverage may be incomplete. This is an exploratory ranking, not a validated start/sit recommendation.")
    if teams is not None and players is not None and len(all_games) > 0:
        if st.button("Scan today’s matchups · 2,000 sims/game", use_container_width=True):
            slate_parts, skipped = [], []
            relevant = [g for g in all_games if game_alias(g["home"]["abbr"]) in set(teams["TEAM_ABBREVIATION"])
                        and game_alias(g["away"]["abbr"]) in set(teams["TEAM_ABBREVIATION"])]
            prog = st.progress(0)
            for i, g in enumerate(relevant):
                try:
                    hm, aw = game_alias(g["home"]["abbr"]), game_alias(g["away"]["abbr"])
                    inj_cfg = {}
                    slate_players = players
                    if mode == "Live NBA data":
                        rosters_for_game = {}
                        for role in ("home", "away"):
                            team_abbr = game_alias(g[role]["abbr"])
                            this_roster, roster_warning = roster_with_fallback(g[role])
                            rosters_for_game[team_abbr] = this_roster
                            if roster_warning:
                                skipped.append(roster_warning)
                        try:
                            previous_year = int(season_str[:4]) - 1
                            previous_code = f"{previous_year}-{str(previous_year + 1)[-2:]}"
                            past = fetch_prior_player_stats(previous_code)
                        except Exception:
                            past = None
                        # Never silently use an old-team roster for slate fantasy projections.
                        reconciled, unknown = reconcile_matchup_rosters(players, rosters_for_game, past)
                        if unknown:
                            skipped.append(f'{aw} @ {hm}: {len(unknown)} players have no NBA stat baseline')
                        slate_players = pd.concat([players[~players["TEAM_ABBREVIATION"].isin([hm, aw])], reconciled],
                                                  ignore_index=True)
                    if mode != "Synthetic demo":
                        for role in ("home", "away"):
                            im, _, warning = injury_dict(g[role])
                            if warning:
                                skipped.append(f'{aw} @ {hm}: {warning}')
                            for p in slate_players[slate_players["TEAM_ABBREVIATION"] == game_alias(g[role]["abbr"])] ["PLAYER_NAME"]:
                                if normalize_name(p) in im:
                                    inj_cfg[p] = im[normalize_name(p)]
                    if (S.get("ml_active") and mode == "Live NBA data" and g.get("state") == "pre"
                            and (TRAINING_FOLDER / "manifest.json").exists()):
                        try:
                            slate_players, _, _ = forecast_ml(slate_players, hm, aw, g["tipoff"], TRAINING_FOLDER)
                        except Exception as exc:
                            skipped.append(f'{aw} @ {hm}: trained model skipped ({type(exc).__name__}); historical averages used')
                    cfg = Config(iterations=2000, seed=int(seed) + i, injury_overrides=inj_cfg,
                                 home_advantage=home_adv, pace_adjustment=pace_delta,
                                 volatility=uncertainty)
                    one = run(teams, slate_players, hm, aw, cfg)
                    board = leaderboard(one, scoring, fantasy_threshold)
                    board["Matchup"] = f"{aw} @ {hm}"
                    slate_parts.append(board)
                except Exception as exc:
                    skipped.append(f'{g.get("id")}: {str(exc)[:80]}')
                prog.progress((i + 1) / max(1, len(relevant)))
            if slate_parts:
                st.session_state.slate_result = pd.concat(slate_parts, ignore_index=True).sort_values("Fantasy AVG", ascending=False)
            st.session_state.slate_date = str(schedule_date)
            if skipped:
                st.warning("Skipped games: " + "; ".join(skipped))
        if st.session_state.get("slate_date") == str(schedule_date) and "slate_result" in st.session_state:
            board = st.session_state.slate_result
            st.dataframe(board[["Player", "Team", "Matchup", "Status", "Fantasy AVG", "Fantasy P10", "Fantasy P90",
                                f"P(≥{fantasy_threshold:g})", "MIN"]], use_container_width=True, hide_index=True, height=440)
            st.download_button("↓ Export slate fantasy rankings", board.to_csv(index=False), "fantasy_slate.csv", "text/csv")
    else:
        st.caption("No compatible NBA slate found for the selected date and dataset.")

elif page == "Box Score":
    st.button("← Back to simulator / Command Center", key="box_back_home",
              use_container_width=True, on_click=navigate_to, args=("Command Center",))
    hero("Projected box scores.", "Every player's outcomes sampled within the same team-level simulation.", "03 / BOX SCORE FORECAST")
    if matching:
        for abbr in [away, home]:
            section(f"{abbr} · projected rotation and box score")
            table = result["box"][result["box"]["Team"] == abbr].copy()
            table["FANTASY"] = [float(fantasy_values(result["samples"][r["ID"]], scoring).mean()) for _, r in table.iterrows()]
            cols = ["Player", "Status", "MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV", "FANTASY"]
            if S["show_bands"]:
                cols += ["PTS_P10", "PTS_P90", "MIN_P10", "MIN_P90"]
            st.dataframe(table[cols].round(1), hide_index=True, use_container_width=True, height=395)
            st.caption(f"Projected team score: {result['home_mean'] if abbr == home else result['away_mean']:.1f} · Player points sum to the team score in each simulated outcome. Regulation minutes total 240.")
        st.download_button("↓ Download projected box scores", result["box"].to_csv(index=False),
                           file_name=f"boxscore_{away}_{home}.csv", mime="text/csv")
    else:
        st.info("Run a matchup in Command Center to generate the projected box score.")

elif page == "Shot Lab":
    with st.expander("Automatic NBA.com lineup and defended-shot feeds", expanded=True):
        st.caption("Free official NBA.com tracking-derived aggregates: five-man lineup weights, defended shots <6 ft, and defended three-pointers. Defender counts now blend current and previous seasons with a 12-game-equivalent historical prior. These are observed contested-shot summaries, not optical coordinates or exact matchup assignments. Sources can fail or be throttled; last successful cache is retained.")
        meta = st.session_state.get("advanced_feed_meta", {})
        st.write("Cached lineup rows:", len(st.session_state.lineup_priors), "· defender-zone proxy rows:", len(st.session_state.defender_profiles))
        if st.session_state.defender_profiles.empty:
            st.warning("No observed defender-shot aggregates available. The simulation still uses conservative box-score/lineup defensive proxies; it does NOT have observed individual contests or tracking matchups.")
        else:
            st.success("Observed defended-shot aggregates loaded. These are aggregate proxies, not optical tracking or actual shot-by-shot matchups.")
        if meta.get('defender_categories'):
            st.caption(f"Available defended-shot categories: {meta['defender_categories']}")
        if 'defender_raw_rows' in meta:
            st.write('NBA.com raw defended-shot rows:', meta['defender_raw_rows'])
        if 'defender_identity_matches' in meta:
            st.write('Current-roster defender matches:', meta['defender_identity_matches'])
        if meta.get("checked_at"):
            st.caption(f"Last fetch attempt: {meta['checked_at']}")
        if meta.get("defender_prior_season"):
            st.caption(f"Defensive history: {meta['defender_prior_season']} · prior weight {meta.get('defender_prior_weight', 0):.2f} · validated defender rows {meta.get('defenders', 0)}")
        for err in meta.get("errors", []):
            st.warning(err)
        if st.button("↻ Fetch NBA.com lineup and defender aggregates", disabled=mode != "Live NBA data" or teams is None or players is None):
            with st.spinner("Fetching public NBA.com aggregate data; this can take a minute…"):
                try:
                    st.session_state.advanced_feed_meta = fetch_public_feeds(season_str, players, teams)
                    st.session_state.lineup_priors, st.session_state.defender_profiles, _ = read_advanced_cache(season_str)
                    st.success("Fetch finished. Review source errors and record counts before using the data.")
                except Exception as exc:
                    st.error(f"Fetch failed: {type(exc).__name__}: {exc}")
    with st.expander("Five-man lineups and individual defender inputs", expanded=False):
        st.caption("Optional observed lineup frequency and defender-by-zone CSVs. These are NOT optical tracking feeds; missing information falls back to conservative priors.")
        lfile = st.file_uploader("Observed five-man lineup frequencies (CSV)", type="csv", key="lineup_csv")
        if lfile is not None and st.button("Load observed lineup frequencies"):
            try:
                st.session_state.lineup_priors = validate_lineups(pd.read_csv(lfile))
                st.success(f"Loaded {len(st.session_state.lineup_priors)} observed lineup records")
            except Exception as exc:
                st.error(str(exc))
        dfile = st.file_uploader("Defender-by-zone observed shooting (CSV)", type="csv", key="defender_csv")
        if dfile is not None and st.button("Load defender observations"):
            try:
                st.session_state.defender_profiles = validate_defenders(pd.read_csv(dfile))
                st.success(f"Loaded {len(st.session_state.defender_profiles)} defender-zone records")
            except Exception as exc:
                st.error(str(exc))
        st.caption(f"Lineup records: {len(st.session_state.lineup_priors)} · defender-zone records: {len(st.session_state.defender_profiles)}. Historical units are blended with new minute-weighted combinations; small, concentrated or ineligible samples are discounted. Exact substitutions and defensive assignments remain unknown.")
    hero("Shot Intelligence Lab.", "Shot locations, Bayesian shooting efficiency, opponent-zone adjustments and experimental shot-driven simulations.", "05 / SHOT MODEL")
    st.info("NEW · Defensive lineup engine: turnover pressure, block rates, defensive rebounding, rim deterrence and lineup-weighted defender shot contests now influence possession simulations. Tune in sidebar → Shot-level intelligence. All estimates are conservative proxies, not verified one-on-one matchups.")
    st.warning("Research mode: shot locations and defender proximity are different datasets. This model does not know who is guarding whom or where every defender stands. NBA.com can rate-limit downloads.")
    c1, c2 = st.columns(2)
    with c1:
        shot_upload = st.file_uploader("Import aggregated player shot chart (CSV)", type="csv", key="shot_upload",
             help="TEAM_ABBREVIATION, PLAYER_NAME, ZONE, FGA, FGM")
        if shot_upload is not None and st.button("Load player shot chart", key="load_shots"):
            try:
                incoming = validate_profiles(pd.read_csv(shot_upload))
                st.session_state.shot_profiles = incoming
                st.session_state.shot_source = "User-imported shot-zone CSV"
                st.session_state.pop("result", None)
                st.success(f"Loaded {len(incoming)} player-zone records. Rerun simulation.")
            except Exception as exc:
                st.error(str(exc))
    with c2:
        def_upload = st.file_uploader("Import opponent zone defense (CSV)", type="csv", key="def_upload",
            help="TEAM_ABBREVIATION, ZONE, FGA_ALLOWED, FGM_ALLOWED")
        if def_upload is not None and st.button("Load defensive zones", key="load_def"):
            try:
                st.session_state.defense_profiles = validate_defense(pd.read_csv(def_upload))
                st.session_state.pop("result", None)
                st.success("Loaded opponent-zone defense. Rerun simulation.")
            except Exception as exc:
                st.error(str(exc))
    st.caption("NBA.com shot chart source: user-initiated download. Select a real matchup and click below; not available in demo mode.")
    if mode == "Live NBA data" and home and away and teams is not None and "TEAM_ID" in teams.columns:
        if st.button(f"Fetch real shot locations for {away} and {home} · {season_str}", type="primary"):
            try:
                fetched = []
                for abbr in (away, home):
                    tid = teams.loc[teams["TEAM_ABBREVIATION"] == abbr, "TEAM_ID"].iloc[0]
                    with st.spinner(f"Downloading {abbr} shot attempts from NBA.com…"):
                        fetched.append(fetch_team_shots(int(tid), season_str, abbr))
                combined = pd.concat(fetched, ignore_index=True)
                prior = st.session_state.shot_profiles
                remaining = prior[~prior["TEAM_ABBREVIATION"].isin([home, away])]
                st.session_state.shot_profiles = validate_profiles(pd.concat([remaining, combined], ignore_index=True))
                SHOT_DIR.mkdir(parents=True, exist_ok=True)
                st.session_state.shot_profiles.to_csv(SHOT_DIR / f"shots_{season_str}.csv", index=False)
                st.session_state.shot_source = f"NBA.com ShotChartDetail · {season_str}"
                st.session_state.pop("result", None)
                st.success(f"Fetched {len(combined)} zone records; enable shot sampling in the sidebar and rerun.")
            except Exception as exc:
                st.error(f"NBA.com shot feed unavailable: {type(exc).__name__}: {exc}. Import a CSV instead.")
    coverage = st.session_state.shot_profiles
    if coverage.empty:
        st.info("No actual shot chart loaded. The experimental model will use league-prior zone splits until you import or fetch data.")
    else:
        st.caption("Current shot data: " + st.session_state.shot_source)
        st.dataframe(coverage, hide_index=True, use_container_width=True, height=260)
        st.download_button("Export shot-zone profiles", coverage.to_csv(index=False), "shot_profiles.csv", "text/csv")
        if home and away and not active_roster.empty:
            options = active_roster[["PLAYER_NAME", "TEAM_ABBREVIATION"]].drop_duplicates()
            labels = [f"{r.PLAYER_NAME} · {r.TEAM_ABBREVIATION}" for r in options.itertuples()]
            pick = st.selectbox("Inspect shot profile", labels)
            row = options.iloc[labels.index(pick)]
            team_code = row["TEAM_ABBREVIATION"]
            opp = home if team_code == away else away
            info = player_zone_model(row["PLAYER_NAME"], team_code, coverage, st.session_state.defense_profiles,
                                     opp, S.get("defense_strength", 1.))
            view = pd.DataFrame({"Zone": ZONES, "Attempts": info["attempts"].astype(int),
                                 "Zone share": info["shares"], "Projected FG%": info["make_prob"],
                                 "Defensive adjustment": info["defense_adjustment"]})
            st.dataframe(view.style.format({"Zone share":"{:.1%}", "Projected FG%":"{:.1%}",
                                            "Defensive adjustment":"{:+.1%}"}), hide_index=True, use_container_width=True)
            st.metric("Modeled points per field-goal attempt", f"{info['points_per_attempt']:.2f}")
            st.caption(f"Sample size {info['sample_attempts']:.0f} FGA · {info['source']}. Make rates are shrunk toward league priors; opponent allowed FG% is a noisy defensive proxy, not an individual defender matchup.")
    with st.expander("Supported CSV formats and model limitations"):
        st.code("TEAM_ABBREVIATION,PLAYER_NAME,ZONE,FGA,FGM\nBOS,Example Player,Rim,120,83\nBOS,Example Player,Above-break 3,150,57", language="csv")
        st.code("TEAM_ABBREVIATION,ZONE,FGA_ALLOWED,FGM_ALLOWED\nNYK,Rim,500,305", language="csv")
        st.caption("Allowed zones: " + ", ".join(ZONES) + ". NBA.com shot data excludes precise on-ball defender positions; possession ordering and foul-by-foul events are not modeled. No historical backtest for shot mode has been established.")

elif page == "Player Lab":
    hero("Go beyond season averages.", "Distributions, realistic uncertainty ranges and fantasy-impact explanations for each player.", "04 / PLAYER INTELLIGENCE")
    if matching:
        box = result["box"].sort_values("PTS", ascending=False)
        pick = st.selectbox("Player", box["Player"].tolist(), key="player_focus")
        row = box[box["Player"] == pick].iloc[0]
        v = result["samples"][row["ID"]]
        c = st.columns(4)
        for col, (lbl, field) in zip(c, [("Minutes", "MIN"), ("Points", "PTS"), ("Rebounds", "REB"), ("Assists", "AST")]):
            with col: metric_card(lbl, f'{row[field]:.1f}', f'10–90%: {row[field+"_P10"]:.1f}–{row[field+"_P90"]:.1f}')
        c2 = st.columns(4)
        for col, (lbl, field) in zip(c2, [("Steals", "STL"), ("Blocks", "BLK"), ("Turnovers", "TOV"), ("Made threes", "3PM")]):
            with col: metric_card(lbl, f'{row[field]:.1f}')
        stats = st.multiselect("Chart player outcomes", ["PTS", "REB", "AST", "MIN", "STL", "BLK", "TOV", "3PM", "PRA", "FANTASY"], default=["PTS", "REB", "AST"])
        items = {**v, "FANTASY": fantasy_values(v, scoring)}
        if S["show_distribution"]:
            cols = st.columns(2)
            for j, stat in enumerate(stats):
                with cols[j % 2]: plot_hist(items[stat], f"{pick} · {stat}")
        if S["show_insights"]:
            st.caption("Model context: minutes constrained by simulated rotation, status and caps; statistical production reweighted for game environment and offensive share. Injury labels from ESPN are unconfirmed until verified; estimated percentile bands are model-specific and not calibrated guarantees.")
    else:
        st.info("Run a simulation in Command Center, then select any player.")

elif page == "Prop Lab":
    hero("Understand the entire prop distribution.", "Compare a chosen line against simulated player outcomes. Market odds are entered manually.", "05 / PROBABILITY RESEARCH")
    if matching:
        box = result["box"].sort_values("PTS", ascending=False)
        a,b,c = st.columns([1.6, 1., 1.])
        with a: pname = st.selectbox("Player", box["Player"].tolist(), key="prop_player")
        with b: market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA", "3PM", "STL", "BLK", "TOV", "FANTASY"])
        key = box.loc[box["Player"] == pname, "ID"].iloc[0]
        vals = fantasy_values(result["samples"][key], scoring) if market == "FANTASY" else result["samples"][key][market]
        with c: line = st.number_input("Line", min_value=0.0, max_value=150.0, value=float(np.floor(vals.mean()) + .5), step=.5)
        over, under, push = float(np.mean(vals > line)), float(np.mean(vals < line)), float(np.mean(vals == line))
        col1,col2,col3,col4 = st.columns(4)
        with col1: metric_card("Over probability", f"{over*100:.1f}%")
        with col2: metric_card("Under probability", f"{under*100:.1f}%")
        with col3: metric_card("Projected mean", f"{np.mean(vals):.1f}")
        with col4: metric_card("Push probability", f"{push*100:.1f}%")
        if S["show_distribution"]: plot_hist(vals, f"{pname} · {market} distribution", line)
        section("Optional sportsbook price comparison")
        l, r = st.columns(2)
        with l: american_over = st.number_input("Over odds (American)", min_value=-1000, max_value=1000, value=-110)
        with r: american_under = st.number_input("Under odds (American)", min_value=-1000, max_value=1000, value=-110)
        def implied(odds):
            return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)
        a, b = implied(american_over), implied(american_under)
        vigfree_o = a/(a+b)
        st.caption(f"Sportsbook implied over chance: **{a*100:.1f}%**; normalized no-vig estimate: **{vigfree_o*100:.1f}%**. Simulated over: **{over*100:.1f}%**. These numbers are useful only if the underlying model has been independently calibrated. Lines are manually entered; no odds feed is connected.")
    else:
        st.info("Run a simulation to examine prop probabilities.")

elif page == "Live Sync":
    hero("Your lineup intelligence, continuously refreshed.",
         "Monitor injury status changes, roster membership, and significant statistical shifts. Updates alter unlocked simulation inputs.",
         "07 / AUTOMATIC DATA MONITOR")
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Automatic checking", "ACTIVE" if S.get("auto_sync") and mode == "Live NBA data" else "PAUSED", f'Every {S["sync_minutes"]} min · while dashboard open')
    with c2: metric_card("Teams being tracked", str(len(verified_rosters)), S.get("watch_scope", "Selected matchup"))
    with c3: metric_card("Last comparison", (sync_last_check() or "Never")[:19].replace("T", " "), "UTC · locally logged")
    st.caption("The roster and injury feed comes from an unofficial ESPN endpoint. NBA.com player/team baselines are cached for up to one hour. The injury/roster cache lasts 5 minutes. No polling occurs after you close the dashboard or your Mac sleeps.")
    if latest_changes:
        st.success(f"{len(latest_changes)} newly detected update(s) on this refresh.")
    section("Recent changes · newest first")
    alerts = sync_history()
    if alerts:
        changes_df = pd.DataFrame(alerts)
        cols = ["timestamp", "type", "team", "player", "before", "after", "detail"]
        st.dataframe(changes_df[[x for x in cols if x in changes_df]], use_container_width=True,
                     hide_index=True, height=380)
        st.download_button("↓ Export update history", changes_df.to_csv(index=False),
                           file_name="courtside_feed_changes.csv", mime="text/csv")
    else:
        st.info("Waiting for two successful readings of the same team feeds. The first check creates the baseline; later changes appear here.")
    section("Current injury feed · selected matchup")
    if raw_injuries:
        st.dataframe(pd.DataFrame(raw_injuries), hide_index=True, use_container_width=True)
    else:
        st.caption("No usable injury listings returned for this matchup. This does not confirm everyone is healthy.")
    if injury_errors:
        st.warning("Some feeds could not be verified: " + " · ".join(injury_errors[:5]))
    st.markdown("**Important:** 'Roster addition' and 'Roster departure' are feed changes, not verified transactions. An injury disappearing from a secondary feed is not medical clearance. Verify important decisions with the [NBA official injury report](https://official.nba.com/nba-injury-report-2026-27-season/).")

elif page == "NBA Database":
    hero("NBA game intelligence archive", "Every scheduled game, full available box scores, play-by-play, shots, fouls, substitutions, estimated possessions and pace — stored locally with source records.", "08 / HISTORICAL DATA ENGINE")
    st.caption("The database collects games independently of simulations. It is not an optical-tracking feed: missing defender assignments and lineup stints are explicitly marked unavailable.")
    try:
        with archive_connect() as db:
            summary = archive_overview(db)
            recent_games = pd.read_sql_query("""SELECT game_date,away,home,away_score,home_score,state,detail_status
                      FROM games ORDER BY game_date DESC,game_id DESC LIMIT 50""",db)
            coverage_rows = pd.read_sql_query("""SELECT category, COUNT(*) AS games_checked,
                      SUM(available) AS games_with_data FROM coverage GROUP BY category ORDER BY category""",db)
            pace_rows = pd.read_sql_query("""SELECT team,COUNT(*) AS games,ROUND(AVG(pace_est),2) AS pace,
                      ROUND(AVG(off_rating_est),2) AS offensive_rating,
                      ROUND(AVG(def_rating_est),2) AS defensive_rating,
                      ROUND(AVG(oreb_pct)*100,1) AS offensive_rebound_pct,
                      ROUND(AVG(efg_pct)*100,1) AS effective_fg_pct,
                      ROUND(AVG(tov_pct)*100,1) AS turnover_pct,
                      ROUND(AVG(ft_rate),3) AS ft_attempt_rate
                      FROM team_games WHERE pace_est IS NOT NULL GROUP BY team ORDER BY games DESC""",db)
            errors = pd.read_sql_query("SELECT day,game_id,error FROM sync_errors ORDER BY day DESC LIMIT 20",db)
        cols=st.columns(5)
        for col,(label,key) in zip(cols,[("Games discovered","games_discovered"),
              ("Complete box scores","games_complete"),("Player game records","players_logged"),
              ("Play-by-play events","plays_logged"),("Shots logged","shots_logged")]):
            col.metric(label, f"{summary[key]:,}")
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[("Pace-calculated games","estimated_pace_games"),
             ("Shots with coordinates","shots_with_coordinates"),
             ("Substitutions","substitutions_logged"),("Injury snapshots","injury_snapshots")]):
            col.metric(label,f"{summary[key]:,}")
        st.caption(f"Last sync: {summary['last_sync_utc'] or 'Not yet'} · Historical backfill next date: {summary['backfill_cursor'] or 'Not started'} · Database: {ARCHIVE_PATH}")
        st.markdown("#### Collection controls")
        S["archive_auto"] = st.toggle("Collect all games automatically when app opens", value=bool(S.get("archive_auto", True)))
        S["archive_backfill"] = st.toggle("Backfill previous seasons in batches", value=bool(S.get("archive_backfill", True)))
        b1,b2,b3=st.columns(3)
        def launch_archive(days,backfill):
            try:
                if CLOUD_MODE:
                    st.warning("Background archive workers are disabled on shared free hosting.")
                    return
                subprocess.Popen([sys.executable,str(Path(__file__).with_name("archive_worker.py")),
                        "--recent-days",str(days),"--max-days",str(backfill)],
                        cwd=str(Path(__file__).parent),stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,start_new_session=True)
                st.success("Archive collection started separately. Refresh this page after a minute to see progress. The worker skips already-complete games.")
            except Exception as exc:st.error(f"Cannot start archive: {exc}")
        if b1.button("↻ Sync recent games",use_container_width=True): launch_archive(7,0)
        if b2.button("⏪ Continue history (14 days)",use_container_width=True):launch_archive(3,14)
        if b3.button("⏪ Continue history (60 days)",use_container_width=True):launch_archive(3,60)
        st.caption("The background worker catches up from October 2023. Historical collection takes time and requires internet. A daily Mac scheduler can run it even when the dashboard is closed; install it using INSTALL_BACKGROUND_SYNC.command inside the app's source folder.")
        st.markdown("#### Game archive")
        if not recent_games.empty:st.dataframe(recent_games,hide_index=True,use_container_width=True)
        else:st.info("No games collected yet. Start a sync; the database begins empty and grows as verified games arrive.")
        st.markdown("#### Inspect one game — every available event")
        if not recent_games.empty:
            ids = []
            with archive_connect() as db:
                ids = db.execute("""SELECT game_id,game_date,away,home,detail_status FROM games
                         ORDER BY game_date DESC,game_id DESC LIMIT 150""").fetchall()
            labels = [f"{r['game_date']} · {r['away']} @ {r['home']} · {r['detail_status']} · {r['game_id']}" for r in ids]
            selection=st.selectbox("Choose archived game",labels,key="archive_game_selector")
            if selection:
                gid=ids[labels.index(selection)]['game_id']
                with archive_connect() as db:
                    gp=pd.read_sql_query("SELECT * FROM team_games WHERE game_id=?",db,params=(gid,))
                    pp=pd.read_sql_query("""SELECT team,player_name,starter,minutes,points,reb,ast,stl,blk,tov,pf,
                         fgm,fga,fg3m,fg3a,ftm,fta,oreb,dreb,plus_minus FROM player_games
                         WHERE game_id=? ORDER BY team,minutes DESC""",db,params=(gid,))
                    ev=pd.read_sql_query("""SELECT p.period,p.clock,p.team,p.event_type,p.description,
                         p.home_score,p.away_score,f.category,f.shot_zone,f.clutch
                         FROM plays p LEFT JOIN play_features f USING(game_id,play_id)
                         WHERE p.game_id=? ORDER BY p.sequence""",db,params=(gid,))
                    cov=pd.read_sql_query("SELECT category,available,note FROM coverage WHERE game_id=?",db,params=(gid,))
                    ctx=pd.read_sql_query("SELECT * FROM game_team_context WHERE game_id=?",db,params=(gid,))
                tab1,tab2,tab3,tab4=st.tabs(["Teams + pace","Player minutes + fouls","Every event","Coverage + context"])
                with tab1:
                    st.dataframe(gp.drop(columns=['raw_json'],errors='ignore'),hide_index=True,use_container_width=True)
                with tab2:
                    st.dataframe(pp,hide_index=True,use_container_width=True)
                with tab3:
                    st.dataframe(ev,hide_index=True,use_container_width=True,height=450)
                    if not ev.empty:st.download_button("↓ Export full event log",ev.to_csv(index=False),
                                       file_name=f"courtside_events_{gid}.csv",mime="text/csv")
                with tab4:
                    st.dataframe(cov,hide_index=True,use_container_width=True)
                    st.dataframe(ctx,hide_index=True,use_container_width=True)
        st.markdown("#### Automatic video discovery (optional)")
        st.caption("Matches candidate videos against both teams and game date, and labels likely clips versus longer uploads. Verified metadata is NOT verified footage or full-game coverage. Authorized local event clips can be analyzed automatically by the background video worker.")
        _key_file=Path.home()/'.courtsideiq'/'youtube_api_key'
        _new_key=st.text_input("YouTube Data API key (optional; stored locally)",type="password",key="video_discovery_key")
        if st.button("Save video discovery API key"):
            _key_file.parent.mkdir(parents=True,exist_ok=True)
            _key_file.write_text(_new_key.strip())
            _key_file.chmod(0o600)
            st.success("Saved locally. No Prime Video credentials needed.")
        if st.button("Discover public video candidates for recent games"):
            from automatic_video import discover as _discover_video
            with archive_connect() as db:
                _key=_new_key.strip() or (_key_file.read_text().strip() if _key_file.exists() else '')
                st.json(_discover_video(db,_key))
        with archive_connect() as db:
            from automatic_video import setup as _setup_video_discoveries
            _setup_video_discoveries(db)
            _candidate_videos=pd.read_sql_query("SELECT game_id,title,url,channel,status,verification_reason,video_kind,duration_seconds FROM video_discoveries ORDER BY checked_utc DESC LIMIT 30",db)
            if not _candidate_videos.empty:st.dataframe(_candidate_videos,hide_index=True,use_container_width=True)
        st.caption("For unattended authorized local clip analysis, run INSTALL_AUTOMATIC_VIDEO.command in the app source folder. Your Mac must be awake and vision dependencies installed.")
        st.markdown("#### Full-game play-by-play learning and video coverage")
        st.caption("The entire available event timeline is collected automatically. Linked video clips are separate: they do not equal a full-game recording. Shot-selection learning uses only past games with sufficiently complete shot-event data.")
        if not recent_games.empty:
            from pbp_learning import game_coverage, team_event_profile, index_local_event_clips
            with archive_connect() as db:
                _ids=db.execute("SELECT game_id,game_date,away,home FROM games WHERE state='post' ORDER BY game_date DESC LIMIT 50").fetchall()
                if _ids:
                    _labels=[f"{r['game_date']} {r['away']} @ {r['home']} · {r['game_id']}" for r in _ids]
                    _selected=st.selectbox("Game event/video coverage",_labels,key="pbp_video_game")
                    _id=_ids[_labels.index(_selected)]['game_id']
                    st.json(game_coverage(db,_id))
                    _clip_rows=pd.read_sql_query("SELECT play_id,source,status,local_path,video_ref FROM event_videos WHERE game_id=? LIMIT 150",db,params=(_id,))
                    if not _clip_rows.empty:st.dataframe(_clip_rows,hide_index=True,use_container_width=True)
                    _team=_ids[_labels.index(_selected)]['home']
                    st.json({'example_team_event_profile':team_event_profile(db,_team,_ids[_labels.index(_selected)]['game_date'])})
        st.caption("Optional local event clips: put files named GAMEID__PLAYID.mp4 in ~/.courtsideiq/event_clips. This links clips to exact play-by-play events; it does not download restricted broadcasts.")
        if st.button("Link local event video clips"):
            from pbp_learning import index_local_event_clips
            with archive_connect() as db: st.json(index_local_event_clips(db,Path.home()/'.courtsideiq'/'event_clips'))
        if st.button("Analyze linked local event clips (experimental)"):
            from courtside_vision import process_video as _process_event_video
            from pbp_learning import setup as _setup_event_video
            with archive_connect() as db:
                _setup_event_video(db)
                _rows=db.execute("SELECT game_id,play_id,video_ref,local_path FROM event_videos WHERE local_path IS NOT NULL AND (vision_video_id IS NULL OR status!='analyzed') LIMIT 12").fetchall()
            if not _rows:st.info("No new linked local clips to analyze.")
            for _r in _rows:
                try:
                    _res=_process_event_video(_r['local_path'],max_frames=600,stride=3)
                    with archive_connect() as db:
                        db.execute("UPDATE event_videos SET vision_video_id=?,status='analyzed',checked_utc=datetime('now') WHERE game_id=? AND play_id=? AND video_ref=?",
                                   (_res['video_id'],_r['game_id'],_r['play_id'],_r['video_ref']))
                        db.commit()
                    st.success(f"Linked { _r['game_id'] } / { _r['play_id'] }: { _res['status'] }")
                except Exception as _exc:st.error(f"Clip {_r['play_id']}: {_exc}")
        st.markdown("#### Pace, offense, defense and four factors")
        if not pace_rows.empty:
            st.dataframe(pace_rows,hide_index=True,use_container_width=True)
            st.download_button("↓ Export team pace and efficiency",pace_rows.to_csv(index=False),"courtside_team_pace.csv","text/csv")
        else:st.info("Team pace and efficiency appear once complete team box scores are available.")
        st.markdown("#### Source coverage (unavailable ≠ zero)")
        if not coverage_rows.empty:st.dataframe(coverage_rows,hide_index=True,use_container_width=True)
        st.markdown("#### Data-quality and synchronization errors")
        if not errors.empty:st.dataframe(errors,hide_index=True,use_container_width=True)
        else:st.caption("No stored collection errors.")
        st.markdown("#### Historical inputs used by the simulator")
        st.json(st.session_state.get("archive_evidence",{"status":"Not evaluated for current matchup"}))
        st.caption("The simulator blends archive minutes and pace only from games before the selected game date. NBA.com historical priors remain primary. Backtesting must use time-correct NBA.com snapshots to avoid future leakage.")
    except Exception as exc:
        st.error(f"Archive unavailable: {type(exc).__name__}: {exc}")

elif page == "COURTSIDE VISION":
    hero("COURTSIDE VISION", "Analyze permitted game and event clips alongside the full-game play-by-play archive.", "09 / COMPUTER VISION")
    st.warning("Experimental. This is NOT Second Spectrum: generic object detection, no reliable NBA player names, no calibrated feet/second, no verified possession, and no automatic access to NBA broadcasts. No unvalidated video estimates enter the prediction engine.")
    st.caption("Runs on your Mac. First run downloads a pretrained detection model. CPU processing may be slow; video analysis is separate from the game-statistics collector.")
    from courtside_vision import connect as vision_connect, INBOX as VISION_INBOX, OUT as VISION_OUT, process_video as vision_process
    with st.expander("Install video tracking dependencies", expanded=True):
        st.code(".venv/bin/python -m pip install ultralytics opencv-python-headless", language="bash")
        st.caption("Simpler: use the INSTALL_VISION.command file included alongside the app. It locates the app's private Python environment.")
    VISION_INBOX.mkdir(parents=True,exist_ok=True)
    VISION_OUT.mkdir(parents=True,exist_ok=True)
    st.markdown(f"**Automatic folder:** `{VISION_INBOX}`")
    st.caption("Copy your own authorized MP4/MOV game videos into that folder. Use 'Process inbox' to handle every unprocessed file. For unattended processing while your Mac is awake, run WATCH_VISION.command.")
    uploaded=st.file_uploader("Upload a game clip (saved locally before analysis)",type=["mp4","mov","m4v","avi","mkv"],key="vision_upload")
    if uploaded and st.button("Save video to Vision inbox"):
        from pathlib import Path as _P
        target=VISION_INBOX/_P(uploaded.name).name
        with target.open('wb') as f:
            while True:
                chunk=uploaded.read(1024*1024)
                if not chunk:break
                f.write(chunk)
        st.success(f"Saved {target.name} ({target.stat().st_size:,} bytes)")
    v_stride=st.select_slider("Frame sampling (lower = more detail, slower)",options=[1,2,3,5,10],value=2)
    v_limit=st.number_input("Maximum analyzed frames per video (0 = entire video)",min_value=0,max_value=1000000,value=0,step=100)
    videos=sorted([p for p in VISION_INBOX.iterdir() if p.is_file() and p.suffix.lower() in {'.mp4','.mov','.m4v','.avi','.mkv'}])
    st.caption(f"{len(videos)} videos in inbox")
    if st.button("Process inbox with local computer vision",type="primary",disabled=not videos):
        bar=st.progress(0)
        for index,video in enumerate(videos):
            with st.status(f"Analyzing {video.name}",expanded=False) as status:
                try:
                    def progress(done,total):bar.progress(min(1.,(index+min(1.,done/max(total,1)))/len(videos)))
                    result=vision_process(video,stride=v_stride,max_frames=int(v_limit),progress=progress)
                    st.json(result);status.update(label=f"Finished: {video.name}",state="complete")
                except Exception as exc:
                    st.error(f"{video.name}: {exc}");status.update(label=f"Failed: {video.name}",state="error")
        bar.progress(1.)
    try:
        with vision_connect() as conn:
            video_df=pd.read_sql_query("SELECT video_id,source_path,processed_utc,frames_sampled,detections,status,output_video,data_quality FROM vision_videos ORDER BY processed_utc DESC LIMIT 40",conn)
            counts=pd.read_sql_query("SELECT category,COUNT(*) AS observations,COUNT(DISTINCT video_id) AS videos FROM vision_observations GROUP BY category",conn)
        st.subheader("Local video tracking database")
        if video_df.empty:st.info("No videos processed yet. This database does not contain raw tracking until videos are analyzed.")
        else:
            st.dataframe(video_df,hide_index=True,use_container_width=True)
            st.dataframe(counts,hide_index=True,use_container_width=True)
            selected=st.selectbox("Review tracking output",video_df['video_id'].tolist())
            row=video_df.loc[video_df['video_id']==selected].iloc[0]
            if row['output_video'] and Path(row['output_video']).exists():
                with open(row['output_video'],'rb') as f:st.download_button("Download annotated MP4",f,file_name=f"{selected}_tracked.mp4",mime="video/mp4")
            with vision_connect() as conn:
                metrics=pd.read_sql_query("SELECT * FROM vision_metrics WHERE video_id=? ORDER BY observations DESC LIMIT 100",conn,params=(selected,))
                events=pd.read_sql_query("SELECT * FROM vision_events WHERE video_id=? ORDER BY seconds LIMIT 100",conn,params=(selected,))
            st.markdown("**Visible track metrics (pixels, not feet)**");st.dataframe(metrics,hide_index=True,use_container_width=True)
            st.markdown("**Ball proximity transitions (not confirmed passes)**");st.dataframe(events,hide_index=True,use_container_width=True)
    except Exception as exc:st.error(f"Video database unavailable: {exc}")

elif page == "Accuracy Lab":
    hero("Automatic prediction audit", "Local pregame forecast snapshots and automatically verified final-game results. Every recorded prediction retains its original timestamp.", "08 / CONTINUOUS VALIDATION")
    if st.session_state.get("audit_save_message"):
        st.caption(st.session_state["audit_save_message"])
    if st.button("↻ Grade all eligible finished games now"):
        with st.spinner("Checking completed NBA games and grading saved pregame predictions…"):
            graded, errors = grade_completed(fetch_games, fetch_actual, max_games=20)
            st.session_state["auto_grades"] = graded
            st.session_state["auto_grade_errors"] = errors
    if "auto_grades" not in st.session_state:
        from automatic_audit import GRADE_PATH, _read
        st.session_state["auto_grades"] = _read(GRADE_PATH)
    grades = st.session_state["auto_grades"]
    summary = grade_summary(grades)
    if not summary.empty:
        st.metric("Automatically graded forecasts", len(summary))
        st.metric("Overall fantasy MAE", f"{summary['fantasy_mae'].mean():.2f}")
        st.dataframe(summary.sort_values("saved_at", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("↓ Export verified accuracy history", summary.to_csv(index=False), "courtside_accuracy.csv", "text/csv")
        frame = calibration_frame(grades)
        if not frame.empty:
            st.caption(f"{len(frame)} unique completed games with pregame win probabilities. Formal calibration requires at least 80 games across 15 dates; no fabricated observations.")
            st.download_button("↓ Export real game calibration dataset", frame.to_csv(index=False), "courtside_calibration.csv", "text/csv")
    else:
        st.info("No completed forecasts have been graded yet. New predictions are saved before tipoff when auto-saving is enabled; grades appear after games become final and the dashboard checks them.")
    for err in st.session_state.get("auto_grade_errors", [])[:3]:
        st.warning("Grading feed: " + err)
    with st.expander("Historical win-probability calibration · time-ordered holdout", expanded=True):
        st.caption("Upload genuine, timestamped pregame predictions and final results. Required: GAME_DATE, HOME_WIN_PROB, HOME_WON (0/1). Recommended: GAME_ID and FORECAST_SAVED_AT. No fabricated historical scores.")
        calfile = st.file_uploader("Completed-game calibration CSV", type="csv", key="calibration_csv")
        if calfile is not None and st.button("Evaluate chronological probability calibration"):
            try:
                assessment = evaluate_calibration(pd.read_csv(calfile))
                st.session_state.calibration_audit = assessment
            except Exception as exc:
                st.error(str(exc))
        if st.session_state.get("calibration_audit") is not None:
            a = st.session_state.calibration_audit
            c1, c2, c3 = st.columns(3)
            c1.metric("Holdout games", a['holdout_games'])
            c2.metric("Raw Brier score", f"{a['raw_brier']:.4f}")
            c3.metric("Calibrated Brier score", f"{a['calibrated_brier']:.4f}")
            st.write("Held-out log loss: raw", round(a['raw_log_loss'], 4), "calibrated", round(a['calibrated_log_loss'], 4))
            st.success("Calibration improved BOTH holdout scores; research candidate only.") if a['passed_holdout'] else st.warning("Calibration did not improve both holdout scores; do not deploy it.")
            st.dataframe(a['holdout'], use_container_width=True, hide_index=True)
    hero("Show the receipts.", "Timestamp pregame predictions and compare them with final NBA box scores. Honest tracking beats impressive-looking percentages.", "08 / VALIDATION + ACCOUNTABILITY")
    if matching:
        st.caption("Eligible snapshots require an actual scheduled NBA game and a real-data source, saved before tipoff. Demo and past games cannot be passed off as verified pregame forecasts.")
        if st.button("Save pregame forecast for tracking", type="primary"):
            if selected_game is None:
                st.error("Select a scheduled game rather than a custom matchup.")
            else:
                try:
                    snapshot = store_snapshot(result, selected_game, scoring, source_description)
                    st.success(f"Forecast saved under {snapshot['id']}. It will be available on this Mac for later review.")
                except Exception as exc:
                    st.error(str(exc))
    hist = load_history()
    if hist:
        snaps = pd.DataFrame([{k: s.get(k) for k in ["id", "saved_at", "matchup", "game_id", "simulations", "source"]} for s in hist])
        section("Saved pregame forecasts")
        st.dataframe(snaps.sort_values("saved_at", ascending=False), use_container_width=True, hide_index=True)
        selected = st.selectbox("Evaluate snapshot", [s["id"] for s in reversed(hist)])
        record = next(x for x in hist if x["id"] == selected)
        if st.button("Fetch final box score and grade forecast"):
            try:
                day = date.fromisoformat(record["tipoff"][:10])
                historic = fetch_games(str(day))
                original = next((g for g in historic if str(g["id"]) == str(record["game_id"])), None)
            except Exception:
                original = None
            if original is None or original["state"] != "post":
                st.error("Unable to verify that this game is final. Accuracy grading is disabled until the final status can be confirmed.")
            else:
                try:
                    actual = fetch_actual(record["game_id"])
                    grade = grade_snapshot(record, actual)
                    if grade.empty:
                        st.warning("No matching athlete statistics were found. ESPN box-score fields or roster names may differ.")
                    else:
                        st.metric("Fantasy projection MAE", f'{grade["Abs. Error"].mean():.2f}')
                        st.dataframe(grade.sort_values("Abs. Error"), hide_index=True, use_container_width=True)
                        st.download_button("↓ Download accuracy report", grade.to_csv(index=False), f"accuracy_{selected}.csv", "text/csv")
                except Exception as exc:
                    st.error(f"Could not grade the box score: {exc}")
    else:
        st.info("No saved pregame predictions yet. Run a real NBA matchup and save it before tipoff to start a personal accuracy history.")
    section("How to know if it actually works")
    st.markdown("Track **mean absolute error** for projected minutes and every scoring category, fantasy-point error, win-probability calibration, performance against naive season-average baselines, and true out-of-sample results. A better-looking dashboard isn't a more accurate model.")

elif page == "ML Studio":
    hero("Train it. Test it. Trust the evidence.",
         "Real historical NBA box scores, leak-resistant pregame features, and a time-ordered holdout. Track every model against a rolling baseline.",
         "09 / HISTORICAL INTELLIGENCE")
    if st.session_state.get("model_sync_message"):
        st.info(st.session_state["model_sync_message"])
    st.caption("Training can take several minutes and requires NBA.com access or an imported genuine NBA LeagueGameLog CSV. Forecasts remain baseline-only until a completed training run is saved.")
    if (TRAINING_FOLDER / "manifest.json").exists():
        try:
            manifest, _, _ = load_trained(TRAINING_FOLDER)
            a, b, c = st.columns(3)
            with a: metric_card("Historical player-games", f"{manifest['all_games']:,}", "Usable rows")
            with b: metric_card("Holdout player-games", f"{manifest['holdout_games']:,}", f"From {manifest['holdout_from']}")
            with c: metric_card("Latest archived game", manifest['latest_game'], "Not a live feed timestamp")
            perf = pd.DataFrame(manifest["results"])
            st.dataframe(perf.rename(columns={"stat":"Statistic", "model_mae":"ML MAE", "baseline_mae":"Baseline MAE", "improvement_pct":"Change %", "live_model_weight":"Active ML weight", "holdout_n":"N", "residual_sd":"Residual SD"}),
                         hide_index=True, use_container_width=True)
            if manifest.get("fantasy"):
                fm = manifest["fantasy"]
                m1, m2 = st.columns(2)
                with m1: st.metric("Held-out fantasy MAE", f"{fm['model']['mae']:.2f}")
                with m2: st.metric("Rolling-baseline fantasy MAE", f"{fm['baseline']['mae']:.2f}")
            st.caption("Held-out evaluation is chronological (latest 20% of dates). It is NOT evidence of profitability, calibration or model parity with betting markets.")
        except Exception as exc:
            st.error(f"Saved model unreadable: {type(exc).__name__}: {exc}")
    else:
        st.info("No historical model is installed yet. Train below; the standard simulation remains usable without one.")
    st.subheader("Train / retrain the prediction system")
    historic_mode = st.radio("Historical data source", ["Download actual NBA game logs", "Upload NBA game-log CSV"], horizontal=True)
    logs_to_train = None
    if historic_mode == "Download actual NBA game logs":
        selected_years = st.multiselect("NBA regular seasons", ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"],
                                        default=["2023-24", "2024-25", "2025-26"])
        st.caption("NBA.com can throttle or block automated requests. Downloaded seasons are cached on this Mac. Use CSV mode if needed.")
    else:
        uploaded_logs = st.file_uploader("NBA LeagueGameLog CSV (one file with all seasons)", type=["csv"], key="training_logs")
        if uploaded_logs is not None:
            try:
                logs_to_train = pd.read_csv(uploaded_logs, dtype={"GAME_ID":str, "PLAYER_ID":str})
                st.success(f"Loaded {len(logs_to_train):,} game-log rows. Columns will be validated during training.")
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")
    if st.button("Train + run chronological holdout", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading historical games, fitting separate stat models, and checking unseen matchups…"):
                if historic_mode == "Download actual NBA game logs":
                    if not selected_years:
                        raise ValueError("Select at least one season")
                    logs_to_train = fetch_historical(tuple(selected_years), TRAINING_FOLDER / "game_logs")
                if logs_to_train is None:
                    raise ValueError("Upload historical player game logs")
                completed = train_models(logs_to_train, folder=TRAINING_FOLDER)
            st.success(f"Trained on {completed['all_games']:,} usable player games. Latest archived game: {completed['latest_game']}")
            st.session_state.pop("result", None)
            st.session_state.pop("slate_result", None)
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed without overwriting existing model: {type(exc).__name__}: {str(exc)[:260]}")
    st.subheader("Player-level model provenance")
    audit = st.session_state.get("model_audit")
    if isinstance(audit, pd.DataFrame) and not audit.empty:
        st.dataframe(audit, hide_index=True, use_container_width=True)
    else:
        st.caption("After training, run a pregame matchup and return here to inspect each player's baseline source.")
    st.subheader("Validation methodology")
    st.markdown("Each historical training row uses only earlier performances, earlier opponent trends, rest and venue. The most recent 20% of dates are withheld, and a trained stat model is activated only when it beats a rolling-average baseline by at least 1% MAE. After evaluation, a production copy is refit on known historical games. Injury projections still depend on feed quality; historical test metrics do not include today's unverified injury changes.")

elif page == "Market Lab":
    hero("Model vs market.",
         "Compare simulated prop distributions with available sportsbook prices. Raw differences aren't automatically betting edges.",
         "10 / OPTIONAL MARKET INTELLIGENCE")
    st.caption("This panel uses The Odds API if you supply your own key. Key is used only for a request in this local session; it isn't saved in COURTSIDE IQ settings. Event/player markets may need a paid API plan.")
    market_key = st.text_input("The Odds API key", type="password", placeholder="Paste key to fetch current markets")
    if not matching:
        st.info("Run a matchup prediction before comparing sportsbook prices.")
    if st.button("Find available NBA games and prices", disabled=not bool(market_key)):
        try:
            with st.spinner("Loading current book listings…"):
                st.session_state["market_events"] = fetch_market_events(market_key)
                st.session_state.pop("market_prices", None)
            st.success(f"Found {len(st.session_state['market_events'])} currently listed games")
        except Exception as exc:
            st.error(f"Odds service error: {type(exc).__name__}: {str(exc)[:180]}")
    events = st.session_state.get("market_events", [])
    if events:
        options = {f"{e.get('away_team','?')} @ {e.get('home_team','?')} · {e.get('commence_time','')[:16]}": e for e in events}
        matched_names = []
        if selected_game is not None:
            matching_events = match_event(events, selected_game['away']['name'], selected_game['home']['name'])
            matched_ids = {e.get('id') for e in matching_events}
            matched_names = [label for label, e in options.items() if e.get('id') in matched_ids]
        selected_name = st.selectbox("Sportsbook matchup", matched_names or list(options))
        market_event = options[selected_name]
        market_match_confirmed = bool(matched_names and selected_name in matched_names)
        if not market_match_confirmed:
            st.warning("No exact book-market match to the selected simulation game. Comparison is disabled to avoid mixing different matchups.")
        if st.button("Fetch player props for this matchup", disabled=not bool(market_key)):
            try:
                with st.spinner("Loading available player prop quotes…"):
                    odds = fetch_event_odds(market_event['id'], market_key)
                    st.session_state["market_prices"] = parse_props(odds)
                    st.session_state["market_event_id"] = market_event['id']
                st.success(f"Received {len(st.session_state['market_prices'])} prop quotes")
            except Exception as exc:
                st.error(f"Could not load player props: {type(exc).__name__}: {str(exc)[:190]}")
    quotes = st.session_state.get("market_prices")
    if isinstance(quotes, pd.DataFrame):
        if quotes.empty:
            st.warning("No prop quotes were returned. This plan or bookmaker may not provide that market.")
        elif matching and selected_game is not None and events and 'market_event' in locals() and \
                st.session_state.get('market_event_id') == market_event.get('id') and market_match_confirmed:
            comparison = compare_props(result, quotes)
            if comparison.empty:
                st.warning("No player names matched this simulated game. Check that the sportsbook matchup equals the selected NBA matchup.")
            else:
                st.warning("Research only. Model probabilities have NOT been demonstrated to be calibrated to real odds; the apparent differences can be misleading. Quotes may already have moved.")
                columns = ["Book", "Player", "Stat", "Line", "Side", "American odds", "Model probability", "Raw implied probability", "No-vig fair probability", "Fair probability gap", "Hypothetical $1 EV", "Updated"]
                st.dataframe(comparison[columns], hide_index=True, use_container_width=True)
                st.download_button("Export market comparison CSV", comparison.to_csv(index=False), file_name="courtside_market_comparison.csv", mime="text/csv")
            st.caption("No-vig fair probability requires both Over and Under quoted for the same player/line/book. Probabilities are from your current Monte Carlo run. Markets are not assumed historically available.")

elif page == "Appearance":
    hero("Make it your workspace.", "Themes, accents, chart shapes, layouts and content density are yours to control.", "11 / PERSONALIZATION")
    a, b, c = st.columns(3)
    with a: metric_card("Current theme", S["theme"], "Night / daylight palette")
    with b: metric_card("Accent color", S["accent"], "Six curated accent palettes")
    with c: metric_card("Dashboard density", S["density"], "Compact or comfortable")
    st.info("Use Display preferences in the sidebar for custom color hex, font scaling, chart types, card corners, table density and optional sections. Save your preferences locally to keep them between launches.")
    section("Brand system preview")
    st.markdown('<div class="glass"><div class="eyebrow">CUSTOM INTERFACE</div>'
                '<h2 style="margin:.5rem 0">Control your court.</h2>'
                '<p class="micro">Responsive, streamlined analytics surfaces with crisp tables and interactive plots. No generic casino-style UI.</p>'
                '<span class="pill live">LIVE DATA</span><span class="pill">150K SIMULATIONS</span>'
                '<span class="pill">FANTASY PROJECTIONS</span></div>', unsafe_allow_html=True)

# One-stop diagnostics on EVERY page, so no sidebar hunting or separate exports.
st.divider()
st.markdown("### 📋 Send everything to ChatGPT · One box")
st.caption("After running a matchup, click inside the box, press ⌘A then ⌘C, and paste it into ChatGPT. Or download the same complete report as one JSON file. No screenshots needed.")
current_result = st.session_state.get("result")
if current_result is None:
    st.info("Run a matchup in Command Center first. Your report will appear here automatically on every page.")
else:
    try:
        # Cache by run identity so switching pages does not recompute summaries.
        run_key = (id(current_result), str(season_str), str(current_result.get("home")), str(current_result.get("away")))
        if st.session_state.get("one_box_report_key") != run_key:
            text_report = make_report(
                current_result,
                game=f"{current_result.get('away')} @ {current_result.get('home')}",
                season=season_str, data_mode=mode,
                source=st.session_state.get("result_meta", {}).get("source", source_description),
                injury_issues=injury_errors,
                shot_profiles=st.session_state.get("shot_profiles"),
                lineup_rows=len(st.session_state.get("lineup_priors", [])),
                defender_rows=len(st.session_state.get("defender_profiles", [])),
                player_inputs=st.session_state.get('last_simulation_inputs', players_for_selected if 'players_for_selected' in globals() else None),
                team_inputs=teams,
                model_used=bool(st.session_state.get("model_used")),
                model_issue=st.session_state.get("model_issue"),
                defender_feed_meta=st.session_state.get("advanced_feed_meta"),
                shot_source=st.session_state.get("shot_source"),
            )
            st.session_state["one_box_report"] = text_report
            st.session_state["one_box_report_key"] = run_key
        text_report = st.session_state["one_box_report"]
        st.code(text_report, language="json", wrap_lines=True)
        st.download_button(
            "↓ Download EVERYTHING in one diagnostic JSON",
            data=text_report.encode("utf-8"),
            file_name=f"COURTSIDE_IQ_Full_Diagnostic_{current_result.get('away')}_{current_result.get('home')}.json",
            mime="application/json", key="one_box_download", use_container_width=True,
        )
        st.caption("This includes the full projected player box scores, input statistics, shot-chart coverage, pace/event summaries, calibration, injuries, and simulation verification. Raw 150,000-game event traces are not stored in this report.")
    except Exception as exc:
        st.error(f"Diagnostic report could not be generated: {type(exc).__name__}: {exc}")

st.markdown('<div class="footer">COURTSIDE IQ · Private local research dashboard · Not affiliated with the NBA, ESPN or any sportsbook. Simulations are experimental and unvalidated. Auto Sync checks feeds while the app is open; source injuries, rosters and live stats may be delayed or incomplete. Treat predictions as uncertain estimates, not facts.</div>', unsafe_allow_html=True)
