import streamlit as st
from supabase import create_client
import pandas as pd

# --- CONNECTION ---
supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IDX Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- HEADER ---
st.title("📊 IDX Fundamental Analyzer")
st.caption("Personal stock analysis tool — Indonesia Market")

# --- LOAD DATA ---
@st.cache_data(ttl=300)
def load_ratios():
    response = supabase.table("fundamental_ratios").select("*").execute()
    return pd.DataFrame(response.data)

df = load_ratios()

if df.empty:
    st.warning("No data yet. Add companies and financials in your Supabase database.")
    st.stop()

# --- FILTERS ---
st.subheader("🔍 Screener")

col1, col2, col3 = st.columns(3)

with col1:
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    selected_sector = st.selectbox("Sector", sectors)

with col2:
    years = sorted(df["year"].dropna().unique().tolist(), reverse=True)
    selected_year = st.selectbox("Year", years)

with col3:
    search = st.text_input("Search ticker or name")

# --- FILTER DATA ---
filtered = df[df["year"] == selected_year].copy()

if selected_sector != "All":
    filtered = filtered[filtered["sector"] == selected_sector]

if search:
    filtered = filtered[
        filtered["ticker"].str.contains(search.upper(), na=False) |
        filtered["name"].str.contains(search, case=False, na=False)
    ]

# --- DISPLAY COLUMNS ---
display_cols = {
    "ticker": "Ticker",
    "name": "Company",
    "sector": "Sector",
    "per": "PER",
    "pbv": "PBV",
    "roe": "ROE %",
    "net_margin": "Net Margin %",
    "der": "DER",
    "revenue_growth_yoy": "Rev Growth %",
    "dividend_yield": "Div Yield %"
}

available_cols = [c for c in display_cols.keys() if c in filtered.columns]
display_df = filtered[available_cols].rename(columns=display_cols)

# Round numeric columns
numeric_cols = display_df.select_dtypes(include='number').columns
display_df[numeric_cols] = display_df[numeric_cols].round(2)

# --- TABLE ---
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(f"Showing {len(display_df)} companies")