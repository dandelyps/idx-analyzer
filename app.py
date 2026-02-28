import streamlit as st
from supabase import create_client
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# --- COMPACT STYLE ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px 12px; }
    .stMetric label { font-size: 0.75rem !important; color: #666 !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.8rem; padding: 6px 12px; }
    .stDataFrame { font-size: 0.82rem; }
    h1 { font-size: 1.5rem !important; margin-bottom: 0 !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    .stAlert { padding: 8px 12px; font-size: 0.85rem; }
    .stSelectbox label, .stNumberInput label, .stTextInput label { font-size: 0.82rem; }
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data(ttl=300)
def load_ratios():
    response = supabase.table("fundamental_ratios").select("*").execute()
    return pd.DataFrame(response.data)

@st.cache_data(ttl=300)
def load_company_history(ticker):
    response = supabase.table("fundamental_ratios").select("*").eq("ticker", ticker).execute()
    df = pd.DataFrame(response.data)
    if not df.empty:
        df = df[df["period"] == "annual"].sort_values("year")
    return df

@st.cache_data(ttl=300)
def load_companies():
    response = supabase.table("companies").select("*").execute()
    return pd.DataFrame(response.data)

@st.cache_data(ttl=300)
def load_financials():
    response = supabase.table("financial_statements").select("*").execute()
    return pd.DataFrame(response.data)

@st.cache_data(ttl=300)
def load_market_data():
    response = supabase.table("market_data").select("*").execute()
    return pd.DataFrame(response.data)

def clear_cache():
    load_ratios.clear()
    load_company_history.clear()
    load_companies.clear()
    load_financials.clear()
    load_market_data.clear()

# --- SESSION STATE ---
if "page" not in st.session_state:
    st.session_state.page = "screener"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📊 IDX Analyzer")
    st.divider()
    if st.button("🔍 Screener", use_container_width=True):
        st.session_state.page = "screener"
        st.session_state.selected_ticker = None
        st.rerun()
    if st.button("📊 Sector Comparison", use_container_width=True):
        st.session_state.page = "comparison"
        st.session_state.selected_ticker = None
        st.rerun()
    if st.button("➕ Data Input Center", use_container_width=True):
        st.session_state.page = "data_input"
        st.session_state.selected_ticker = None
        st.rerun()
    st.divider()
    st.caption("Indonesia Market · Personal Tool")


# ============================================================
# SCREENER PAGE
# ============================================================
def show_screener():
    st.title("📊 IDX Fundamental Analyzer")

    df = load_ratios()
    if df.empty:
        st.warning("No data yet. Use the Data Input Center to add companies.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Sector", sectors, label_visibility="collapsed")
    with c2:
        years = sorted(df["year"].dropna().unique().tolist(), reverse=True)
        selected_year = st.selectbox("Year", years, label_visibility="collapsed")
    with c3:
        search = st.text_input("Search", placeholder="🔍 Search ticker or name...", label_visibility="collapsed")

    filtered = df[df["year"] == selected_year].copy()
    if selected_sector != "All":
        filtered = filtered[filtered["sector"] == selected_sector]
    if search:
        filtered = filtered[
            filtered["ticker"].str.contains(search.upper(), na=False) |
            filtered["name"].str.contains(search, case=False, na=False)
        ]

    display_cols = {
        "ticker": "Ticker", "name": "Company", "sector": "Sector",
        "per": "PER", "pbv": "PBV", "roe": "ROE %",
        "net_margin": "Net Margin %", "der": "DER",
        "revenue_growth_yoy": "Rev Growth %", "dividend_yield": "Div Yield %"
    }
    available = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[available].rename(columns=display_cols)
    numeric_cols = display_df.select_dtypes(include="number").columns
    display_df[numeric_cols] = display_df[numeric_cols].round(2)

    st.caption(f"{len(display_df)} companies · FY{selected_year} · click ticker for details")
    st.divider()

    cols_layout = [0.8, 2, 1.5, 0.7, 0.7, 0.8, 1, 0.7, 1, 0.8]
    header = st.columns(cols_layout)
    for i, label in enumerate(["Ticker", "Company", "Sector", "PER", "PBV", "ROE %", "Net Margin %", "DER", "Rev Growth %", "Div Yield %"]):
        header[i].markdown(f"<small><b>{label}</b></small>", unsafe_allow_html=True)
    st.divider()

    for _, row in display_df.iterrows():
        cols = st.columns(cols_layout)
        if cols[0].button(str(row["Ticker"]), key=f"btn_{row['Ticker']}_{selected_year}"):
            st.session_state.selected_ticker = row["Ticker"]
            st.session_state.page = "company"
            st.rerun()
        cols[1].write(row["Company"])
        cols[2].write(row["Sector"])
        for i, col_name in enumerate(["PER", "PBV", "ROE %", "Net Margin %", "DER", "Rev Growth %", "Div Yield %"]):
            val = row.get(col_name, None)
            cols[i+3].write("—" if pd.isna(val) else val)


# ============================================================
# COMPANY DETAIL PAGE
# ============================================================
def show_company(ticker):
    df = load_company_history(ticker)
    if df.empty:
        st.error(f"No data found for {ticker}")
        return

    latest = df.iloc[-1]
    company_name = latest.get("name", ticker)

    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("← Back"):
            st.session_state.page = "screener"
            st.session_state.selected_ticker = None
            st.rerun()
    with c2:
        st.markdown(f"### {ticker} — {company_name}")
        st.caption(f"{latest.get('sector','')} · {latest.get('sub_sector','')} · FY{int(latest.get('year',''))} · {latest.get('report_unit','')} IDR")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Snapshot", "📋 Financials", "📈 Trends",
        "🚨 Red Flags", "💰 DCF", "💵 Dividends"
    ])

    # TAB 1: SNAPSHOT
    with tab1:
        def mc(label, value, fmt="{:.2f}", suffix=""):
            st.metric(label, "—" if (value is None or pd.isna(value)) else f"{fmt.format(value)}{suffix}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("**VALUATION**")
            mc("PER", latest.get("per"))
            mc("PBV", latest.get("pbv"))
        with c2:
            st.caption("**VALUATION**")
            mc("P/S", latest.get("ps_ratio"))
            mc("P/FCF", latest.get("p_fcf"))
        with c3:
            st.caption("**PROFITABILITY**")
            mc("ROE", latest.get("roe"), suffix="%")
            mc("ROA", latest.get("roa"), suffix="%")
        with c4:
            st.caption("**PROFITABILITY**")
            mc("Net Margin", latest.get("net_margin"), suffix="%")
            mc("Gross Margin", latest.get("gross_margin"), suffix="%")

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("**HEALTH**")
            mc("DER", latest.get("der"))
            mc("Current Ratio", latest.get("current_ratio"))
        with c2:
            st.caption("**HEALTH**")
            mc("Interest Coverage", latest.get("interest_coverage"))
            mc("OCF / NI", latest.get("ocf_to_ni"))
        with c3:
            st.caption("**GROWTH YoY**")
            mc("Revenue", latest.get("revenue_growth_yoy"), suffix="%")
            mc("Net Income", latest.get("ni_growth_yoy"), suffix="%")
        with c4:
            st.caption("**GROWTH YoY**")
            mc("Equity", latest.get("equity_growth_yoy"), suffix="%")
            mc("DPS", latest.get("dps_growth_yoy"), suffix="%")

    # TAB 2: FINANCIALS
    with tab2:
        st.caption(f"All figures in {latest.get('report_unit','reported unit')} IDR")

        def show_table(cols_map):
            avail = [c for c in cols_map if c in df.columns]
            t = df[avail].rename(columns=cols_map).set_index("Year")
            t[t.select_dtypes(include="number").columns] = t.select_dtypes(include="number").round(1)
            st.dataframe(t, use_container_width=True)

        show_table({"year":"Year","revenue":"Revenue","gross_profit":"Gross Profit","ebit":"EBIT",
                    "net_income":"Net Income","depreciation_amortization":"D&A",
                    "interest_expense":"Interest Exp","eps_diluted":"EPS"})
        show_table({"year":"Year","cash":"Cash","accounts_receivable":"Receivables","inventory":"Inventory",
                    "total_current_assets":"Curr Assets","total_assets":"Total Assets",
                    "total_current_liabilities":"Curr Liab","total_debt":"Total Debt","total_equity":"Equity"})
        show_table({"year":"Year","operating_cash_flow":"OCF","capex":"Capex",
                    "fcf":"FCF","dividends_paid":"Dividends"})

    # TAB 3: TRENDS
    with tab3:
        def lc(title, col, suffix=""):
            if col in df.columns and df[col].notna().any():
                fig = px.line(df, x="year", y=col, markers=True,
                              title=f"{title}{' ('+suffix+')' if suffix else ''}")
                fig.update_layout(height=250, margin=dict(t=30,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

        def bc(title, col):
            if col in df.columns and df[col].notna().any():
                fig = px.bar(df, x="year", y=col, title=title)
                fig.update_layout(height=250, margin=dict(t=30,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1: bc("Revenue", "revenue")
        with c2: bc("Net Income", "net_income")
        with c3: lc("Net Margin", "net_margin", "%")

        c1, c2, c3 = st.columns(3)
        with c1: lc("ROE", "roe", "%")
        with c2: lc("DER", "der")
        with c3: lc("Current Ratio", "current_ratio")

        c1, c2, c3 = st.columns(3)
        with c1: lc("PER", "per")
        with c2: lc("PBV", "pbv")
        with c3: lc("OCF / NI", "ocf_to_ni")

    # TAB 4: RED FLAGS
    with tab4:
        flags, warnings, green = [], [], []

        if len(df) >= 2:
            last = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.notna(last.get("total_debt")) and pd.notna(prev.get("total_debt")) and \
               pd.notna(last.get("revenue")) and pd.notna(prev.get("revenue")):
                debt_growth = (last["total_debt"] - prev["total_debt"]) / abs(prev["total_debt"]) * 100
                rev_growth = (last["revenue"] - prev["revenue"]) / abs(prev["revenue"]) * 100
                if debt_growth > rev_growth + 20:
                    flags.append(f"🔴 Debt grew {debt_growth:.1f}% vs revenue growth {rev_growth:.1f}% — debt outpacing revenue")
                else:
                    green.append(f"🟢 Debt growth ({debt_growth:.1f}%) in line with revenue growth ({rev_growth:.1f}%)")

            margins = df["net_margin"].dropna().tolist()
            if len(margins) >= 3 and margins[-1] < margins[-2] < margins[-3]:
                flags.append(f"🔴 Net margin declining 3 years in a row ({margins[-3]:.1f}% → {margins[-2]:.1f}% → {margins[-1]:.1f}%)")
            elif len(margins) >= 2 and margins[-1] < margins[-2]:
                warnings.append(f"🟡 Net margin dropped this year ({margins[-2]:.1f}% → {margins[-1]:.1f}%)")
            elif len(margins) >= 2:
                green.append(f"🟢 Net margin stable or improving ({margins[-2]:.1f}% → {margins[-1]:.1f}%)")

            if pd.notna(last.get("fcf")) and pd.notna(last.get("net_income")):
                if last["fcf"] < 0 and last["net_income"] > 0:
                    flags.append(f"🔴 Negative FCF ({last['fcf']:.0f}) while reporting positive net income")
                elif last["fcf"] > 0:
                    green.append(f"🟢 Positive Free Cash Flow ({last['fcf']:.0f})")

            if pd.notna(last.get("ocf_to_ni")):
                if last["ocf_to_ni"] < 0.7:
                    warnings.append(f"🟡 OCF/Net Income ratio is low ({last['ocf_to_ni']:.2f}) — profit may not be fully backed by cash")
                elif last["ocf_to_ni"] >= 1.0:
                    green.append(f"🟢 OCF/Net Income = {last['ocf_to_ni']:.2f} — strong earnings quality")

            if pd.notna(last.get("accounts_receivable")) and pd.notna(prev.get("accounts_receivable")) and \
               pd.notna(last.get("revenue")) and pd.notna(prev.get("revenue")):
                rec_growth = (last["accounts_receivable"] - prev["accounts_receivable"]) / abs(prev["accounts_receivable"]) * 100
                rev_growth = (last["revenue"] - prev["revenue"]) / abs(prev["revenue"]) * 100
                if rec_growth > rev_growth + 15:
                    warnings.append(f"🟡 Receivables growing faster than revenue ({rec_growth:.1f}% vs {rev_growth:.1f}%)")

            if pd.notna(last.get("dividends_paid")) and pd.notna(last.get("net_income")):
                if last["dividends_paid"] > 0 and last["net_income"] > 0:
                    if last["dividends_paid"] > last["net_income"]:
                        flags.append(f"🔴 Dividends paid ({last['dividends_paid']:,.0f}) exceeded net income ({last['net_income']:,.0f})")
                    elif last["dividends_paid"] > last["net_income"] * 0.9:
                        warnings.append(f"🟡 Dividends paid is {last['dividends_paid']/last['net_income']*100:.0f}% of net income — very high payout")

            if pd.notna(last.get("der")):
                if last["der"] > 2.0:
                    flags.append(f"🔴 High DER of {last['der']:.2f} — significant leverage risk")
                elif last["der"] > 1.0:
                    warnings.append(f"🟡 DER of {last['der']:.2f} — moderate leverage")
                else:
                    green.append(f"🟢 DER of {last['der']:.2f} — healthy leverage level")

            if pd.notna(last.get("interest_coverage")):
                if last["interest_coverage"] < 2:
                    flags.append(f"🔴 Interest coverage {last['interest_coverage']:.2f}x — dangerously low")
                elif last["interest_coverage"] < 4:
                    warnings.append(f"🟡 Interest coverage {last['interest_coverage']:.2f}x — adequate but tight")
                else:
                    green.append(f"🟢 Interest coverage {last['interest_coverage']:.2f}x — comfortable")

        if flags:
            st.markdown("**🔴 Critical Flags**")
            for f in flags: st.error(f)
        if warnings:
            st.markdown("**🟡 Watch Items**")
            for w in warnings: st.warning(w)
        if green:
            st.markdown("**🟢 Healthy Signals**")
            for g in green: st.success(g)
        if not flags and not warnings and not green:
            st.info("Need at least 2 years of data.")

    # TAB 5: DCF
    with tab5:
        latest_fcf = latest.get("fcf")
        latest_price = latest.get("stock_price")

        if pd.isna(latest_fcf) or latest_fcf is None:
            st.warning("FCF not available. Enter Operating Cash Flow and Capex first.")
        else:
            st.caption(f"Base FCF: {latest_fcf:,.0f} {latest.get('report_unit','')}")
            c1, c2 = st.columns(2)
            with c1:
                growth_1_5 = st.slider("Growth Yr 1–5 (%)", 0, 30, 10) / 100
                growth_6_10 = st.slider("Growth Yr 6–10 (%)", 0, 20, 5) / 100
                terminal_growth = st.slider("Terminal growth (%)", 0, 5, 2) / 100
                discount_rate = st.slider("WACC (%)", 5, 20, 10) / 100

            fcf = latest_fcf
            projected_fcfs = []
            for yr in range(1, 11):
                fcf = fcf * (1 + (growth_1_5 if yr <= 5 else growth_6_10))
                pv = fcf / ((1 + discount_rate) ** yr)
                projected_fcfs.append({"Year": f"Yr {yr}", "FCF": round(fcf, 1), "PV": round(pv, 1)})

            terminal_value = projected_fcfs[-1]["FCF"] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / ((1 + discount_rate) ** 10)
            total_pv = sum(p["PV"] for p in projected_fcfs) + pv_terminal
            shares = latest.get("shares_normalized") or latest.get("shares_outstanding", 1)
            iv_per_share = total_pv / shares if shares else None

            with c2:
                st.metric("PV of FCFs", f"{sum(p['PV'] for p in projected_fcfs):,.0f}")
                st.metric("PV Terminal Value", f"{pv_terminal:,.0f}")
                st.metric("Total Intrinsic Value", f"{total_pv:,.0f}")
                if iv_per_share:
                    st.metric("Per Share", f"Rp {iv_per_share:,.0f}")
                    if latest_price and not pd.isna(latest_price):
                        margin = (iv_per_share - latest_price) / iv_per_share * 100
                        if margin > 0:
                            st.success(f"✅ Margin of Safety: {margin:.1f}%")
                        else:
                            st.error(f"⚠️ {abs(margin):.1f}% above intrinsic value")

            st.dataframe(pd.DataFrame(projected_fcfs), use_container_width=True, hide_index=True)

    # TAB 6: DIVIDENDS
    with tab6:
        div_cols = ["year","dividend_per_share","dividend_yield","payout_ratio","fcf_payout_ratio","dps_growth_yoy"]
        available_div = [c for c in div_cols if c in df.columns]
        div_df = df[available_div].copy()
        has_div = div_df["dividend_per_share"].notna().any() if "dividend_per_share" in div_df.columns else False

        if not has_div:
            st.info("No dividend data yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Div Yield", f"{latest.get('dividend_yield'):.2f}%" if pd.notna(latest.get('dividend_yield')) else "—")
            with c2: st.metric("Payout Ratio", f"{latest.get('payout_ratio'):.1f}%" if pd.notna(latest.get('payout_ratio')) else "—")
            with c3: st.metric("FCF Payout", f"{latest.get('fcf_payout_ratio'):.1f}%" if pd.notna(latest.get('fcf_payout_ratio')) else "—")
            with c4: st.metric("Years Paying", f"{div_df['dividend_per_share'].notna().sum()} yr(s)")

            c1, c2 = st.columns(2)
            with c1:
                if div_df["dividend_per_share"].notna().any():
                    fig = px.bar(div_df.dropna(subset=["dividend_per_share"]), x="year", y="dividend_per_share", title="DPS History")
                    fig.update_layout(height=250, margin=dict(t=30,b=10))
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "dividend_yield" in div_df.columns and div_df["dividend_yield"].notna().any():
                    fig = px.line(div_df.dropna(subset=["dividend_yield"]), x="year", y="dividend_yield", markers=True, title="Yield %")
                    fig.update_layout(height=250, margin=dict(t=30,b=10))
                    st.plotly_chart(fig, use_container_width=True)

            if pd.notna(latest.get("payout_ratio")):
                pr = latest["payout_ratio"]
                if pr > 100: st.error(f"🔴 Payout {pr:.1f}% — paying more than earned")
                elif pr > 75: st.warning(f"🟡 Payout {pr:.1f}% — high")
                else: st.success(f"🟢 Payout {pr:.1f}% — sustainable")


# ============================================================
# SECTOR COMPARISON PAGE
# ============================================================
def show_comparison():
    st.title("📊 Sector Comparison")

    df = load_ratios()
    if df.empty:
        st.warning("No data yet.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        years = sorted(df["year"].dropna().unique().tolist(), reverse=True)
        sel_year = st.selectbox("Year", years)
    with c2:
        sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
        sel_sector = st.selectbox("Sector", sectors)
    with c3:
        all_tickers = sorted(df["ticker"].dropna().unique().tolist())
        sel_tickers = st.multiselect("Or pick specific companies", all_tickers)

    year_df = df[df["year"] == sel_year].copy()
    if sel_tickers:
        comp_df = year_df[year_df["ticker"].isin(sel_tickers)]
    elif sel_sector != "All":
        comp_df = year_df[year_df["sector"] == sel_sector]
    else:
        comp_df = year_df

    if comp_df.empty:
        st.info("No companies match your selection.")
        return

    st.caption(f"{len(comp_df)} companies · FY{sel_year}")
    st.divider()

    # ── RANKING TABLE ──
    st.markdown("**📋 Rankings Table**")
    rank_cols = {
        "ticker": "Ticker", "name": "Company",
        "per": "PER", "pbv": "PBV", "roe": "ROE %",
        "net_margin": "Net Margin %", "gross_margin": "Gross Margin %",
        "der": "DER", "current_ratio": "Curr Ratio",
        "interest_coverage": "Int Coverage", "ocf_to_ni": "OCF/NI",
        "revenue_growth_yoy": "Rev Growth %", "ni_growth_yoy": "NI Growth %",
        "dividend_yield": "Div Yield %"
    }
    avail_rank = [c for c in rank_cols if c in comp_df.columns]
    rank_df = comp_df[avail_rank].rename(columns=rank_cols)
    rank_df = rank_df.set_index("Ticker")
    if "Company" in rank_df.columns:
        rank_df = rank_df.drop(columns=["Company"])
    rank_df = rank_df.select_dtypes(include="number").round(2)

    def color_table(val, col_name):
        if pd.isna(val): return ""
        good_high = ["ROE %","Net Margin %","Gross Margin %","Curr Ratio","Int Coverage","OCF/NI","Rev Growth %","NI Growth %","Div Yield %"]
        good_low = ["PER","PBV","DER"]
        try:
            col_vals = rank_df[col_name].dropna()
            if len(col_vals) < 2: return ""
            if col_name in good_high:
                if val >= col_vals.quantile(0.66): return "background-color: #d4edda; color: #155724"
                if val <= col_vals.quantile(0.33): return "background-color: #f8d7da; color: #721c24"
            elif col_name in good_low:
                if val <= col_vals.quantile(0.33): return "background-color: #d4edda; color: #155724"
                if val >= col_vals.quantile(0.66): return "background-color: #f8d7da; color: #721c24"
        except: pass
        return ""

    styled = rank_df.style.apply(lambda col: [color_table(v, col.name) for v in col], axis=0)
    st.dataframe(styled, use_container_width=True)
    st.caption("🟢 Best in group · 🔴 Worst in group")
    st.divider()

    # ── BAR CHARTS ──
    st.markdown("**📊 Revenue & Net Income**")
    c1, c2 = st.columns(2)
    with c1:
        if "revenue" in comp_df.columns:
            fig = px.bar(comp_df.sort_values("revenue", ascending=False),
                         x="ticker", y="revenue", color="ticker",
                         title="Revenue", color_discrete_sequence=px.colors.qualitative.Set2,
                         text="revenue")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(height=300, margin=dict(t=30,b=10,r=10), showlegend=False,
                               yaxis=dict(range=[0, comp_df["revenue"].max() * 1.2]))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "net_income" in comp_df.columns:
            fig = px.bar(comp_df.sort_values("net_income", ascending=False),
                         x="ticker", y="net_income", color="ticker",
                         title="Net Income", color_discrete_sequence=px.colors.qualitative.Set2,
                         text="net_income")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(height=300, margin=dict(t=30,b=10,r=10), showlegend=False,
                               yaxis=dict(range=[0, comp_df["net_income"].max() * 1.2]))
            st.plotly_chart(fig, use_container_width=True)

    # ── PROFITABILITY ──
    st.markdown("**📊 Profitability**")
    c1, c2 = st.columns(2)
    with c1:
        if "net_margin" in comp_df.columns and "gross_margin" in comp_df.columns:
            margin_df = comp_df[["ticker","net_margin","gross_margin"]].melt(
                id_vars="ticker", var_name="type", value_name="margin")
            margin_df["type"] = margin_df["type"].map({"net_margin":"Net Margin","gross_margin":"Gross Margin"})
            fig = px.bar(margin_df, x="ticker", y="margin", color="type", barmode="group",
                         title="Margins (%)", color_discrete_sequence=["#2196F3","#4CAF50"],
                         text="margin")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=300, margin=dict(t=30,b=10),
                               yaxis=dict(range=[0, margin_df["margin"].max() * 1.2]))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "roe" in comp_df.columns and "roa" in comp_df.columns:
            returns_df = comp_df[["ticker","roe","roa"]].melt(
                id_vars="ticker", var_name="type", value_name="value")
            returns_df["type"] = returns_df["type"].map({"roe":"ROE","roa":"ROA"})
            fig = px.bar(returns_df, x="ticker", y="value", color="type", barmode="group",
                         title="ROE vs ROA (%)", color_discrete_sequence=["#FF9800","#9C27B0"],
                         text="value")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=300, margin=dict(t=30,b=10),
                               yaxis=dict(range=[0, returns_df["value"].max() * 1.2]))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── RADAR CHART ──
    st.markdown("**🕸️ Radar Chart — Multi-Ratio Comparison**")
    st.caption("Normalized 0–100 within selected group (100 = best)")

    radar_metrics = {
        "net_margin": "Net Margin",
        "roe": "ROE",
        "current_ratio": "Curr Ratio",
        "interest_coverage": "Int Coverage",
        "revenue_growth_yoy": "Rev Growth",
        "ocf_to_ni": "OCF Quality",
    }

    radar_df = comp_df[["ticker"] + [m for m in radar_metrics if m in comp_df.columns]].copy()
    radar_df = radar_df.dropna(thresh=3)

    if len(radar_df) < 2:
        st.info("Need at least 2 companies with complete data for radar chart.")
    else:
        normalized = radar_df.copy()
        for col in radar_metrics:
            if col in normalized.columns:
                col_min = normalized[col].min()
                col_max = normalized[col].max()
                if col_max > col_min:
                    normalized[col] = (normalized[col] - col_min) / (col_max - col_min) * 100
                else:
                    normalized[col] = 50

        fig = go.Figure()
        categories = [radar_metrics[m] for m in radar_metrics if m in normalized.columns]
        colors = px.colors.qualitative.Set2

        for i, (_, row) in enumerate(normalized.iterrows()):
            values = [row[m] for m in radar_metrics if m in normalized.columns]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=row["ticker"],
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            height=420, margin=dict(t=30,b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── RED FLAG SUMMARY ──
    st.markdown("**🚨 Red Flag Summary**")
    st.caption("Quick health check across all selected companies")

    flag_rows = []
    for _, company_row in comp_df.iterrows():
        t = company_row["ticker"]
        hist = load_company_history(t)
        if len(hist) < 2:
            flag_rows.append({"Ticker": t, "🔴 Critical": 0, "🟡 Watch": 0, "🟢 Healthy": 0, "Overall": "⬜ Not enough data"})
            continue

        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        f, w, g = 0, 0, 0

        if pd.notna(last.get("total_debt")) and pd.notna(prev.get("total_debt")) and \
           pd.notna(last.get("revenue")) and pd.notna(prev.get("revenue")):
            dg = (last["total_debt"] - prev["total_debt"]) / abs(prev["total_debt"]) * 100
            rg = (last["revenue"] - prev["revenue"]) / abs(prev["revenue"]) * 100
            if dg > rg + 20: f += 1
            else: g += 1

        margins = hist["net_margin"].dropna().tolist()
        if len(margins) >= 3 and margins[-1] < margins[-2] < margins[-3]: f += 1
        elif len(margins) >= 2 and margins[-1] < margins[-2]: w += 1
        elif len(margins) >= 2: g += 1

        if pd.notna(last.get("fcf")) and pd.notna(last.get("net_income")):
            if last["fcf"] < 0 and last["net_income"] > 0: f += 1
            elif last["fcf"] > 0: g += 1

        if pd.notna(last.get("ocf_to_ni")):
            if last["ocf_to_ni"] < 0.7: w += 1
            elif last["ocf_to_ni"] >= 1.0: g += 1

        if pd.notna(last.get("dividends_paid")) and pd.notna(last.get("net_income")):
            if last["dividends_paid"] > 0 and last["net_income"] > 0:
                if last["dividends_paid"] > last["net_income"]: f += 1
                elif last["dividends_paid"] > last["net_income"] * 0.9: w += 1

        if pd.notna(last.get("der")):
            if last["der"] > 2.0: f += 1
            elif last["der"] > 1.0: w += 1
            else: g += 1

        if pd.notna(last.get("interest_coverage")):
            if last["interest_coverage"] < 2: f += 1
            elif last["interest_coverage"] < 4: w += 1
            else: g += 1

        if f == 0 and w == 0: status = "🟢 All clear"
        elif f == 0: status = "🟡 Watch"
        elif f >= 2: status = "🔴 Multiple issues"
        else: status = "🔴 Critical flag"

        flag_rows.append({"Ticker": t, "🔴 Critical": f, "🟡 Watch": w, "🟢 Healthy": g, "Overall": status})

    st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)


# ============================================================
# DATA INPUT CENTER
# ============================================================
def show_data_input():
    st.title("➕ Data Input Center")

    input_tab1, input_tab2, input_tab3 = st.tabs([
        "🏢 Companies", "📊 Financial Statements", "💹 Stock Prices"
    ])

    with input_tab1:
        companies_df = load_companies()
        col_add, col_edit = st.columns(2)

        with col_add:
            st.subheader("Add New Company")
            with st.form("add_company_form"):
                ticker = st.text_input("Ticker *", placeholder="e.g. BBCA").upper().strip()
                name = st.text_input("Company Name *", placeholder="e.g. Bank Central Asia Tbk")
                sector = st.text_input("Sector *", placeholder="e.g. Financials")
                sub_sector = st.text_input("Sub Sector", placeholder="e.g. Banking")
                shares = st.number_input("Shares Outstanding", min_value=0, value=0, step=1000000)
                report_unit = st.selectbox("Report Unit", ["millions","billions","thousands"])
                submitted = st.form_submit_button("➕ Add Company", use_container_width=True)
                if submitted:
                    if not ticker or not name or not sector:
                        st.error("Ticker, Name, and Sector are required.")
                    else:
                        existing = companies_df[companies_df["ticker"] == ticker] if not companies_df.empty else pd.DataFrame()
                        if not existing.empty:
                            st.error(f"{ticker} already exists. Use edit.")
                        else:
                            try:
                                supabase.table("companies").insert({
                                    "ticker": ticker, "name": name, "sector": sector,
                                    "sub_sector": sub_sector or None,
                                    "shares_outstanding": int(shares) if shares > 0 else None,
                                    "report_unit": report_unit
                                }).execute()
                                clear_cache()
                                st.success(f"✅ {ticker} added!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

        with col_edit:
            st.subheader("Edit Existing Company")
            if companies_df.empty:
                st.info("No companies yet.")
            else:
                edit_ticker = st.selectbox("Select company", companies_df["ticker"].tolist(), key="edit_co")
                company_row = companies_df[companies_df["ticker"] == edit_ticker].iloc[0]
                with st.form("edit_company_form"):
                    e_name = st.text_input("Name", value=str(company_row.get("name","")))
                    e_sector = st.text_input("Sector", value=str(company_row.get("sector","")))
                    e_sub = st.text_input("Sub Sector", value=str(company_row.get("sub_sector","") or ""))
                    e_shares = st.number_input("Shares", value=int(company_row.get("shares_outstanding") or 0), min_value=0, step=1000000)
                    e_unit = st.selectbox("Report Unit", ["millions","billions","thousands"],
                                          index=["millions","billions","thousands"].index(company_row.get("report_unit","millions")))
                    if st.form_submit_button("💾 Save", use_container_width=True):
                        try:
                            supabase.table("companies").update({
                                "name": e_name, "sector": e_sector,
                                "sub_sector": e_sub or None,
                                "shares_outstanding": int(e_shares) if e_shares > 0 else None,
                                "report_unit": e_unit
                            }).eq("id", company_row["id"]).execute()
                            clear_cache()
                            st.success(f"✅ {edit_ticker} updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.divider()
        if not companies_df.empty:
            st.dataframe(companies_df[["ticker","name","sector","sub_sector","shares_outstanding","report_unit"]],
                         use_container_width=True, hide_index=True)

    with input_tab2:
        companies_df = load_companies()
        financials_df = load_financials()

        if companies_df.empty:
            st.warning("Add a company first.")
        else:
            mode = st.radio("Mode", ["Add New","Edit Existing"], horizontal=True)

            if mode == "Edit Existing":
                if financials_df.empty:
                    st.info("No financial statements yet.")
                else:
                    fin_with_ticker = financials_df.merge(
                        companies_df[["id","ticker"]], left_on="company_id", right_on="id", how="left")
                    fin_options = fin_with_ticker.apply(
                        lambda r: f"{r['ticker']} — {int(r['year'])} ({r['period']})", axis=1).tolist()
                    selected_fin = st.selectbox("Select record", fin_options)
                    sel_idx = fin_options.index(selected_fin)
                    fin_row = fin_with_ticker.iloc[sel_idx]
                    fin_id = financials_df.iloc[sel_idx]["id"]

                    def v(col):
                        val = fin_row.get(col)
                        if val is None or (isinstance(val, float) and pd.isna(val)): return 0.0
                        return float(val)

                    st.markdown(f"**Editing: {fin_row['ticker']} FY{int(fin_row['year'])}**")
                    with st.form("edit_fin_form"):
                        st.markdown("**Income Statement**")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            e_rev = st.number_input("Revenue", value=v("revenue"), format="%.0f")
                            e_cogs = st.number_input("COGS", value=v("cogs"), format="%.0f")
                            e_gp = st.number_input("Gross Profit", value=v("gross_profit"), format="%.0f")
                        with c2:
                            e_ebit = st.number_input("EBIT", value=v("ebit"), format="%.0f")
                            e_int = st.number_input("Interest Expense", value=v("interest_expense"), format="%.0f")
                            e_tax = st.number_input("Tax Expense", value=v("tax_expense"), format="%.0f")
                        with c3:
                            e_ni = st.number_input("Net Income", value=v("net_income"), format="%.0f")
                            e_da = st.number_input("D&A", value=v("depreciation_amortization"), format="%.0f")
                            e_eps = st.number_input("EPS Diluted", value=v("eps_diluted"), format="%.2f")

                        st.markdown("**Balance Sheet**")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            e_cash = st.number_input("Cash", value=v("cash"), format="%.0f")
                            e_ar = st.number_input("Accounts Receivable", value=v("accounts_receivable"), format="%.0f")
                            e_inv = st.number_input("Inventory", value=v("inventory"), format="%.0f")
                        with c2:
                            e_ca = st.number_input("Total Current Assets", value=v("total_current_assets"), format="%.0f")
                            e_ta = st.number_input("Total Assets", value=v("total_assets"), format="%.0f")
                            e_cl = st.number_input("Total Current Liabilities", value=v("total_current_liabilities"), format="%.0f")
                        with c3:
                            e_debt = st.number_input("Total Debt", value=v("total_debt"), format="%.0f")
                            e_tl = st.number_input("Total Liabilities", value=v("total_liabilities"), format="%.0f")
                            e_eq = st.number_input("Total Equity", value=v("total_equity"), format="%.0f")

                        st.markdown("**Cash Flow**")
                        c1, c2, c3 = st.columns(3)
                        with c1: e_ocf = st.number_input("Operating CF", value=v("operating_cash_flow"), format="%.0f")
                        with c2: e_capex = st.number_input("Capex", value=v("capex"), format="%.0f")
                        with c3: e_div = st.number_input("Dividends Paid", value=v("dividends_paid"), format="%.0f")

                        if st.form_submit_button("💾 Save Changes", use_container_width=True):
                            try:
                                supabase.table("financial_statements").update({
                                    "revenue": e_rev or None, "cogs": e_cogs or None,
                                    "gross_profit": e_gp or None, "ebit": e_ebit or None,
                                    "interest_expense": e_int or None, "tax_expense": e_tax or None,
                                    "net_income": e_ni or None, "depreciation_amortization": e_da or None,
                                    "eps_diluted": e_eps or None, "cash": e_cash or None,
                                    "accounts_receivable": e_ar or None, "inventory": e_inv or None,
                                    "total_current_assets": e_ca or None, "total_assets": e_ta or None,
                                    "total_current_liabilities": e_cl or None, "total_debt": e_debt or None,
                                    "total_liabilities": e_tl or None, "total_equity": e_eq or None,
                                    "operating_cash_flow": e_ocf or None, "capex": e_capex or None,
                                    "dividends_paid": e_div or None
                                }).eq("id", fin_id).execute()
                                clear_cache()
                                st.success("✅ Updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

            else:
                with st.form("add_fin_form"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        sel_ticker = st.selectbox("Company *", companies_df["ticker"].tolist())
                        sel_year = st.number_input("Year *", min_value=2000, max_value=2030, value=2023, step=1)
                        sel_period = st.selectbox("Period", ["annual","Q1","Q2","Q3","Q4"])

                    st.markdown("**Income Statement**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        n_rev = st.number_input("Revenue", value=0.0, format="%.0f")
                        n_cogs = st.number_input("COGS", value=0.0, format="%.0f")
                        n_gp = st.number_input("Gross Profit", value=0.0, format="%.0f")
                    with c2:
                        n_ebit = st.number_input("EBIT", value=0.0, format="%.0f")
                        n_int = st.number_input("Interest Expense", value=0.0, format="%.0f")
                        n_tax = st.number_input("Tax Expense", value=0.0, format="%.0f")
                    with c3:
                        n_ni = st.number_input("Net Income", value=0.0, format="%.0f")
                        n_da = st.number_input("D&A", value=0.0, format="%.0f")
                        n_eps = st.number_input("EPS Diluted", value=0.0, format="%.2f")

                    st.markdown("**Balance Sheet**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        n_cash = st.number_input("Cash", value=0.0, format="%.0f")
                        n_ar = st.number_input("Accounts Receivable", value=0.0, format="%.0f")
                        n_inv = st.number_input("Inventory", value=0.0, format="%.0f")
                    with c2:
                        n_ca = st.number_input("Total Current Assets", value=0.0, format="%.0f")
                        n_ta = st.number_input("Total Assets", value=0.0, format="%.0f")
                        n_cl = st.number_input("Total Current Liabilities", value=0.0, format="%.0f")
                    with c3:
                        n_debt = st.number_input("Total Debt", value=0.0, format="%.0f")
                        n_tl = st.number_input("Total Liabilities", value=0.0, format="%.0f")
                        n_eq = st.number_input("Total Equity", value=0.0, format="%.0f")

                    st.markdown("**Cash Flow**")
                    c1, c2, c3 = st.columns(3)
                    with c1: n_ocf = st.number_input("Operating CF", value=0.0, format="%.0f")
                    with c2: n_capex = st.number_input("Capex", value=0.0, format="%.0f")
                    with c3: n_div = st.number_input("Dividends Paid", value=0.0, format="%.0f")

                    if st.form_submit_button("➕ Add Financial Statement", use_container_width=True):
                        company_id = companies_df[companies_df["ticker"] == sel_ticker].iloc[0]["id"]
                        try:
                            supabase.table("financial_statements").insert({
                                "company_id": company_id, "year": int(sel_year), "period": sel_period,
                                "revenue": n_rev or None, "cogs": n_cogs or None,
                                "gross_profit": n_gp or None, "ebit": n_ebit or None,
                                "interest_expense": n_int or None, "tax_expense": n_tax or None,
                                "net_income": n_ni or None, "depreciation_amortization": n_da or None,
                                "eps_diluted": n_eps or None, "cash": n_cash or None,
                                "accounts_receivable": n_ar or None, "inventory": n_inv or None,
                                "total_current_assets": n_ca or None, "total_assets": n_ta or None,
                                "total_current_liabilities": n_cl or None, "total_debt": n_debt or None,
                                "total_liabilities": n_tl or None, "total_equity": n_eq or None,
                                "operating_cash_flow": n_ocf or None, "capex": n_capex or None,
                                "dividends_paid": n_div or None
                            }).execute()
                            clear_cache()
                            st.success(f"✅ {sel_ticker} FY{sel_year} added!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    with input_tab3:
        companies_df = load_companies()
        market_df = load_market_data()
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Add / Update Stock Price")
            if companies_df.empty:
                st.info("Add a company first.")
            else:
                with st.form("add_price_form"):
                    p_ticker = st.selectbox("Company", companies_df["ticker"].tolist())
                    p_date = st.date_input("Date")
                    p_price = st.number_input("Stock Price (IDR)", min_value=0.0, value=0.0, format="%.0f")
                    if st.form_submit_button("➕ Add Price", use_container_width=True):
                        if p_price <= 0:
                            st.error("Price must be > 0.")
                        else:
                            company_id = companies_df[companies_df["ticker"] == p_ticker].iloc[0]["id"]
                            try:
                                supabase.table("market_data").upsert({
                                    "company_id": company_id,
                                    "date": str(p_date),
                                    "stock_price": float(p_price)
                                }, on_conflict="company_id,date").execute()
                                clear_cache()
                                st.success(f"✅ {p_ticker} on {p_date} saved!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

        with c2:
            st.subheader("Existing Prices")
            if market_df.empty:
                st.info("No prices yet.")
            else:
                prices_with_ticker = market_df.merge(
                    companies_df[["id","ticker"]], left_on="company_id", right_on="id", how="left")
                display_prices = prices_with_ticker[["ticker","date","stock_price"]].sort_values(
                    ["ticker","date"], ascending=[True,False])
                st.dataframe(display_prices, use_container_width=True, hide_index=True)


# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == "screener":
    show_screener()
elif st.session_state.page == "company" and st.session_state.selected_ticker:
    show_company(st.session_state.selected_ticker)
elif st.session_state.page == "comparison":
    show_comparison()
elif st.session_state.page == "data_input":
    show_data_input()
else:
    st.session_state.page = "screener"
    show_screener()