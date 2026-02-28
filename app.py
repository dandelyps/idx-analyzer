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

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("📊 IDX Analyzer")
    st.divider()
    if st.button("🔍 Screener", use_container_width=True):
        st.session_state.page = "screener"
        st.session_state.selected_ticker = None
        st.rerun()
    if st.button("➕ Data Input Center", use_container_width=True):
        st.session_state.page = "data_input"
        st.session_state.selected_ticker = None
        st.rerun()
    st.divider()
    st.caption("Personal stock analysis tool\nIndonesia Market")


# ============================================================
# SCREENER PAGE
# ============================================================
def show_screener():
    st.title("📊 IDX Fundamental Analyzer")
    st.caption("Personal stock analysis tool — Indonesia Market")

    df = load_ratios()

    if df.empty:
        st.warning("No data yet. Use the Data Input Center to add companies.")
        return

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

    st.caption(f"Showing {len(display_df)} companies — click a ticker to view details")

    header = st.columns([1, 2, 2, 1, 1, 1, 1, 1, 1, 1])
    header[0].markdown("**Ticker**")
    header[1].markdown("**Company**")
    header[2].markdown("**Sector**")
    for i, col_name in enumerate(["PER", "PBV", "ROE %", "Net Margin %", "DER", "Rev Growth %", "Div Yield %"]):
        header[i+3].markdown(f"**{col_name}**")
    st.divider()

    for _, row in display_df.iterrows():
        cols = st.columns([1, 2, 2, 1, 1, 1, 1, 1, 1, 1])
        if cols[0].button(str(row["Ticker"]), key=f"btn_{row['Ticker']}"):
            st.session_state.selected_ticker = row["Ticker"]
            st.session_state.page = "company"
            st.rerun()
        cols[1].write(row["Company"])
        cols[2].write(row["Sector"])
        for i, col_name in enumerate(["PER", "PBV", "ROE %", "Net Margin %", "DER", "Rev Growth %", "Div Yield %"]):
            val = row.get(col_name, None)
            cols[i+3].write("—" if pd.isna(val) else val)

    st.divider()
    st.caption("Showing " + str(len(display_df)) + " companies")


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

    if st.button("← Back to Screener"):
        st.session_state.page = "screener"
        st.session_state.selected_ticker = None
        st.rerun()

    st.title(f"{ticker} — {company_name}")
    st.caption(f"{latest.get('sector', '')} · {latest.get('sub_sector', '')} · Reporting in {latest.get('report_unit', '')}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Snapshot", "📋 Financials", "📈 Ratios & Trends",
        "🚨 Red Flags", "💰 DCF Valuation", "💵 Dividends"
    ])

    # TAB 1: SNAPSHOT
    with tab1:
        st.subheader("Key Metrics — Latest Year")
        year = int(latest.get("year", ""))
        st.caption(f"Based on FY{year} data")

        def metric_card(label, value, format_str="{:.2f}", suffix=""):
            if value is None or pd.isna(value):
                st.metric(label, "—")
            else:
                st.metric(label, f"{format_str.format(value)}{suffix}")

        st.markdown("**Valuation**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("PER", latest.get("per"))
        with c2: metric_card("PBV", latest.get("pbv"))
        with c3: metric_card("P/S", latest.get("ps_ratio"))
        with c4: metric_card("P/FCF", latest.get("p_fcf"))

        st.markdown("**Profitability**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("ROE", latest.get("roe"), suffix="%")
        with c2: metric_card("ROA", latest.get("roa"), suffix="%")
        with c3: metric_card("Net Margin", latest.get("net_margin"), suffix="%")
        with c4: metric_card("Gross Margin", latest.get("gross_margin"), suffix="%")

        st.markdown("**Financial Health**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("DER", latest.get("der"))
        with c2: metric_card("Interest Coverage", latest.get("interest_coverage"))
        with c3: metric_card("Current Ratio", latest.get("current_ratio"))
        with c4: metric_card("OCF / Net Income", latest.get("ocf_to_ni"))

        st.markdown("**Growth (YoY)**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Revenue Growth", latest.get("revenue_growth_yoy"), suffix="%")
        with c2: metric_card("Net Income Growth", latest.get("ni_growth_yoy"), suffix="%")
        with c3: metric_card("Equity Growth", latest.get("equity_growth_yoy"), suffix="%")
        with c4: metric_card("DPS Growth", latest.get("dps_growth_yoy"), suffix="%")

    # TAB 2: FINANCIALS
    with tab2:
        st.subheader("Financial Statements")
        st.caption(f"All figures in {latest.get('report_unit', 'reported unit')} IDR")

        income_cols = {
            "year": "Year", "revenue": "Revenue", "gross_profit": "Gross Profit",
            "ebit": "EBIT", "net_income": "Net Income",
            "depreciation_amortization": "D&A", "interest_expense": "Interest Expense",
            "eps_diluted": "EPS (Diluted)"
        }
        balance_cols = {
            "year": "Year", "cash": "Cash", "accounts_receivable": "Receivables",
            "inventory": "Inventory", "total_current_assets": "Current Assets",
            "total_assets": "Total Assets", "total_current_liabilities": "Current Liabilities",
            "total_debt": "Total Debt", "total_equity": "Total Equity"
        }
        cashflow_cols = {
            "year": "Year", "operating_cash_flow": "Operating CF",
            "capex": "Capex", "fcf": "Free Cash Flow", "dividends_paid": "Dividends Paid"
        }

        def show_table(cols_map):
            available = [c for c in cols_map if c in df.columns]
            t = df[available].rename(columns=cols_map).set_index("Year")
            numeric = t.select_dtypes(include="number").columns
            t[numeric] = t[numeric].round(1)
            st.dataframe(t, use_container_width=True)

        st.markdown("**Income Statement**")
        show_table(income_cols)
        st.markdown("**Balance Sheet**")
        show_table(balance_cols)
        st.markdown("**Cash Flow**")
        show_table(cashflow_cols)

    # TAB 3: RATIOS & TRENDS
    with tab3:
        st.subheader("Ratio Trends Over Time")

        def line_chart(title, col, suffix=""):
            if col in df.columns and df[col].notna().any():
                fig = px.line(df, x="year", y=col, markers=True, title=f"{title} ({suffix})" if suffix else title)
                fig.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        def bar_chart(title, col, suffix=""):
            if col in df.columns and df[col].notna().any():
                fig = px.bar(df, x="year", y=col, title=f"{title} ({suffix})" if suffix else title)
                fig.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1: bar_chart("Revenue", "revenue", latest.get('report_unit',''))
        with c2: bar_chart("Net Income", "net_income", latest.get('report_unit',''))

        c1, c2 = st.columns(2)
        with c1: line_chart("ROE", "roe", "%")
        with c2: line_chart("Net Margin", "net_margin", "%")

        c1, c2 = st.columns(2)
        with c1: line_chart("DER", "der")
        with c2: line_chart("Current Ratio", "current_ratio")

        c1, c2 = st.columns(2)
        with c1: line_chart("Revenue Growth YoY", "revenue_growth_yoy", "%")
        with c2: line_chart("Net Income Growth YoY", "ni_growth_yoy", "%")

        c1, c2 = st.columns(2)
        with c1: line_chart("PER", "per")
        with c2: line_chart("PBV", "pbv")

    # TAB 4: RED FLAGS
    with tab4:
        st.subheader("🚨 Automated Red Flag Detection")
        st.caption("Based on available historical data")

        flags = []
        warnings = []
        green = []

        if len(df) >= 2:
            last = df.iloc[-1]
            prev = df.iloc[-2]

            # Debt growing faster than revenue
            if pd.notna(last.get("total_debt")) and pd.notna(prev.get("total_debt")) and \
               pd.notna(last.get("revenue")) and pd.notna(prev.get("revenue")):
                debt_growth = (last["total_debt"] - prev["total_debt"]) / abs(prev["total_debt"]) * 100
                rev_growth = (last["revenue"] - prev["revenue"]) / abs(prev["revenue"]) * 100
                if debt_growth > rev_growth + 20:
                    flags.append(f"🔴 Debt grew {debt_growth:.1f}% vs revenue growth {rev_growth:.1f}% — debt outpacing revenue")
                else:
                    green.append(f"🟢 Debt growth ({debt_growth:.1f}%) in line with revenue growth ({rev_growth:.1f}%)")

            # Declining net margin
            margins = df["net_margin"].dropna().tolist()
            if len(margins) >= 3 and margins[-1] < margins[-2] < margins[-3]:
                flags.append(f"🔴 Net margin declining 3 years in a row ({margins[-3]:.1f}% → {margins[-2]:.1f}% → {margins[-1]:.1f}%)")
            elif len(margins) >= 2 and margins[-1] < margins[-2]:
                warnings.append(f"🟡 Net margin dropped this year ({margins[-2]:.1f}% → {margins[-1]:.1f}%)")
            elif len(margins) >= 2:
                green.append(f"🟢 Net margin stable or improving ({margins[-2]:.1f}% → {margins[-1]:.1f}%)")

            # Negative FCF while reporting profit
            if pd.notna(last.get("fcf")) and pd.notna(last.get("net_income")):
                if last["fcf"] < 0 and last["net_income"] > 0:
                    flags.append(f"🔴 Negative FCF ({last['fcf']:.0f}) while reporting positive net income — earnings quality concern")
                elif last["fcf"] > 0:
                    green.append(f"🟢 Positive Free Cash Flow ({last['fcf']:.0f})")

            # OCF vs Net Income
            if pd.notna(last.get("ocf_to_ni")):
                if last["ocf_to_ni"] < 0.7:
                    warnings.append(f"🟡 OCF/Net Income ratio is low ({last['ocf_to_ni']:.2f}) — profit may not be fully backed by cash")
                elif last["ocf_to_ni"] >= 1.0:
                    green.append(f"🟢 OCF/Net Income = {last['ocf_to_ni']:.2f} — strong earnings quality")

            # Receivables growing faster than revenue
            if pd.notna(last.get("accounts_receivable")) and pd.notna(prev.get("accounts_receivable")) and \
               pd.notna(last.get("revenue")) and pd.notna(prev.get("revenue")):
                rec_growth = (last["accounts_receivable"] - prev["accounts_receivable"]) / abs(prev["accounts_receivable"]) * 100
                rev_growth = (last["revenue"] - prev["revenue"]) / abs(prev["revenue"]) * 100
                if rec_growth > rev_growth + 15:
                    warnings.append(f"🟡 Receivables growing faster than revenue ({rec_growth:.1f}% vs {rev_growth:.1f}%) — potential collection issue")

            # Dividends exceeding net income
            if pd.notna(last.get("dividends_paid")) and pd.notna(last.get("net_income")):
                if last["dividends_paid"] > 0 and last["net_income"] > 0:
                    if last["dividends_paid"] > last["net_income"]:
                        flags.append(f"🔴 Dividends paid ({last['dividends_paid']:,.0f}) exceeded net income ({last['net_income']:,.0f}) — paying out more than earned")
                    elif last["dividends_paid"] > last["net_income"] * 0.9:
                        warnings.append(f"🟡 Dividends paid ({last['dividends_paid']:,.0f}) is {last['dividends_paid']/last['net_income']*100:.0f}% of net income — very high payout ratio")

            # High DER
            if pd.notna(last.get("der")):
                if last["der"] > 2.0:
                    flags.append(f"🔴 High DER of {last['der']:.2f} — significant leverage risk")
                elif last["der"] > 1.0:
                    warnings.append(f"🟡 DER of {last['der']:.2f} — moderate leverage, monitor closely")
                else:
                    green.append(f"🟢 DER of {last['der']:.2f} — healthy leverage level")

            # Interest coverage
            if pd.notna(last.get("interest_coverage")):
                if last["interest_coverage"] < 2:
                    flags.append(f"🔴 Interest coverage of {last['interest_coverage']:.2f}x — dangerously low, risk of default")
                elif last["interest_coverage"] < 4:
                    warnings.append(f"🟡 Interest coverage of {last['interest_coverage']:.2f}x — adequate but tight")
                else:
                    green.append(f"🟢 Interest coverage of {last['interest_coverage']:.2f}x — comfortable debt service")

        if flags:
            st.markdown("### 🔴 Critical Flags")
            for f in flags:
                st.error(f)
        if warnings:
            st.markdown("### 🟡 Watch Items")
            for w in warnings:
                st.warning(w)
        if green:
            st.markdown("### 🟢 Healthy Signals")
            for g in green:
                st.success(g)
        if not flags and not warnings and not green:
            st.info("Not enough historical data to run red flag detection. Add at least 2 years of data.")

    # TAB 5: DCF VALUATION
    with tab5:
        st.subheader("DCF Intrinsic Value Estimator")
        st.caption("Adjust assumptions below. All calculations are based on latest year FCF.")

        latest_fcf = latest.get("fcf")
        latest_price = latest.get("stock_price")

        if pd.isna(latest_fcf) or latest_fcf is None:
            st.warning("FCF not available. Make sure Operating Cash Flow and Capex are entered for this company.")
        else:
            st.markdown(f"**Base FCF (latest year):** {latest_fcf:,.0f} {latest.get('report_unit', '')}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Your Assumptions**")
                growth_1_5 = st.slider("Growth rate Year 1–5 (%)", 0, 30, 10) / 100
                growth_6_10 = st.slider("Growth rate Year 6–10 (%)", 0, 20, 5) / 100
                terminal_growth = st.slider("Terminal growth rate (%)", 0, 5, 2) / 100
                discount_rate = st.slider("Discount rate / WACC (%)", 5, 20, 10) / 100

            fcf = latest_fcf
            projected_fcfs = []
            for yr in range(1, 11):
                rate = growth_1_5 if yr <= 5 else growth_6_10
                fcf = fcf * (1 + rate)
                pv = fcf / ((1 + discount_rate) ** yr)
                projected_fcfs.append({"Year": f"Year {yr}", "Projected FCF": round(fcf, 1), "Present Value": round(pv, 1)})

            terminal_value = projected_fcfs[-1]["Projected FCF"] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / ((1 + discount_rate) ** 10)
            total_pv = sum(p["Present Value"] for p in projected_fcfs) + pv_terminal

            shares = latest.get("shares_normalized") or (latest.get("shares_outstanding", 1))
            intrinsic_value_per_share = total_pv / shares if shares else None

            with col2:
                st.markdown("**Results**")
                st.metric("Total PV of FCFs", f"{sum(p['Present Value'] for p in projected_fcfs):,.0f}")
                st.metric("PV of Terminal Value", f"{pv_terminal:,.0f}")
                st.metric("Total Intrinsic Value", f"{total_pv:,.0f}")
                if intrinsic_value_per_share:
                    st.metric("Intrinsic Value per Share", f"Rp {intrinsic_value_per_share:,.0f}")
                    if latest_price and not pd.isna(latest_price):
                        margin = (intrinsic_value_per_share - latest_price) / intrinsic_value_per_share * 100
                        if margin > 0:
                            st.success(f"✅ Margin of Safety: {margin:.1f}% — potentially undervalued")
                        else:
                            st.error(f"⚠️ Trading {abs(margin):.1f}% above intrinsic value estimate")

            st.markdown("**FCF Projection Table**")
            proj_df = pd.DataFrame(projected_fcfs)
            st.dataframe(proj_df, use_container_width=True, hide_index=True)

    # TAB 6: DIVIDENDS
    with tab6:
        st.subheader("Dividend Analysis")

        div_cols = ["year", "dividend_per_share", "dividend_yield",
                    "payout_ratio", "fcf_payout_ratio", "dps_growth_yoy"]
        available_div = [c for c in div_cols if c in df.columns]
        div_df = df[available_div].copy()
        has_dividend_data = div_df["dividend_per_share"].notna().any() if "dividend_per_share" in div_df.columns else False

        if not has_dividend_data:
            st.info("No dividend data entered yet for this company.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                latest_yield = latest.get("dividend_yield")
                st.metric("Latest Div Yield", f"{latest_yield:.2f}%" if pd.notna(latest_yield) else "—")
            with c2:
                latest_payout = latest.get("payout_ratio")
                st.metric("Payout Ratio", f"{latest_payout:.1f}%" if pd.notna(latest_payout) else "—")
            with c3:
                latest_fcf_payout = latest.get("fcf_payout_ratio")
                st.metric("FCF Payout Ratio", f"{latest_fcf_payout:.1f}%" if pd.notna(latest_fcf_payout) else "—")
            with c4:
                years_paying = div_df["dividend_per_share"].notna().sum()
                st.metric("Years Paying Dividend", f"{years_paying} yr(s)")

            if "dividend_per_share" in div_df.columns and div_df["dividend_per_share"].notna().any():
                fig = px.bar(div_df.dropna(subset=["dividend_per_share"]),
                             x="year", y="dividend_per_share", title="Dividend Per Share (DPS) History")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            if "dividend_yield" in div_df.columns and div_df["dividend_yield"].notna().any():
                fig = px.line(div_df.dropna(subset=["dividend_yield"]),
                              x="year", y="dividend_yield", markers=True, title="Dividend Yield Over Time (%)")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Sustainability Check**")
            if pd.notna(latest.get("payout_ratio")):
                pr = latest["payout_ratio"]
                if pr > 100:
                    st.error(f"🔴 Payout ratio {pr:.1f}% — paying more than it earns, unsustainable")
                elif pr > 75:
                    st.warning(f"🟡 Payout ratio {pr:.1f}% — high, leaves little room for reinvestment")
                else:
                    st.success(f"🟢 Payout ratio {pr:.1f}% — sustainable dividend level")

            rename_map = {
                "year": "Year", "dividend_per_share": "DPS",
                "dividend_yield": "Yield %", "payout_ratio": "Payout %",
                "fcf_payout_ratio": "FCF Payout %", "dps_growth_yoy": "DPS Growth %"
            }
            display_div = div_df.rename(columns=rename_map)
            numeric = display_div.select_dtypes(include="number").columns
            display_div[numeric] = display_div[numeric].round(2)
            st.dataframe(display_div, use_container_width=True, hide_index=True)


# ============================================================
# DATA INPUT CENTER
# ============================================================
def show_data_input():
    st.title("➕ Data Input Center")
    st.caption("Add or edit companies, financial statements, and stock prices")

    input_tab1, input_tab2, input_tab3 = st.tabs([
        "🏢 Companies", "📊 Financial Statements", "💹 Stock Prices"
    ])

    # ──────────────────────────────────────────
    # TAB 1: COMPANIES
    # ──────────────────────────────────────────
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
                shares = st.number_input("Shares Outstanding", min_value=0, value=0, step=1000000,
                                          help="Total number of shares")
                report_unit = st.selectbox("Report Unit", ["millions", "billions", "thousands"])

                submitted = st.form_submit_button("➕ Add Company", use_container_width=True)
                if submitted:
                    if not ticker or not name or not sector:
                        st.error("Ticker, Name, and Sector are required.")
                    else:
                        existing = companies_df[companies_df["ticker"] == ticker] if not companies_df.empty else pd.DataFrame()
                        if not existing.empty:
                            st.error(f"Company {ticker} already exists. Use the edit section.")
                        else:
                            try:
                                supabase.table("companies").insert({
                                    "ticker": ticker,
                                    "name": name,
                                    "sector": sector,
                                    "sub_sector": sub_sector if sub_sector else None,
                                    "shares_outstanding": int(shares) if shares > 0 else None,
                                    "report_unit": report_unit
                                }).execute()
                                clear_cache()
                                st.success(f"✅ {ticker} added successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

        with col_edit:
            st.subheader("Edit Existing Company")
            if companies_df.empty:
                st.info("No companies yet.")
            else:
                edit_ticker = st.selectbox("Select company to edit", companies_df["ticker"].tolist(), key="edit_company_select")
                company_row = companies_df[companies_df["ticker"] == edit_ticker].iloc[0]

                with st.form("edit_company_form"):
                    e_name = st.text_input("Company Name", value=str(company_row.get("name", "")))
                    e_sector = st.text_input("Sector", value=str(company_row.get("sector", "")))
                    e_sub_sector = st.text_input("Sub Sector", value=str(company_row.get("sub_sector", "") or ""))
                    e_shares = st.number_input("Shares Outstanding",
                                               value=int(company_row.get("shares_outstanding") or 0),
                                               min_value=0, step=1000000)
                    e_unit = st.selectbox("Report Unit", ["millions", "billions", "thousands"],
                                          index=["millions", "billions", "thousands"].index(
                                              company_row.get("report_unit", "millions")))

                    save = st.form_submit_button("💾 Save Changes", use_container_width=True)
                    if save:
                        try:
                            company_id = company_row["id"]
                            supabase.table("companies").update({
                                "name": e_name,
                                "sector": e_sector,
                                "sub_sector": e_sub_sector if e_sub_sector else None,
                                "shares_outstanding": int(e_shares) if e_shares > 0 else None,
                                "report_unit": e_unit
                            }).eq("id", company_id).execute()
                            clear_cache()
                            st.success(f"✅ {edit_ticker} updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.divider()
        st.subheader("All Companies")
        if not companies_df.empty:
            st.dataframe(companies_df[["ticker", "name", "sector", "sub_sector", "shares_outstanding", "report_unit"]],
                         use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────
    # TAB 2: FINANCIAL STATEMENTS
    # ──────────────────────────────────────────
    with input_tab2:
        companies_df = load_companies()
        financials_df = load_financials()

        if companies_df.empty:
            st.warning("Add a company first before entering financial data.")
        else:
            ticker_options = companies_df["ticker"].tolist()

            # Select mode
            mode = st.radio("Mode", ["Add New", "Edit Existing"], horizontal=True)

            if mode == "Edit Existing":
                if financials_df.empty:
                    st.info("No financial statements yet.")
                else:
                    # Merge ticker info
                    fin_with_ticker = financials_df.merge(
                        companies_df[["id", "ticker"]], left_on="company_id", right_on="id", how="left"
                    )
                    fin_options = fin_with_ticker.apply(
                        lambda r: f"{r['ticker']} — {int(r['year'])} ({r['period']})", axis=1
                    ).tolist()
                    selected_fin = st.selectbox("Select record to edit", fin_options)
                    selected_idx = fin_options.index(selected_fin)
                    fin_row = fin_with_ticker.iloc[selected_idx]
                    fin_id = financials_df.iloc[selected_idx]["id"]

                    def v(col):
                        val = fin_row.get(col)
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            return 0.0
                        return float(val)

                    st.markdown("---")
                    st.subheader(f"Editing: {fin_row['ticker']} FY{int(fin_row['year'])}")

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
                        with c1:
                            e_ocf = st.number_input("Operating Cash Flow", value=v("operating_cash_flow"), format="%.0f")
                        with c2:
                            e_capex = st.number_input("Capex", value=v("capex"), format="%.0f")
                        with c3:
                            e_div = st.number_input("Dividends Paid", value=v("dividends_paid"), format="%.0f")

                        save_fin = st.form_submit_button("💾 Save Changes", use_container_width=True)
                        if save_fin:
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
                                st.success("✅ Financial statement updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

            else:  # Add New
                st.markdown("---")
                with st.form("add_fin_form"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        sel_ticker = st.selectbox("Company *", ticker_options)
                        sel_year = st.number_input("Year *", min_value=2000, max_value=2030, value=2023, step=1)
                        sel_period = st.selectbox("Period", ["annual", "Q1", "Q2", "Q3", "Q4"])

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
                    with c1:
                        n_ocf = st.number_input("Operating Cash Flow", value=0.0, format="%.0f")
                    with c2:
                        n_capex = st.number_input("Capex", value=0.0, format="%.0f")
                    with c3:
                        n_div = st.number_input("Dividends Paid", value=0.0, format="%.0f")

                    add_fin = st.form_submit_button("➕ Add Financial Statement", use_container_width=True)
                    if add_fin:
                        company_row = companies_df[companies_df["ticker"] == sel_ticker].iloc[0]
                        company_id = company_row["id"]
                        try:
                            supabase.table("financial_statements").insert({
                                "company_id": company_id,
                                "year": int(sel_year),
                                "period": sel_period,
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

    # ──────────────────────────────────────────
    # TAB 3: STOCK PRICES
    # ──────────────────────────────────────────
    with input_tab3:
        companies_df = load_companies()
        market_df = load_market_data()

        col_add, col_edit = st.columns(2)

        with col_add:
            st.subheader("Add / Update Stock Price")
            if companies_df.empty:
                st.info("Add a company first.")
            else:
                with st.form("add_price_form"):
                    p_ticker = st.selectbox("Company", companies_df["ticker"].tolist())
                    p_date = st.date_input("Date")
                    p_price = st.number_input("Stock Price (IDR)", min_value=0.0, value=0.0, format="%.0f")

                    add_price = st.form_submit_button("➕ Add Price", use_container_width=True)
                    if add_price:
                        if p_price <= 0:
                            st.error("Price must be greater than 0.")
                        else:
                            company_id = companies_df[companies_df["ticker"] == p_ticker].iloc[0]["id"]
                            try:
                                supabase.table("market_data").upsert({
                                    "company_id": company_id,
                                    "date": str(p_date),
                                    "stock_price": float(p_price)
                                }, on_conflict="company_id,date").execute()
                                clear_cache()
                                st.success(f"✅ Price for {p_ticker} on {p_date} saved!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

        with col_edit:
            st.subheader("Existing Prices")
            if market_df.empty:
                st.info("No stock prices yet.")
            else:
                prices_with_ticker = market_df.merge(
                    companies_df[["id", "ticker"]], left_on="company_id", right_on="id", how="left"
                )
                display_prices = prices_with_ticker[["ticker", "date", "stock_price"]].sort_values(
                    ["ticker", "date"], ascending=[True, False]
                )
                st.dataframe(display_prices, use_container_width=True, hide_index=True)


# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == "screener":
    show_screener()
elif st.session_state.page == "company" and st.session_state.selected_ticker:
    show_company(st.session_state.selected_ticker)
elif st.session_state.page == "data_input":
    show_data_input()
else:
    st.session_state.page = "screener"
    show_screener()