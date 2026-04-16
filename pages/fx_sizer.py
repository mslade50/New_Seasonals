import streamlit as st

def main():
    st.set_page_config(page_title="FX Position Sizer", layout="centered")
    st.title("FX Position Sizer")

    col1, col2 = st.columns(2)

    with col1:
        pair = st.text_input("Pair", value="EURUSD").upper()
        entry = st.number_input("Entry Price", value=1.1807, format="%.5f")
        atr_pct = st.number_input("ATR (%)", value=0.63, format="%.2f")

    with col2:
        direction = st.selectbox("Direction", ["Short", "Long"])
        atr_mult = st.number_input("Stop (ATR multiple)", value=2.0, format="%.1f")
        risk_usd = st.number_input("Risk ($USD)", value=2500.0, format="%.0f")

    # If quote currency is not USD, user provides conversion rate
    quote_ccy = pair[-3:] if len(pair) >= 6 else "USD"
    needs_conversion = quote_ccy != "USD"

    usdx_rate = 1.0
    if needs_conversion:
        usdx_rate = st.number_input(
            f"USD{quote_ccy} rate (how many {quote_ccy} per 1 USD)",
            value=0.7814, format="%.5f"
        )

    if st.button("Calculate", type="primary", use_container_width=True):
        stop_dist = entry * (atr_pct / 100.0) * atr_mult

        if needs_conversion:
            risk_quote = risk_usd * usdx_rate
        else:
            risk_quote = risk_usd

        base_ccy = pair[:3] if len(pair) >= 6 else "EUR"
        position = risk_quote / stop_dist

        st.markdown("---")
        st.markdown(f"### Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Stop Distance", f"{stop_dist:.5f}")
        c2.metric(f"Risk ({quote_ccy})", f"{risk_quote:,.0f}")
        c3.metric(f"Position ({base_ccy})", f"{position:,.0f}")

        stop_price = entry - stop_dist if direction == "Long" else entry + stop_dist
        st.caption(
            f"{direction} **{position:,.0f} {base_ccy}** at {entry:.5f} | "
            f"Stop: {stop_price:.5f} ({atr_mult:.1f}x ATR) | "
            f"Risk: ${risk_usd:,.0f}"
        )

if __name__ == "__main__":
    main()
