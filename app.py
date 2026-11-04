import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

st.set_page_config(page_title="Marketing Campaign Dashboard", page_icon="🛒", layout="wide")

CAMPAIGNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
LABELS = ["Cmp1", "Cmp2", "Cmp3", "Cmp4", "Cmp5", "Cmp6"]
MNT = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
TIER_ORDER = ["Low", "Medium", "High"]
FATIGUE_ORDER = ["1-Engaged", "2-Low fatigue", "3-High fatigue (dormant)", "4-Severe (never responded)"]


# ---------------------------------------------------------------- data prep
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["Dt"] = pd.to_datetime(df["Dt_Customer"], format="%d/%m/%Y")
    df["Age"] = 2014 - df["Year_Birth"]
    # remove impossible ages / extreme income outlier, fill missing income
    df = df[(df["Age"] <= 90) & ((df["Income"] < 200000) | df["Income"].isna())].copy()
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Kids"] = df["Kidhome"] + df["Teenhome"]

    # RFM + CLTV proxy
    df["Monetary"] = df[MNT].sum(axis=1)
    df["Freq"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    ref = df["Dt"].max()
    df["TenureM"] = ((ref - df["Dt"]).dt.days / 30.4).clip(lower=1)
    df["Spend_pm"] = df["Monetary"] / df["TenureM"]
    df["Retention"] = (1 - df["Recency"] / 100).clip(0.05, 1)
    df["CLTV"] = df["Spend_pm"] * 36 * df["Retention"]
    df["CLTV_Tier"] = pd.qcut(df["CLTV"].rank(method="first"), [0, 1 / 3, 2 / 3, 1], labels=TIER_ORDER)

    # fatigue from Cmp1-5 only (Cmp6 is kept aside to validate)
    df["Prior"] = df[CAMPAIGNS[:5]].sum(axis=1)
    df["Fatigue"] = np.select(
        [
            df["Prior"] == 0,
            df["Prior"] >= 3,
            (df["AcceptedCmp4"] + df["AcceptedCmp5"]) > 0,
        ],
        [FATIGUE_ORDER[3], FATIGUE_ORDER[0], FATIGUE_ORDER[1]],
        default=FATIGUE_ORDER[2],
    )
    return df.reset_index(drop=True)


@st.cache_data
def score_model(df):
    feats = ["Income", "Age", "Kidhome", "Teenhome", "Recency", "TenureM", *MNT,
             "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
             "NumStorePurchases", "NumWebVisitsMonth", "Prior", "Monetary", "Freq"]
    X, y = df[feats], df["Response"]
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=1)
    p = cross_val_predict(gb, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    imp = pd.Series(gb.fit(X, y).feature_importances_, feats).sort_values(ascending=False)
    return p, auc, imp


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛒 Campaign Analytics")
uploaded = st.sidebar.file_uploader("Upload a different CSV (optional)", type="csv")
df = load_data(uploaded if uploaded else "data/marketing_campaign.csv")
df["Propensity"], auc, importance = score_model(df)

st.sidebar.markdown("### Economics")
cost = st.sidebar.number_input("Cost per contact", value=3.0, step=0.5)
revenue = st.sidebar.number_input("Revenue per acceptance", value=11.0, step=0.5)
breakeven = cost / revenue
st.sidebar.caption(f"Break-even response rate: **{breakeven:.1%}**")

page = st.sidebar.radio(
    "Section",
    ["1. Campaign performance", "2. RFM / CLTV segments", "3. Fatigue levels", "4. Targeting strategy"],
)
st.sidebar.caption(f"{len(df):,} customers after cleaning")

st.title("Marketing Campaign Response Dashboard")

# ---------------------------------------------------------------- 1. performance
if page.startswith("1"):
    st.subheader("How are the six campaigns performing?")
    n = len(df)
    perf = pd.DataFrame({
        "Campaign": LABELS,
        "Acceptances": [int(df[c].sum()) for c in CAMPAIGNS],
    })
    perf["Acceptance rate"] = perf["Acceptances"] / n
    perf["Cost"] = n * cost
    perf["Revenue"] = perf["Acceptances"] * revenue
    perf["Net return"] = perf["Revenue"] - perf["Cost"]
    perf["ROI"] = perf["Net return"] / perf["Cost"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{n:,}")
    c2.metric("Latest campaign (Cmp6) rate", f"{df['Response'].mean():.1%}")
    c3.metric("Ever accepted any campaign", f"{(df[CAMPAIGNS].sum(axis=1) > 0).mean():.1%}")
    c4.metric("Break-even rate", f"{breakeven:.1%}")

    left, right = st.columns(2)
    fig = px.bar(perf, x="Campaign", y="Acceptance rate", text=perf["Acceptance rate"].map("{:.1%}".format),
                 title="Acceptance rate by campaign (blanket send)")
    fig.add_hline(y=breakeven, line_dash="dash", line_color="red",
                  annotation_text=f"Break-even {breakeven:.1%}")
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(fig, width="stretch")
    fig2 = px.bar(perf, x="Campaign", y="ROI", text=perf["ROI"].map("{:.0%}".format),
                  title="ROI on contact cost")
    fig2.update_yaxes(tickformat=".0%")
    right.plotly_chart(fig2, width="stretch")

    st.dataframe(
        perf.style.format({"Acceptance rate": "{:.1%}", "Cost": "{:,.0f}", "Revenue": "{:,.0f}",
                           "Net return": "{:,.0f}", "ROI": "{:.0%}"}),
        width="stretch", hide_index=True)

    st.markdown("#### Repeat responders drive the results")
    a = df[df["Prior"] > 0]["Response"].mean()
    b = df[df["Prior"] == 0]["Response"].mean()
    d1, d2 = st.columns(2)
    d1.metric("Cmp6 rate: accepted any of Cmp1-5", f"{a:.1%}", f"{(a - breakeven):+.1%} vs break-even")
    d2.metric("Cmp6 rate: never accepted before", f"{b:.1%}", f"{(b - breakeven):+.1%} vs break-even")
    by_prior = df.groupby("Prior")["Response"].agg(["size", "mean"]).reset_index()
    by_prior.columns = ["Past acceptances (Cmp1-5)", "Customers", "Cmp6 response rate"]
    fig3 = px.bar(by_prior, x="Past acceptances (Cmp1-5)", y="Cmp6 response rate",
                  text=by_prior["Cmp6 response rate"].map("{:.0%}".format))
    fig3.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------- 2. RFM / CLTV
elif page.startswith("2"):
    st.subheader("RFM-based CLTV segments (High / Medium / Low)")
    st.caption("CLTV proxy = monthly spend rate x 36 months x recency-based retention weight, split into three equal tiers. "
               "It is a simple proxy, not a probabilistic model.")
    g = df.groupby("CLTV_Tier", observed=True).agg(
        Customers=("ID", "size"), Avg_spend=("Monetary", "mean"), Avg_frequency=("Freq", "mean"),
        Avg_recency_days=("Recency", "mean"), Median_income=("Income", "median"),
        Web_visits_month=("NumWebVisitsMonth", "mean"), Catalog_purchases=("NumCatalogPurchases", "mean"),
        Kids_at_home=("Kids", "mean"), Cmp6_rate=("Response", "mean"),
        Revenue=("Monetary", "sum"), CLTV_total=("CLTV", "sum")).reindex(TIER_ORDER).reset_index()
    g["Revenue share"] = g["Revenue"] / g["Revenue"].sum()
    g["CLTV share"] = g["CLTV_total"] / g["CLTV_total"].sum()

    cols = st.columns(3)
    for col, (_, r) in zip(cols, g.iterrows()):
        col.metric(f"{r['CLTV_Tier']} CLTV", f"{int(r['Customers']):,} customers",
                   f"{r['Revenue share']:.0%} of revenue | {r['CLTV share']:.0%} of CLTV", delta_color="off")

    left, right = st.columns(2)
    fig = px.pie(g, names="CLTV_Tier", values="Customers", category_orders={"CLTV_Tier": TIER_ORDER},
                 title="Customers by tier", hole=0.45)
    left.plotly_chart(fig, width="stretch")
    share = g.melt(id_vars="CLTV_Tier", value_vars=["Revenue share", "CLTV share"], var_name="Metric", value_name="Share")
    fig2 = px.bar(share, x="CLTV_Tier", y="Share", color="Metric", barmode="group",
                  category_orders={"CLTV_Tier": TIER_ORDER}, title="Value concentration")
    fig2.update_yaxes(tickformat=".0%")
    right.plotly_chart(fig2, width="stretch")

    st.markdown("#### Tier profile")
    show = g.drop(columns=["Revenue", "CLTV_total"])
    st.dataframe(show.style.format({
        "Avg_spend": "{:,.0f}", "Avg_frequency": "{:.1f}", "Avg_recency_days": "{:.0f}",
        "Median_income": "{:,.0f}", "Web_visits_month": "{:.1f}", "Catalog_purchases": "{:.1f}",
        "Kids_at_home": "{:.2f}", "Cmp6_rate": "{:.1%}", "Revenue share": "{:.0%}", "CLTV share": "{:.0%}"}),
        width="stretch", hide_index=True)

    st.markdown("#### Explore the RFM view")
    fig3 = px.scatter(df, x="Recency", y="Monetary", color="CLTV_Tier", size="Freq", opacity=0.6,
                      category_orders={"CLTV_Tier": TIER_ORDER},
                      hover_data=["ID", "Income", "Freq"], title="Recency vs spend (bubble size = purchase frequency)")
    st.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------- 3. fatigue
elif page.startswith("3"):
    st.subheader("Customer fatigue levels")
    st.caption("Fatigue is defined only from Cmp1-Cmp5 history; Cmp6 is used afterwards to test whether the levels make sense.")
    f = df.groupby("Fatigue").agg(Customers=("ID", "size"), Cmp6_rate=("Response", "mean"),
                                  Cmp6_conversions=("Response", "sum"), Avg_spend=("Monetary", "mean"),
                                  Median_income=("Income", "median")).reindex(FATIGUE_ORDER).reset_index()
    f["Share of base"] = f["Customers"] / f["Customers"].sum()
    f["Net if all contacted"] = f["Customers"] * (f["Cmp6_rate"] * revenue - cost)

    cols = st.columns(4)
    for col, (_, r) in zip(cols, f.iterrows()):
        col.metric(r["Fatigue"], f"{int(r['Customers']):,} ({r['Share of base']:.0%})",
                   f"Cmp6 rate {r['Cmp6_rate']:.0%}", delta_color="off")

    left, right = st.columns(2)
    fig = px.bar(f, x="Fatigue", y="Cmp6_rate", text=f["Cmp6_rate"].map("{:.0%}".format),
                 title="Cmp6 response by fatigue level")
    fig.add_hline(y=breakeven, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(fig, width="stretch")
    fig2 = px.bar(f, x="Fatigue", y="Net if all contacted", text=f["Net if all contacted"].map("{:,.0f}".format),
                  title="Net return if the whole level is contacted")
    right.plotly_chart(fig2, width="stretch")

    st.dataframe(f.style.format({
        "Cmp6_rate": "{:.1%}", "Avg_spend": "{:,.0f}", "Median_income": "{:,.0f}",
        "Share of base": "{:.0%}", "Net if all contacted": "{:,.0f}"}), width="stretch", hide_index=True)

    st.markdown("#### Fatigue x CLTV tier (customers)")
    ct = pd.crosstab(df["Fatigue"], df["CLTV_Tier"]).reindex(FATIGUE_ORDER)[TIER_ORDER]
    fig3 = px.imshow(ct, text_auto=True, color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(fig3, width="stretch")
    sev = f[f["Fatigue"] == FATIGUE_ORDER[3]].iloc[0]
    st.info(
        f"The severe-fatigue group is {sev['Share of base']:.0%} of customers and still produced "
        f"{int(sev['Cmp6_conversions'])} of {int(df['Response'].sum())} Cmp6 acceptances, but contacting all of them "
        f"is loss-making. They need rested contact, a different channel, or model-based selection - not more volume.")

# ---------------------------------------------------------------- 4. strategy
else:
    st.subheader("Targeting strategy: who should we contact?")
    st.caption(f"Propensity model: gradient boosting, 5-fold cross-validated, AUC = {auc:.2f}. "
               "Scores are out-of-fold so they are not simply memorising the answers.")

    d = df.sort_values("Propensity", ascending=False).reset_index(drop=True)
    d["Cum_profit"] = (d["Response"] * revenue - cost).cumsum()
    d["Cum_conversions"] = d["Response"].cumsum()
    best = int(d["Cum_profit"].idxmax()) + 1
    pct = st.slider("Contact the top X% of customers by propensity", 5, 100, int(round(best / len(d) * 100)), 5)
    k = int(len(d) * pct / 100)
    sel = d.iloc[:k]
    blanket = df["Response"].sum() * revenue - len(df) * cost

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers contacted", f"{k:,}")
    c2.metric("Response rate in targeted group", f"{sel['Response'].mean():.1%}")
    c3.metric("Acceptors captured", f"{sel['Response'].sum() / df['Response'].sum():.0%}")
    c4.metric("Net return", f"{d['Cum_profit'].iloc[k - 1]:,.0f}", f"{d['Cum_profit'].iloc[k - 1] - blanket:+,.0f} vs blanket send")

    curve = pd.DataFrame({"Customers contacted": np.arange(1, len(d) + 1), "Net return": d["Cum_profit"]})
    fig = px.line(curve, x="Customers contacted", y="Net return", title="Cumulative net return as targeting widens")
    fig.add_vline(x=k, line_dash="dot", annotation_text=f"Selected: {pct}%")
    fig.add_hline(y=0, line_color="grey")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    d["Decile"] = (pd.qcut(d["Propensity"].rank(method="first", ascending=False), 10, labels=False) + 1)
    dec = d.groupby("Decile").agg(Customers=("ID", "size"), Rate=("Response", "mean")).reset_index()
    fig2 = px.bar(dec, x="Decile", y="Rate", text=dec["Rate"].map("{:.0%}".format),
                  title="Actual Cmp6 response by propensity decile (1 = highest)")
    fig2.update_yaxes(tickformat=".0%")
    left.plotly_chart(fig2, width="stretch")
    imp = importance.head(8).sort_values().reset_index()
    imp.columns = ["Feature", "Importance"]
    fig3 = px.bar(imp, x="Importance", y="Feature", orientation="h", title="What drives response")
    right.plotly_chart(fig3, width="stretch")

    st.markdown("#### Recommended playbook")
    st.markdown(
        "1. **Always contact** engaged, low-fatigue and dormant (high-fatigue) responders.\n"
        "2. **Win back dormant responders** with a personalised message rather than a generic discount.\n"
        "3. **Rest the severe-fatigue majority**; only add customers the model scores highly.\n"
        "4. **Switch channel for silent High-CLTV customers**: they buy in store and by catalog, so test those.\n"
        "5. **Match the offer to value tier**: premium/category offers for High, bundles for Medium, deals for Low.\n"
        "6. **Judge every campaign on profit vs the break-even rate**, always with a holdout group."
    )

    st.markdown("#### Download target list")
    out = sel[["ID", "CLTV_Tier", "Fatigue", "Propensity", "Income", "Recency", "Monetary"]].copy()
    out["Propensity"] = out["Propensity"].round(3)
    st.dataframe(out.head(50), width="stretch", hide_index=True)
    st.download_button("Download full target list (CSV)", out.to_csv(index=False).encode(),
                       file_name="target_customers.csv", mime="text/csv")
