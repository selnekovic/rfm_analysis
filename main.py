import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from _import import sidebar_import
from _helpers import *
from _visuals import *

# ---- app setup ----
st.set_page_config(page_title="RFM / RFE analysis", layout="wide")
st.session_state.setdefault("rf_variant", "Recency Frequency Monetary (e.g. revenue)")
st.session_state.setdefault("remove_outliers", False)
st.session_state.setdefault("data_mode", "sample")
st.session_state.setdefault("data_version", 0)  # bumps when a new dataset/mapping is imported
sample_data = "sample_data.csv"

# ---- small cached helpers ----
@st.cache_data(show_spinner=False)
def _to_csv_bytes(df_pd: pd.DataFrame) -> bytes:
    return df_pd.to_csv(index=False).encode("utf-8")

# memoize pandas apply for mapping? (kept as-is; usually fine)
# you could later vectorize `map_user_segment` if needed

# ---- global styles ----
st.markdown(global_styles, unsafe_allow_html=True)

# ---- header ----
st.markdown("<div class='app-title'>RFM / RFE Analysis</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='app-subtitle'> 
        version 0.9 | <a href='https://selnekovic.com' target='_blank' style='color:#2563eb; text-decoration:none; font-weight:500;'>
            selnekovic.com
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- load & process (cached via session_state) ----
# sidebar_import now sets st.session_state["raw_df"] and bumps ["data_version"] when changed
raw_df: pl.DataFrame = sidebar_import(sample_data)  # prepared columns: user_id, date, value

# build a pipeline key to reuse RFM results across reruns
pipeline_key = (
    st.session_state.get("data_version", 0),
    st.session_state.get("rf_variant"),
    st.session_state.get("remove_outliers"),
)

if st.session_state.get("pipeline_key") != pipeline_key:
    df = raw_df
    if st.session_state["remove_outliers"] is True:
        df = remove_outliers_percentile(df)  # polars in, polars out (from _helpers)
    df_transformed = rfm_transformation(df)     # polars
    rfm_polars = rfm_scoring(df_transformed)    # polars
    rfm = rfm_polars.to_pandas()                # pandas for UI and .apply
    st.session_state["rfm"] = rfm
    st.session_state["pipeline_key"] = pipeline_key
else:
    rfm = st.session_state["rfm"]

# ---- about rfm / rfe ----
with st.expander("About RFM / RFE Analysis", expanded=False):
    st.markdown("""
    RFM/RFE analysis is a powerful method for segmenting users based on their behavior. It helps you understand and categorize your user base by answering three key questions:
    
    - **R (Recency)**: How recently did the user make a purchase or visit?
    - **F (Frequency)**: How often do they buy or visit?
    - **M (Monetary) / E (Engagement)**: How much do they spend (Monetary), or how actively do they interact with your platform (Engagement)?
    
    By scoring each dimension (typically on a scale from 1 to 5), you can group users into meaningful segments such as “Champions,” “Active,” “At Risk,” or “Lost.” This segmentation allows you to identify your most valuable users, recognize those at risk of churning, and tailor strategies for improved growth and retention.
    
    ---

    #### What You'll Need for the Analysis
    
    > To perform this analysis, you need a dataset containing transactional or event data. At a minimum, your data must include three specific columns, which you can map in the sidebar:
    > 1.  **User ID**: A unique identifier for each user or customer.
    > 2.  **Date**: The date of the transaction or event (e.g. 2025-11-15).
    > 3.  **Value**: The monetary amount of a transaction or an engagement score (e.g. page views).
    """)

# ---- overview ----
st.subheader("Data Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", f"{rfm.shape[0]:,}")
c2.metric("AVG Recency (days)", f"{rfm['recency'].mean():.1f}")
c3.metric("AVG Frequency", f"{rfm['frequency'].mean():.2f}")
c4.metric("AVG Monetary", f"{rfm['monetary'].mean():.2f}")

# ---- segmentation ----
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Segment Distribution")
rfm["segment"] = rfm.apply(map_user_segment, axis=1)

# aggregated counts for the treemap
seg_counts = (
    rfm.groupby("segment", as_index=False)
    .size()
    .sort_values("size", ascending=False)
)
total_users = seg_counts["size"].sum()
seg_counts["percentage"] = (seg_counts["size"] / total_users) * 100

tab1, tab2, tab3 = st.tabs(["treemap", "table", "segments explanation"])

with tab1:
    fig = create_squarify_treemap(seg_counts)
    st.pyplot(fig, clear_figure=True)

with tab2:
    rfm_table = rfm.drop(columns=["r_score", "f_score", "m_score"], errors="ignore")

    view_segments = ["all"] + sorted(rfm_table["segment"].astype("string").unique())
    view_col1, _ = st.columns([2, 1])
    with view_col1:
        view_segment = st.selectbox("choose a segment to view", options=view_segments, index=0)

    if view_segment == "all":
        df_view = rfm_table
    else:
        df_view = rfm_table[rfm_table["segment"] == view_segment]

    st.dataframe(
        df_view,
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.markdown("""
    ### High Recency
    *These customers bought recently and are therefore still engaged. The naming distinguishes their value level based on frequency and monetary score.*

    - **Champions**: The best customers. They buy often, spend a lot, and purchased recently. They're the brand's most valuable and loyal segment—ideal for retention and advocacy.
    - **Active**: Buyers who are still recent and fairly regular, but not top-tier in spend or frequency. They maintain healthy engagement and can be nurtured into champions.
    - **Newcomers**: Recent buyers with low frequency or low spend. They've just started their journey with the brand and need encouragement to purchase again.

    ---

    ### Medium Recency
    *These customers haven't purchased as recently. They may be drifting away or showing reduced engagement.*

    - **Fading Loyalists**: Once-frequent, high-value buyers whose activity has slowed. They're at risk of churn but have strong past loyalty.
    - **Inactive**: Customers who used to buy occasionally and are now dropping off. They represent moderate past value but require reactivation.
    - **At Risk (Low Value)**: Low-value buyers who haven't purchased in a while. Limited potential, but worth low-effort re-engagement attempts.

    ---

    ### Low Recency
    *These customers haven't bought for a long time. Segmentation here differentiates their historical importance.*

    - **Can't Lose Them**: Previously high-value customers who are now inactive. Their past contribution makes them worth targeted win-back efforts.
    - **Reactivation Pool**: Medium-value customers who are inactive but not yet lost. A good audience for re-engagement campaigns or tailored offers.
    - **Lost Casual**: Low-value, low-frequency customers who haven't returned. They're effectively churned and typically not worth active marketing investment.
    """)

# drop helper scores from the downloadable view (we keep the working copy above)
rfm = rfm.drop(columns=["r_score", "f_score", "m_score"])

# ---- download section (cached bytes; no upstream recompute) ----
st.divider()
st.subheader("Download Data")

main_col, _ = st.columns([2, 2])
with main_col:
    segments_sorted = ["all"] + sorted(rfm["segment"].astype("string").unique())
    dl_c1, dl_c2 = st.columns([2, 1])

    with dl_c1:
        selected_segment = st.selectbox("choose a segment to download", options=segments_sorted, index=0)
    with dl_c2:
        file_base = st.text_input("file name (without .csv)", value="rfm_export")

    if selected_segment == "all":
        df_to_download = rfm
        suffix = "all"
    else:
        df_to_download = rfm[rfm["segment"] == selected_segment]
        suffix = selected_segment.lower().replace(" ", "_")

    # memoize per (pipeline_key, suffix)
    csv_cache_key = (st.session_state["pipeline_key"], suffix)
    if "csv_cache" not in st.session_state:
        st.session_state["csv_cache"] = {}

    if csv_cache_key not in st.session_state["csv_cache"]:
        st.session_state["csv_cache"][csv_cache_key] = _to_csv_bytes(df_to_download)

    csv_bytes = st.session_state["csv_cache"][csv_cache_key]

    st.download_button(
        label="Download",
        data=csv_bytes,
        file_name=f"{file_base}_{suffix}.csv",
        mime="text/csv",
        use_container_width=True
    )
