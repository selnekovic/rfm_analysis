# RFM Analysis Streamlit App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rfm-rfe-customer-analysis.streamlit.app/)

Interactive Streamlit app for **RFM/RFE user segmentation** from transaction or engagement data.

The app calculates per-user:
- **Recency** (days since last event)
- **Frequency** (number of events)
- **Monetary / Engagement** (sum of value)

It then scores users into quantiles, maps them into business-friendly segments, and provides visual + tabular exploration with CSV export.

## Live App

Deployed Streamlit app: [rfm-rfe-customer-analysis.streamlit.app](https://rfm-rfe-customer-analysis.streamlit.app/)

## Features

- Upload your own CSV or use the bundled `sample_data.csv`
- Flexible column mapping (`user_id`, `date`, `value`) from arbitrary input schemas
- Automatic validation and type parsing:
  - `date`: accepts `YYYY-MM-DD`, `YYYYMMDD`, integer `YYYYMMDD`, or datetime
  - `value`: numeric or numeric-like strings
- Optional outlier removal using percentile clipping (1st-99th percentile on `value`)
- RFM scoring (1-5 for each dimension) using quintiles
- Segment treemap visualization and detailed segment table
- Segment-based CSV download
- Session-aware caching for faster reruns

## Tech Stack

- [Streamlit](https://streamlit.io/) - UI and app runtime
- [Polars](https://pola.rs/) - data processing
- [Pandas](https://pandas.pydata.org/) - display/export convenience
- [Matplotlib](https://matplotlib.org/) + [squarify](https://github.com/laserson/squarify) - treemap chart

## Data Requirements

The app expects three logical fields (after mapping in sidebar):

- `user_id`: unique user/customer identifier
- `date`: transaction/event date
- `value`: transaction amount or engagement metric

Example:

```csv
user_id,date,value
5XPTFY8V,2025-10-07,38.26
5XPTFY8V,2025-10-27,17.69
0NU6IEG4,2025-10-24,97.33
```

## License

This project is licensed under the [MIT License](./LICENSE).
