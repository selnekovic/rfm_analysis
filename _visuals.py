import matplotlib
import matplotlib.pyplot as plt
import squarify
import pandas as pd
import numpy as np

def create_squarify_treemap(seg_counts: pd.DataFrame) -> plt.Figure:
    """
    Generates a squarify treemap visualization of segment distribution.

    Parameters
    ----------
    seg_counts : pd.DataFrame
        DataFrame with 'segment', 'size', and 'percentage' columns.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object for the treemap.
    """
    sizes = seg_counts["size"].values
    labels = [
        f"{row['segment']}\n{row['size']:,} users\n({row['percentage']:.2f}%)"
        for _, row in seg_counts.iterrows()
    ]
    # Use the predefined color map, with a fallback color for any unexpected segments
    colors = [
        SEGMENT_COLOR_MAP.get(segment, "#9ca3af") for segment in seg_counts["segment"]
    ]

    fig = plt.figure(figsize=(15, 9))
    squarify.plot(
        sizes=sizes,
        label=labels,
        alpha=0.8,
        color=colors,
        text_kwargs={"fontsize": 10},
        bar_kwargs={"edgecolor": "white", "linewidth": 3},
    )
    plt.axis("off")

    return fig

# ---- global styles ----
global_styles = """
    <style>
        /* app frame */
        .stMainBlockContainer {
            max-width:80%;
        }

        /* page title */
        .app-title {
            text-align: center;
            font-size: 2.4rem;
            line-height: 1.2;
            margin: 0 0 1.25rem 0;
            letter-spacing: 0.4px;
        }
        .app-subtitle {
            text-align: center;
            color: #6b7280;
            margin: 0 0 3rem 0;
            font-size: 1rem;
        }

        /* headings */
        h3, .stMarkdown h3, .stMarkdown h2 {
            margin-top: 2.5rem; /* more spacing before headings */
        }

        /* metric cards */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.05);
            text-align: center;
        }

        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            display: block;
            color: #4b5563;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 8px;
            text-align: center;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            text-align: center;
        }

        /* markdown spacing */
        .stMarkdown p {
            margin-bottom: 0.6rem;
        }

        /* bolded text in a bullet list */
        .stMarkdown li strong {
            font-size: 1rem;
        }
    </style>
    """

SEGMENT_COLOR_MAP = {
    "Champions": "#A8E6A3",          
    "Active": "#B6E3D4",             
    "Newcomers": "#B5D8F6",          
    "Fading Loyalists": "#FFD8A8",   
    "Inactive": "#FFE0CC",           
    "At Risk (Low Value)": "#FFF6A3",
    "Can't Lose Them": "#F6A6A6",    
    "Reactivation Pool": "#D8C7FF",  
    "Lost Casual": "#D9D9D9"         
}


def color_map_from_sizes(sizes):
    cmap = matplotlib.cm.coolwarm
    mini, maxi = np.min(sizes), np.max(sizes)
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    return [cmap(norm(v)) for v in sizes]