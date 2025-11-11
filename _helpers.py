import polars as pl

from datetime import date

def remove_outliers_percentile(
        df: pl.DataFrame,
        column: str = 'value',
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
) -> pl.DataFrame:
    
    """
    Remove outliers from a DataFrame based on specified percentiles.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    column (str): The column name to check for outliers.
    lower_percentile (float): The lower percentile threshold.
    upper_percentile (float): The upper percentile threshold.

    Returns:
    pl.DataFrame: DataFrame with outliers removed.
    """
    lower_bound = df.select(pl.col(column).quantile(lower_percentile)).item()
    upper_bound = df.select(pl.col(column).quantile(upper_percentile)).item()

    return df.filter(
        (pl.col(column) >= lower_bound) & (pl.col(column) <= upper_bound)
    )


def rfm_transformation(
        df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary values per user.

    Parameters
    ----------
    df : pl.DataFrame
        Must include:
        - "user_id": customer ID
        - "date": transaction date
        - "value": transaction amount

    Returns
    -------
    pl.DataFrame
        One row per user with Recency (days since last purchase),
        Frequency (transaction count), and Monetary (total spend).
    """

    analysis_date = df.select(pl.col("date").max()).item()
    
    rfm_df = (
        df
        .with_columns(
            pl.col('date')
        )
        .group_by('user_id')
        .agg([
                (pl.lit(analysis_date).cast(pl.Date) - pl.col('date').max()).dt.total_days().cast(pl.Int64).alias('recency'),
                pl.count().alias('frequency'),
                pl.col('value').sum().alias('monetary')
            ]
        )
    )

    return rfm_df


def rfm_scoring(
        df: pl.DataFrame
) -> pl.DataFrame:
    """
    Assign 1-5 RFM scores based on quintiles.

    Parameters
    ----------
    df : pl.DataFrame
        Must include Recency, Frequency, and Monetary columns.

    Returns
    -------
    pl.DataFrame
        Original RFM metrics plus:
        r_score, f_score, m_score, rfm_total, and rfm_string.
    """

    # Compute quintile thresholds (20%, 40%, 60%, 80%) for each metric.
    def _thresholds(column: str) -> list[float]:
        qs = [0.2, 0.4, 0.6, 0.8]
        return [df.select(pl.col(column).quantile(q)).item() for q in qs]

    r_th = _thresholds('recency')
    f_th = _thresholds('frequency')
    m_th = _thresholds('monetary')

    # Build scoring expression - higher_is_better=True
    def _score_expr(col: str, thresholds: list[float], higher_is_better: bool) -> pl.Expr:
        if higher_is_better:
            return (
                pl.when(pl.col(col) <= thresholds[0]).then(1)
                .when(pl.col(col) <= thresholds[1]).then(2)
                .when(pl.col(col) <= thresholds[2]).then(3)
                .when(pl.col(col) <= thresholds[3]).then(4)
                .otherwise(5)
                .cast(pl.Int64)
            )
        else:
            return (
                pl.when(pl.col(col) <= thresholds[0]).then(5)
                .when(pl.col(col) <= thresholds[1]).then(4)
                .when(pl.col(col) <= thresholds[2]).then(3)
                .when(pl.col(col) <= thresholds[3]).then(2)
                .otherwise(1)
                .cast(pl.Int64)
            )

    scored = (
        df
        .with_columns([
            _score_expr('recency', r_th, higher_is_better=True).alias('r_score'),
            _score_expr('frequency', f_th, higher_is_better=True).alias('f_score'),
            _score_expr('monetary', m_th, higher_is_better=True).alias('m_score'),
        ])
        .with_columns([
            (pl.col('r_score') + pl.col('f_score') + pl.col('m_score')).alias('rfm_total'),
            pl.concat_str([
                pl.col('r_score').cast(pl.Utf8),
                pl.col('f_score').cast(pl.Utf8),
                pl.col('m_score').cast(pl.Utf8)
            ]).alias('rfm_string'),
        ])
    )

    return scored


def map_user_segment(row) -> str:
    """
    Numerical mapper using Recency (R) score and the sum of Frequency (F) + Revenue (M) scores.
    R: 1-5, F: 1-5, M (Revenue): 1-5.
    Segment names reflect Value (Revenue) and Activity (Recency).
    """
    try:
        r = int(row["r_score"])
        f = int(row["f_score"])
        m = int(row["m_score"]) 
    except (ValueError, KeyError):
        return "Invalid Score"

    fm_index = f + m
    
    # --- R = 4-5 (High Recency: Active Buyers) ---
    if r >= 4:
        if fm_index >= 8: # FM: 8-10 (Highest Value)
            return "Champions"
        elif fm_index >= 5: # FM: 5-7 (Medium Value)
            return "Active"
        else: # fm_index <= 4 (Low Value)
            return "Newcomers"

    # --- R = 3 (Medium Recency: Lapsing Activity) ---
    elif r == 3:
        if fm_index >= 8: # FM: 8-10
            return "Fading Loyalists"
        elif fm_index >= 5: # FM: 5-7
            return "Inactive"
        else: # fm_index <= 4
            return "At Risk (Low Value)"

    # --- R = 1-2 (Low Recency: Inactive Buyers) ---
    else: # r <= 2
        if fm_index >= 8: # FM: 8-10 (High Historical Value, Inactive)
            return "Can't Lose Them"
        elif fm_index >= 5: # FM: 5-7 (Medium Historical Value, Inactive)
            return "Reactivation Pool"
        else: # fm_index <= 4 (Low Historical Value, Inactive)
            return "Lost Casual"
        
