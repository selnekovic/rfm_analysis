import io
import streamlit as st
import polars as pl

# -------------------------
# cached I/O + prep helpers
# -------------------------

@st.cache_data(show_spinner=False)
def _read_sample_csv(path: str) -> pl.DataFrame:
    return pl.read_csv(path)

@st.cache_data(show_spinner=False)
def _read_uploaded_csv(content: bytes) -> pl.DataFrame:
    # cache on file content so reruns are instant
    return pl.read_csv(io.BytesIO(content))

@st.cache_data(show_spinner=False)
def _prepare_cached(df: pl.DataFrame) -> pl.DataFrame:
    # local import to avoid circular import
    from _import import prepare_rfm_columns
    return prepare_rfm_columns(df)

def _distinct(*names: str) -> bool:
    return len(set(names)) == len(names)

# -------------------------
# main sidebar entry point
# -------------------------

def sidebar_import(sample_data: str = "sample_data.csv") -> pl.DataFrame:
    """
    behavior:
    - sample mode: read + prepare (cached) and return immediately
    - uploaded mode: after first 'import', keep using the prepared df across
      reruns without showing mapping UI or calling st.stop();
      provide a 'change data / remap' button to re-open mapping.
    """

    # init flags
    st.session_state.setdefault("data_mode", "sample")
    st.session_state.setdefault("data_version", 0)
    st.session_state.setdefault("uploaded_ready", False)  # true after successful import
    st.session_state.setdefault("raw_key", None)
    st.session_state.setdefault("raw_df", None)

    with st.sidebar:
        st.markdown("# Data input")

        uploaded = st.file_uploader(
            "upload a csv file",
            type=["csv"],
            accept_multiple_files=False,
            key="uploader_file",
        )

        # identify the uploaded file deterministically (name + size)
        uploaded_id = None
        uploaded_bytes = None
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            uploaded_id = (uploaded.name, len(uploaded_bytes))

        # if a new file is selected while we were 'ready', force remap
        if uploaded_id is not None:
            prev_id = st.session_state.get("uploaded_id")
            if prev_id is None or prev_id != uploaded_id:
                st.session_state["uploaded_id"] = uploaded_id
                st.session_state["uploaded_ready"] = False  # must remap/confirm
                st.session_state["data_mode"] = "uploaded"

        # ---------- SAMPLE DATA PATH ----------
        if (uploaded is None) and (st.session_state["data_mode"] == "sample"):
            try:
                df_sample = _read_sample_csv(sample_data)
                df_prepared = _prepare_cached(df_sample)
            except Exception as e:
                st.error(f"failed to read/prepare sample '{sample_data}': {e}")
                st.stop()

            # set stable key + version bump if changed
            new_key = ("sample", sample_data, df_prepared.height, df_prepared.width)
            if st.session_state.get("raw_key") != new_key:
                st.session_state["raw_key"] = new_key
                st.session_state["data_version"] += 1
            st.session_state["raw_df"] = df_prepared

            with st.expander("data source", expanded=False):
                st.write("using bundled sample data")
                st.code(f"path: {sample_data}", language="text")

            return st.session_state["raw_df"]

        # ---------- UPLOADED DATA READY PATH ----------
        # once imported, keep using the mapped df across reruns (no st.stop)
        if st.session_state["data_mode"] == "uploaded" and st.session_state["uploaded_ready"]:
            if st.session_state.get("raw_df") is None:
                # safety fallback, treat as not ready
                st.session_state["uploaded_ready"] = False
            else:
                with st.expander("data source", expanded=False):
                    fname = st.session_state.get("uploaded_id", ("uploaded.csv", 0))[0]
                    st.write(f"using imported file: **{fname}**")
                    if st.button("change data / remap", use_container_width=True, key="btn_remap"):
                        st.session_state["uploaded_ready"] = False  # reopen mapping UI on next rerun
                return st.session_state["raw_df"]

        # if we reach here, either:
        # - a file is uploaded but not yet imported, or
        # - user clicked 'change data / remap'
        # show the mapping UI; only block with st.stop() until 'import' is clicked

        if uploaded is None:
            st.info("upload a csv or clear the file to go back to sample data")
            st.stop()

        # read uploaded content (cached by bytes)
        try:
            df_uploaded = _read_uploaded_csv(uploaded_bytes)
        except Exception as e:
            st.error(f"failed to read uploaded file: {e}")
            st.stop()

        if df_uploaded.is_empty():
            st.error("the uploaded file is empty")
            st.stop()

        st.markdown("## column mapping")
        cols = df_uploaded.columns

        user_col = st.selectbox(
            "user id column",
            options=cols,
            index=None,
            placeholder="choose a column…",
            key="map_user",
        )
        date_col = st.selectbox(
            "date column (YYYY-MM-DD or YYYYMMDD)",
            options=cols,
            index=None,
            placeholder="choose a column…",
            key="map_date",
        )
        value_col = st.selectbox(
            "value column",
            options=cols,
            index=None,
            placeholder="choose a column…",
            key="map_value",
        )

        st.markdown("## additional settings")
        st.session_state["rf_variant"] = st.radio(
            "metric set",
            options=[
                "Recency Frequency Monetary (e.g. revenue)",
                "Recency Frequency Engagement (e.g. page views)",
            ],
            index=0,
            key="rf_variant_radio",
        )
        st.session_state["remove_outliers"] = st.checkbox(
            "remove outliers",
            value=st.session_state.get("remove_outliers", False),
            key="remove_outliers_cb",
        )

        if st.button("import", use_container_width=True, key="btn_import"):
            if user_col is None or date_col is None or value_col is None:
                st.error("please choose all three columns")
                st.stop()
            if not _distinct(user_col, date_col, value_col):
                st.error("please choose three different columns")
                st.stop()

            try:
                df_mapped = (
                    df_uploaded.select([pl.col(user_col), pl.col(date_col), pl.col(value_col)])
                    .rename({user_col: "user_id", date_col: "date", value_col: "value"})
                )
                df_prepared = _prepare_cached(df_mapped)
            except Exception as e:
                st.error(f"data preparation failed: {e}")
                st.stop()

            # set keys + bump version if changed
            new_key = (
                "uploaded",
                uploaded_id,           # (name, size)
                user_col, date_col, value_col,
                df_prepared.height, df_prepared.width,
            )
            if st.session_state.get("raw_key") != new_key:
                st.session_state["raw_key"] = new_key
                st.session_state["data_version"] += 1

            st.session_state["raw_df"] = df_prepared
            st.session_state["data_mode"] = "uploaded"
            st.session_state["uploaded_ready"] = True

            # return immediately (no stop), main.py will continue this rerun
            return st.session_state["raw_df"]

        # while waiting for 'import', we intentionally block execution,
        # so the main app doesn't run with half-configured data
        st.info("choose the three columns, then click **import**")
        st.stop()

# -------------------------
# validation / preparation
# -------------------------

def prepare_rfm_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    ensures df has columns: user_id, date, value.
    - value: cast to Float64 (strict)
    - date: accept YYYY-MM-DD, YYYYMMDD, integer YYYYMMDD, Datetime; output pl.Date
    """
    required_cols = ["user_id", "date", "value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    out = df.clone().drop_nulls()

    # value -> Float64
    val_dtype = out.schema["value"]
    if val_dtype == pl.Utf8:
        value_cast = pl.col("value").cast(pl.Float64, strict=False)
        invalid = value_cast.is_null() & pl.col("value").is_not_null()
        n_invalid = out.select(invalid.sum().alias("n")).to_series(0)[0]
        if n_invalid and n_invalid > 0:
            examples = (
                out.filter(invalid)
                .select("value")
                .head(5)
                .to_series(0)
                .to_list()
            )
            raise ValueError(
                "value contains non-numeric strings that cannot be cast to float "
                f"(examples: {examples})"
            )
        out = out.with_columns(value_cast.alias("value"))
    elif val_dtype in pl.INTEGER_DTYPES or val_dtype in pl.FLOAT_DTYPES:
        out = out.with_columns(pl.col("value").cast(pl.Float64))
    else:
        raise ValueError(f"unsupported dtype for value: {val_dtype}")

    # date -> pl.Date
    date_dtype = out.schema["date"]
    if date_dtype == pl.Date:
        pass
    elif date_dtype == pl.Datetime:
        out = out.with_columns(pl.col("date").dt.date())
    elif date_dtype in pl.INTEGER_DTYPES:
        parsed = (
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
        )
        invalid = parsed.is_null() & pl.col("date").is_not_null()
        n_invalid = out.select(invalid.sum().alias("n")).to_series(0)[0]
        if n_invalid and n_invalid > 0:
            examples = (
                out.filter(invalid)
                .select("date")
                .head(5)
                .to_series(0)
                .to_list()
            )
            raise ValueError("date could not be parsed (integer YYYYMMDD expected)")
        out = out.with_columns(parsed.alias("date"))
    elif date_dtype == pl.Utf8:
        parsed_iso = pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        parsed_compact = pl.col("date").str.strptime(pl.Date, format="%Y%m%d", strict=False)
        parsed = pl.coalesce([parsed_iso, parsed_compact])
        invalid = parsed.is_null() & pl.col("date").is_not_null()
        n_invalid = out.select(invalid.sum().alias("n")).to_series(0)[0]
        if n_invalid and n_invalid > 0:
            examples = (
                out.filter(invalid)
                .select("date")
                .head(5)
                .to_series(0)
                .to_list()
            )
            raise ValueError(
                "date must be 'YYYY-MM-DD' or 'YYYYMMDD' (string). "
                f"unparseable examples: {examples}"
            )
        out = out.with_columns(parsed.alias("date"))
    else:
        raise ValueError(f"unsupported dtype for date: {date_dtype}")

    # guarantees
    assert out.schema["value"] == pl.Float64, "value is not pl.Float64 after casting"
    assert out.schema["date"] == pl.Date, "date is not pl.Date after parsing"
    return out
