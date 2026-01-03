# app.py — EasyWIF (What-If Scenario Planning)
# Run: python -m streamlit run app.py
import io
import os
import uuid
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import tempfile
from pathlib import Path
import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Authentication System - MUST BE FIRST
# ============================================================
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password.
        else:
            st.session_state["password_correct"] = False

    # Show authentication screen
    if "password_correct" not in st.session_state:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #0071C5; margin-bottom: 1rem;">EasyWIF</h1>
            <p style="color: #FAFAFA; font-size: 1.1rem; margin-bottom: 1rem;">
                Proof of Concept Only - Using Dummy Data
            </p>
            <p style="color: #FAFAFA; font-size: 0.9rem; margin-bottom: 2rem;">
                For questions please contact Paul Landrum
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the password input
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Enter Password", type="password", on_change=password_entered, key="password", 
                         placeholder="Enter your password")
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; color: #FAFAFA; font-size: 0.9rem;">
                🔒 Secure Application
            </div>
            """, unsafe_allow_html=True)
        return False
    # Password correct.
    elif st.session_state["password_correct"]:
        return True
    # Password incorrect, show input + error.
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #0071C5; margin-bottom: 1rem;">EasyWIF</h1>
            <p style="color: #FAFAFA; font-size: 1.1rem; margin-bottom: 1rem;">
                Proof of Concept Only - Using Dummy Data
            </p>
            <p style="color: #FAFAFA; font-size: 0.9rem; margin-bottom: 2rem;">
                For questions please contact Paul Landrum
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the password input
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Enter Password", type="password", on_change=password_entered, key="password",
                         placeholder="Enter your password")
            st.error("😕 Incorrect password. Please try again.")
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; color: #FAFAFA; font-size: 0.9rem;">
                🔒 Secure Application
            </div>
            """, unsafe_allow_html=True)
        return False

# Check authentication BEFORE anything else
if not check_password():
    st.stop()  # Do not continue if not authenticated.

# Debug logging function
def debug_log(message: str):
    """Write debug message to file only"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_message = f"[{timestamp}] {message}"
    
    # Write to debug file
    try:
        with open("debug.txt", "a", encoding="utf-8") as f:
            f.write(debug_message + "\n")
    except Exception as e:
        # Avoid infinite recursion - just print to console if file write fails
        try:
            print(f"Debug file write error: {e}")
        except:
            pass  # If even print fails, just ignore it


# ============================================================
# App config (MUST be defined before loaders use them)
# ============================================================

# GROQ API Configuration




FILTER_COLUMNS_ORDER = [
    "Resource_Type", "Resource_Name", "Department",
    "Team", "Workstream", "Project", "Account_Code",
]
DATE_COLUMNS = ["TI Date", "PO Date", "PRQ Date"]

DISPLAY_YEARS = [2025, 2026, 2027, 2028, 2029]  # 5y for profile / gantt
YEARS3 = DISPLAY_YEARS[:3]                      # 3y for summary row
X_MIN = pd.Timestamp(f"{min(DISPLAY_YEARS)}-01-01")
X_MAX = pd.Timestamp(f"{max(DISPLAY_YEARS)}-12-31")

# ============================================================
# Helpers: file signatures, time utils, smart loader, accelerators
# ============================================================
def apply_shift_months(df: pd.DataFrame, row_mask: pd.Series, months: int, time_cols: list[str]) -> pd.DataFrame:
    """Shift selected rows by whole months along time_cols. Positive = to the future (right)."""
    if not int(months):
        return df
    k = int(months)

    # Extract numeric matrix for the masked rows
    mat = (
        df.loc[row_mask, time_cols]
          .apply(pd.to_numeric, errors="coerce")
          .fillna(0.0)
          .to_numpy(dtype="float64")
    )
    if mat.size == 0:
        return df

    shifted = np.zeros_like(mat)
    if k > 0:
        # shift right by k: the first k months become 0
        if k < mat.shape[1]:
            shifted[:, k:] = mat[:, :-k]
        # else everything becomes 0 (shift exceeds horizon)
    else:
        k = -k
        # shift left by k: the last k months become 0
        if k < mat.shape[1]:
            shifted[:, :-k] = mat[:, k:]
        # else everything becomes 0

    # Write back
    df.loc[row_mask, time_cols] = shifted
    return df

def shift_phase_dates(df: pd.DataFrame, row_mask: pd.Series, months: int) -> pd.DataFrame:
    """Shift TI/PO/PRQ date columns and Start_Month/End_Month by months if present."""
    if months == 0:
        return df
    # Shift standard date columns
    for c in ["TI Date", "PO Date", "PRQ Date"]:
        if c in df.columns:
            df.loc[row_mask, c] = pd.to_datetime(df.loc[row_mask, c], errors="coerce") + pd.DateOffset(months=months)
    
    # Shift Start_Month and End_Month columns (MM/YYYY format)
    def shift_month_string(month_str, months_offset):
        """Shift a MM/YYYY or YYYY-MM string by months_offset months. Handles extended periods where months 13-24 represent next year."""
        if pd.isna(month_str):
            return month_str
        month_str = str(month_str).strip()
        try:
            # Try MM/YYYY format first
            if '/' in month_str:
                parts = month_str.split('/')
                if len(parts) == 2:
                    month, year = int(parts[0]), int(parts[1])
                    # Handle extended periods: months 13-24 = months 1-12 of following year
                    if month > 12:
                        year += (month - 1) // 12
                        month = ((month - 1) % 12) + 1
                    dt = pd.Timestamp(year=year, month=month, day=1)
                    shifted = dt + pd.DateOffset(months=months_offset)
                    return f"{shifted.month:02d}/{shifted.year}"
            # Try YYYY-MM format (including extended months like 2026-24)
            elif '-' in month_str:
                parts = month_str.split('-')
                if len(parts) == 2:
                    year, month = int(parts[0]), int(parts[1])
                    # Handle extended periods: months 13-24 = months 1-12 of following year
                    if month > 12:
                        year += (month - 1) // 12
                        month = ((month - 1) % 12) + 1
                    dt = pd.Timestamp(year=year, month=month, day=1)
                    shifted = dt + pd.DateOffset(months=months_offset)
                    return f"{shifted.year}-{shifted.month:02d}"
        except:
            pass
        return month_str
    
    # Apply shift to Start_Month and End_Month
    for c in ["Start_Month", "End_Month"]:
        if c in df.columns and row_mask.any():
            # Get the values for the masked rows
            masked_values = df.loc[row_mask, c].copy()
            # Apply the shift function
            shifted_values = masked_values.apply(lambda x: shift_month_string(x, months))
            # Assign back to the dataframe
            df.loc[row_mask, c] = shifted_values
    
    return df

def _file_sig(path: str) -> str:
    p = Path(path)
    try:
        return f"{p.resolve()}::{p.stat().st_mtime_ns}"
    except Exception:
        return str(path)

def is_headcount_row(df: pd.DataFrame) -> pd.Series:
    if "Account" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Account"].astype(str).str.contains("headcount", case=False, na=False)

def _filehash_from_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

def _is_yyyymm(s: str) -> bool:
    """Return True if s looks like YYYYMM (e.g., '202501') or YYYY-MM (e.g., '2026-01')."""
    s = str(s).strip()
    # Check for YYYYMM format (6 digits)
    if len(s) == 6 and s.isdigit():
        return True
    # Check for YYYY-MM format (e.g., '2026-01')
    if len(s) == 7 and s.count('-') == 1:
        parts = s.split('-')
        if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4 and parts[1].isdigit() and len(parts[1]) == 2:
            return True
    return False

def yyyymm_to_dt(mm: str) -> pd.Timestamp:
    """Convert YYYYMM or YYYY-MM format to datetime. Handles extended periods where months 13-24 represent next year."""
    mm = str(mm).strip()
    try:
        # Handle YYYY-MM format
        if '-' in mm:
            parts = mm.split('-')
            if len(parts) == 2:
                year, month = int(parts[0]), int(parts[1])
                # Handle extended periods: months 13-24 = months 1-12 of following year
                if month > 12:
                    year += (month - 1) // 12
                    month = ((month - 1) % 12) + 1
                return pd.to_datetime(f"{year}-{month:02d}-01", format="%Y-%m-%d", errors="coerce")
        # Handle YYYYMM format
        else:
            if len(mm) == 6 and mm.isdigit():
                year, month = int(mm[:4]), int(mm[4:])
                # Handle extended periods: months 13-24 = months 1-12 of following year
                if month > 12:
                    year += (month - 1) // 12
                    month = ((month - 1) % 12) + 1
                return pd.to_datetime(f"{year}{month:02d}01", format="%Y%m%d", errors="coerce")
    except:
        pass
    return pd.NaT

def total_series(df_: pd.DataFrame, tcols: List[str]) -> pd.Series:
    """Monthly totals in raw units; filters to spending rows (non-headcount)."""
    df_ = df_[~is_headcount_row(df_)]
    if not tcols:
        return pd.Series([], dtype="float64")
    m = df_[tcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return m.sum(axis=0)

@st.cache_data(show_spinner=True, max_entries=4, ttl=3600)
def smart_load_any(path: str) -> pd.DataFrame:
    """CSV → fast parquet sidecar on first load; reuses parquet thereafter."""
    p = Path(path)
    
    # If parquet/feather, load directly (fastest)
    if p.suffix.lower() in {".parquet", ".feather"}:
        debug_log(f"Loading {path} directly (fastest)")
        return (pd.read_parquet(path) if p.suffix.lower()==".parquet" else pd.read_feather(path))

    # Sidecar parquet check (very fast)
    parquet_sidecar = p.with_suffix(".parquet")
    if parquet_sidecar.exists() and parquet_sidecar.stat().st_mtime >= p.stat().st_mtime:
        debug_log(f"Loading cached parquet sidecar: {parquet_sidecar}")
        return pd.read_parquet(parquet_sidecar, engine="pyarrow")

    # First time: read CSV (slowest)
    debug_log(f"First time loading CSV: {path}")
    try:
        # Use pyarrow with optimized settings for fastest CSV loading
        df = pd.read_csv(path, engine="pyarrow", dtype_backend="pyarrow", memory_map=True)
        debug_log(f"Loaded with pyarrow (fast)")
    except Exception as e:
        debug_log(f"Pyarrow failed: {e}, falling back to pandas")
        df = pd.read_csv(path)

    # Optimize data types for speed
    debug_log("Optimizing data types...")
    
    # Convert known date columns
    for c in DATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Cast filter columns to category for speed (much faster than object)
    for c in FILTER_COLUMNS_ORDER:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("").astype("category")

    # Ensure YYYYMM time cols are numeric float32 (faster than float64)
    tcols = [c for c in df.columns if _is_yyyymm(c)]
    if tcols:
        df[tcols] = df[tcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    # Save optimized parquet sidecar for future fast loads
    debug_log(f"Saving optimized parquet sidecar: {parquet_sidecar}")
    try:
        df.to_parquet(parquet_sidecar, compression="zstd", index=False, engine="pyarrow")
        debug_log("Saved with zstd compression (fastest)")
    except Exception as e:
        debug_log(f"Zstd failed: {e}, using default compression")
        df.to_parquet(parquet_sidecar, index=False)
    
    debug_log(f"Data loading complete. Shape: {df.shape}")
    return df

@st.cache_resource(show_spinner=False, max_entries=4, ttl=3600)
def build_accelerators(df: pd.DataFrame,
                       filter_cols: List[str],
                       cache_key: str) -> dict:
    time_cols = [c for c in df.columns if _is_yyyymm(c)]
    months = pd.Series(
        [yyyymm_to_dt(c) for c in time_cols],
        dtype="datetime64[ns]"
    )
    valid = months.notna().to_numpy()
    time_cols_valid = [c for c, ok in zip(time_cols, valid) if ok]

    # axis max on valid cols only
    axis_max = float(df[time_cols_valid].sum(axis=0).max() or 1.0) if time_cols_valid else 1.0

    # boolean masks for filters (works great with category dtype)
    mask_index: dict[str, dict[str, np.ndarray]] = {}
    for col in filter_cols:
        if col not in df.columns:
            continue
        s = df[col].astype("string").fillna("")
        uniq = sorted(s.unique().tolist())
        colmap = {v: (s == v).to_numpy() for v in uniq}
        mask_index[col] = colmap

    return {
        "months": months,
        "time_cols_valid": time_cols_valid,
        "axis_max": axis_max,
        "mask_index": mask_index,
        "sig": cache_key,
    }

def scope_mask(df: pd.DataFrame, selected: dict) -> pd.Series:
    """
    Build a boolean mask ON THIS DF'S INDEX ONLY.
    - AND across columns
    - OR within a column (multiselect) via .isin()
    - No selection for a column => no restriction
    """
    mask = pd.Series(True, index=df.index)  # start all True on this df's index

    if not selected:
        return mask

    for col, vals in (selected or {}).items():
        if col not in df.columns:
            continue

        # normalize selections
        vals = [str(v) for v in (vals or []) if v is not None and str(v) != ""]
        if not vals:
            continue  # nothing chosen for this column => no restriction

        col_s = df[col].astype("string")
        m = col_s.isin(vals)  # OR within the column
        # align to df.index explicitly to avoid length mismatches
        mask = mask & m.reindex(df.index, fill_value=False)

    # ensure strictly boolean and aligned
    return mask.astype(bool)

def get_dynamic_filter_options(df: pd.DataFrame, filter_columns: List[str], current_selections: dict) -> dict:
    """
    Get available options for each filter column based on current selections.
    This implements cascading/dependent filtering.
    """
    available_options = {}
    
    # Start with all data
    current_mask = pd.Series(True, index=df.index)
    
    for col in filter_columns:
        if col not in df.columns:
            available_options[col] = []
            continue
            
        # Apply all previous filters to get available options for this column
        filtered_df = df[current_mask]
        
        # Get unique values for this column from the filtered data
        if filtered_df.empty:
            available_options[col] = []
        else:
            available_options[col] = sorted(filtered_df[col].dropna().astype(str).unique().tolist())
        
        # Update mask for next iteration by applying current column's selection
        if col in current_selections and current_selections[col]:
            col_mask = df[col].astype("string").isin([str(v) for v in current_selections[col]])
            current_mask = current_mask & col_mask
    
    return available_options

def get_dynamic_time_options(df: pd.DataFrame, current_selections: dict, time_cols: List[str]) -> List[str]:
    """
    Get available time columns based on current filter selections.
    Only returns months where the filtered data actually has non-zero values.
    """
    if not current_selections or not any(current_selections.values()):
        return time_cols
    
    # Apply current filters to get relevant data
    mask_scope = scope_mask(df, current_selections)
    filtered_df = df[mask_scope]
    
    if filtered_df.empty:
        return []
    
    # Find months with non-zero values in the filtered data
    available_months = []
    for col in time_cols:
        if col in filtered_df.columns:
            # Check if this month has any non-zero values
            if filtered_df[col].sum() > 0:
                available_months.append(col)
    
    return available_months if available_months else time_cols

def create_slicer_buttons(column_name: str, options: List[str], selected_values: List[str], key_prefix: str, disabled_options: List[str] = None) -> List[str]:
    """
    Create compact Excel-style slicer buttons for filtering.
    Returns the selected values.
    """
    if disabled_options is None:
        disabled_options = []
    
    # Create compact container
    st.markdown(f"""
    <div class="slicer-container">
        <div class="slicer-title">{column_name}</div>
        <div class="slicer-buttons">
    """, unsafe_allow_html=True)
    
    selected = selected_values.copy()
    
    # Create buttons in a 2-column grid layout
    for i, option in enumerate(options):
        is_selected = option in selected_values
        is_disabled = option in disabled_options
        button_key = f"{key_prefix}_{option}"
        
        # Create clickable button
        if st.button(
            option,
            key=button_key,
            type="primary" if is_selected else "secondary",
            disabled=is_disabled,
            use_container_width=True
        ):
            if not is_disabled:
                if is_selected:
                    selected.remove(option)
                else:
                    selected.append(option)
                st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    return selected

def create_slicer_row(col1_name: str, col1_options: List[str], col1_selected: List[str], col1_key: str,
                     col2_name: str, col2_options: List[str], col2_selected: List[str], col2_key: str,
                     col1_disabled: List[str] = None, col2_disabled: List[str] = None):
    """
    Create a row with two side-by-side slicer sections.
    """
    if col1_disabled is None:
        col1_disabled = []
    if col2_disabled is None:
        col2_disabled = []
    
    st.markdown('<div class="slicer-row">', unsafe_allow_html=True)
    
    # First column
    with st.container():
        create_slicer_buttons(col1_name, col1_options, col1_selected, col1_key, col1_disabled)
    
    # Second column
    with st.container():
        create_slicer_buttons(col2_name, col2_options, col2_selected, col2_key, col2_disabled)
    
    st.markdown('</div>', unsafe_allow_html=True)



def create_excel_style_pivot_table(df: pd.DataFrame, time_cols: List[str], metric_name: str):
    """
    Create an Excel-style pivot table with embedded expand/collapse buttons.
    """
    # Determine hierarchy levels
    hierarchy_levels = ["Group", "Division", "CCRU", "Cost Center"]
    available_levels = [level for level in hierarchy_levels if level in df.columns]
    
    if not available_levels:
        st.info(f"No hierarchy columns found for {metric_name.lower()} data.")
        return
    
    # Initialize expansion state
    if "pivot_row_level" not in st.session_state:
        st.session_state["pivot_row_level"] = 0
    if "pivot_col_level" not in st.session_state:
        st.session_state["pivot_col_level"] = 0
    
    current_row_level = available_levels[min(st.session_state["pivot_row_level"], len(available_levels)-1)]
    
    # Create time period columns based on expansion level
    if st.session_state["pivot_col_level"] == 0:
        # Show years only
        time_periods = [str(year) for year in DISPLAY_YEARS if year >= 2025]
        time_period_cols = []
        for year in time_periods:
            year_months = [col for col in time_cols if col.startswith(year)]
            if year_months:
                time_period_cols.append(year)
    else:
        # Show quarters
        time_periods = []
        time_period_cols = []
        for year in DISPLAY_YEARS:
            if year >= 2025:
                for quarter in range(1, 5):
                    quarter_months = []
                    for month in range((quarter-1)*3+1, quarter*3+1):
                        month_str = f"{year}{month:02d}"
                        if month_str in time_cols:
                            quarter_months.append(month_str)
                    if quarter_months:
                        quarter_name = f"Q{quarter}'{str(year)[-2:]}"
                        time_periods.append(quarter_name)
                        time_period_cols.append(quarter_months)
    
    # Aggregate data
    if st.session_state["pivot_col_level"] == 0:
        # Year aggregation
        pivot_data = df.groupby(current_row_level)[time_cols].sum().reset_index()
        for year in time_periods:
            year_months = [col for col in time_cols if col.startswith(year)]
            if year_months:
                pivot_data[year] = pivot_data[year_months].sum(axis=1)
    else:
        # Quarter aggregation
        pivot_data = df.groupby(current_row_level)[time_cols].sum().reset_index()
        for quarter_name, quarter_months in zip(time_periods, time_period_cols):
            if quarter_months:
                pivot_data[quarter_name] = pivot_data[quarter_months].sum(axis=1)
    
    # Create Excel-style table with expand buttons
    st.markdown(f"### {metric_name} by {current_row_level}")
    
    # Create the table with embedded controls
    table_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
        <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
            <thead>
                <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                    <th style="padding: 12px; text-align: left; border-right: 1px solid #dee2e6; min-width: 200px;">
                        {current_row_level}
                        <button onclick="expandRows()" style="margin-left: 10px; padding: 2px 6px; background: #0078d4; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 10px;">
                            {'➖' if st.session_state['pivot_row_level'] > 0 else '➕'}
                        </button>
                    </th>
    """
    
    # Add time period headers with expand button
    for period in time_periods:
        table_html += f"""
                    <th style="padding: 12px; text-align: center; border-right: 1px solid #dee2e6; min-width: 100px;">
                        {period}
                    </th>
        """
    
    table_html += """
                </tr>
            </thead>
            <tbody>
    """
    
    # Add data rows
    for _, row in pivot_data.iterrows():
        table_html += f"""
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 12px; border-right: 1px solid #dee2e6; font-weight: bold;">
                        {row[current_row_level]}
                    </td>
        """
        
        for period in time_periods:
            value = row.get(period, 0)
            formatted_value = f"${value:,.0f}" if value != 0 else "-"
            table_html += f"""
                    <td style="padding: 12px; text-align: right; border-right: 1px solid #dee2e6;">
                        {formatted_value}
                    </td>
            """
        
        table_html += """
                </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    </div>
    """
    
    # Add column expansion control
    col_expand_html = f"""
    <div style="margin-top: 10px; text-align: right;">
        <button onclick="expandColumns()" style="padding: 6px 12px; background: #0078d4; color: white; border: none; border-radius: 4px; cursor: pointer;">
            {'➖ Collapse to Years' if st.session_state['pivot_col_level'] > 0 else '➕ Expand to Quarters'}
        </button>
    </div>
    """
    
    # Display the table
    st.markdown(table_html + col_expand_html, unsafe_allow_html=True)
    
    # Handle expansion controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Expand Rows", key="expand_rows_btn"):
            if st.session_state["pivot_row_level"] < len(available_levels) - 1:
                st.session_state["pivot_row_level"] += 1
            st.rerun()
    with col2:
        if st.button("Expand Columns", key="expand_cols_btn"):
            st.session_state["pivot_col_level"] = 1 - st.session_state["pivot_col_level"]
            st.rerun()


# ============================================================
# Data bootstrap (order matters!) - with performance optimization
# ============================================================
if "data_name" not in st.session_state:
    # Always default to "Demo Data_Post Production (Small - 50 lines, 12 months).csv"
    default_file = "Demo Data_Post Production (Small - 50 lines, 12 months).csv"
    # Check if the file exists, if not, try to find it or use it anyway (will error if truly missing)
    if os.path.exists(default_file):
        st.session_state["data_name"] = default_file
    else:
        # Fallback: try to find any Post Production file
        demo_files = [f for f in os.listdir(".") if f.startswith("Demo Data") and f.endswith(".csv")]
        post_prod_files = [f for f in demo_files if "Post Production" in f and "12 months" in f]
        if post_prod_files:
            st.session_state["data_name"] = sorted(post_prod_files)[0]
        elif demo_files:
            st.session_state["data_name"] = sorted(demo_files)[0]
        else:
            st.session_state["data_name"] = default_file

if "df_raw" not in st.session_state:
    with st.spinner("Loading data..."):
        st.session_state["df_raw"] = smart_load_any(st.session_state["data_name"])

df_raw = st.session_state["df_raw"]
if df_raw.empty:
    st.error("No data loaded.")
    st.stop()

# Build accelerators once per file signature (after df_raw is set)
acc_sig = _file_sig(st.session_state["data_name"])
st.session_state["accel"] = build_accelerators(df_raw, FILTER_COLUMNS_ORDER, acc_sig)

# Detect time columns and axis max (once, here)
all_cols = list(df_raw.columns)
time_cols = [c for c in all_cols if _is_yyyymm(c)]
if not time_cols:
    st.error("No time columns detected. Expect YYYYMM (e.g., 202501).")
    st.stop()

# Ensure numeric, compute axis max
df_raw[time_cols] = df_raw[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
axis_max = float(df_raw[time_cols].sum(axis=0).max() or 1.0)

# 5-year subset lists (only used for some summaries)
months_dt_all = pd.Series([yyyymm_to_dt(c) for c in time_cols])
valid_idx = months_dt_all.notna()
months_dt_all = months_dt_all[valid_idx]
time_cols_valid = [c for c, ok in zip(time_cols, valid_idx) if ok]
mask5 = pd.Series(months_dt_all).dt.year.isin(DISPLAY_YEARS).to_numpy()
months_dt5 = months_dt_all[mask5]
time_cols5 = [c for c, keep in zip(time_cols_valid, mask5) if keep]

# ============================================================
# Page / Theme / Styles
# ============================================================
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

st.set_page_config(
    page_title="EasyWIF - Roadmap & Forecast Planning", 
    layout="wide",
    initial_sidebar_state="expanded"
)
PLOTLY_TEMPLATE = "plotly_dark"

COLOR_CURRENT = "#1f77b4"   # solid blue
COLOR_NEW     = "#2ca02c"   # solid green
COLOR_TARGET  = "#FFD166"   # thin yellow for YoY trend
COLOR_CURRENT_BURDEN = "#7fb2df"  # lighter blue
COLOR_CURRENT_GSD    = COLOR_CURRENT
COLOR_NEW_BURDEN     = "#86d68a"  # lighter green
COLOR_NEW_GSD        = COLOR_NEW

st.markdown(
        """
        <style>
          .block-container { padding-top: 2rem; }
          .tight-title { font-size: 1.125rem; font-weight: 700; margin: 0.15rem 0 0.25rem 0; }
          .subtitle { font-size: 1.5rem; color: #cfd8e3; line-height: 1.15; margin: 0.1rem 0 0 0; }
          .directions { font-size: 1.1rem; color: #cfd8e3; margin: 0.25rem 0 0.6rem 0; }
          .demo-mode { color:#FFD600; font-weight:800; font-size:12pt; }
          .stPlotlyChart { margin-top: 0.10rem; }
          .yoy-compact input { max-width: 60px !important; text-align: right; }
          .stButton > button {
            white-space: nowrap !important;
            height: 38px !important;
            width: 100% !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
          }
          
          /* Mobile-specific button sizing */
          @media (max-width: 768px) {
            .stButton > button {
              height: 44px !important;
              font-size: 14px !important;
              padding: 8px 16px !important;
            }
          }
          .co-subhead { font-weight: 700; font-size: 1.0rem; margin-top: 0.25rem; }
          .co-help { font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.25rem; }
          
          /* Compact Excel-style slicer buttons */
          .slicer-row {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
          }
          .slicer-container {
            flex: 1;
            padding: 6px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background: #fafafa;
          }
          .slicer-title {
            font-size: 13px;
            font-weight: bold;
            color: white;
            background: #333;
            padding: 4px 8px;
            margin-bottom: 6px;
            text-transform: uppercase;
            border-radius: 3px;
          }
          .slicer-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
          }
          .slicer-button {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 3px;
            padding: 2px 6px;
            margin: 1px;
            font-size: 10px;
            cursor: pointer;
            display: inline-block;
            transition: all 0.2s;
            white-space: nowrap;
            min-width: 40px;
            text-align: center;
          }
          .slicer-button:hover {
            background-color: #f0f0f0;
            border-color: #0078d4;
          }
          .slicer-button.selected {
            background-color: #0078d4;
            color: white;
            border-color: #0078d4;
            font-weight: bold;
          }
          .slicer-button.disabled {
            background-color: #f5f5f5;
            color: #ccc;
            cursor: not-allowed;
            opacity: 0.4;
            border-color: #e0e0e0;
          }
          
          /* Professional floating chat */
          .floating-chat-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(40, 40, 40, 0.95);
            border-top: 1px solid #555;
            padding: 15px 20px;
            z-index: 1000;
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-height: 180px;
            margin-left: 400px;
          }
          .chat-header {
            text-align: center;
            color: white;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
          }
          .chat-content {
            display: flex;
            align-items: flex-start;
            gap: 15px;
            flex: 1;
          }
          .chat-toggle {
            background: #555;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 12px;
          }
          .chat-toggle:hover {
            background: #666;
          }
          .chat-input-section {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 10px;
          }
          .chat-input {
            background: #2a2a2a;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 20px 12px;
            color: white;
            flex: 1;
            font-size: 14px;
            min-height: 60px;
          }
          .chat-input:focus {
            outline: none;
            border-color: #0078d4;
          }
          .chat-send {
            background: #0078d4;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 20px 16px;
            cursor: pointer;
            font-size: 12px;
            min-height: 60px;
          }
          .chat-send:hover {
            background: #106ebe;
          }
          .chat-response {
            flex: 1;
            background: #2a2a2a;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 20px 12px;
            color: #e0e0e0;
            font-size: 13px;
            max-height: 250px;
            min-height: 60px;
            overflow-y: auto;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Header (title + directions)
# ============================================================
# Logo and title inline
st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
logo_col, title_col = st.columns([0.2, 0.8])
with logo_col:
    logo_path = "ezw logo_white.png"
    if Path(logo_path).exists():
        st.image(logo_path, width=180)
    else:
        st.write("")  # Empty space if logo not found
with title_col:
    st.markdown("<h1 style='margin-top: 0.5rem; padding-top: 0;'>EasyWIF - Roadmap & Forecast Planning</h1>", unsafe_allow_html=True)
    st.markdown('<div class="directions">Directions: Filter Data --> Make a Change --> Review Impact</div>', unsafe_allow_html=True)

# ============================================================
# Pending actions (before widgets)
# ============================================================
if st.session_state.get("pending_clear", False):
    for col in FILTER_COLUMNS_ORDER:
        st.session_state.pop(f"f_{col}", None)
    st.session_state["changes"] = []
    # Always default to "Demo Data_Post Production (Small - 50 lines, 12 months).csv"
    default_file = "Demo Data_Post Production (Small - 50 lines, 12 months).csv"
    # Check if file exists, if not try to find a similar one
    if not os.path.exists(default_file):
        demo_files = [f for f in os.listdir(".") if f.startswith("Demo Data") and f.endswith(".csv")]
        post_prod_12m = [f for f in demo_files if "Post Production" in f and "12 months" in f]
        if post_prod_12m:
            default_file = sorted(post_prod_12m)[0]
        elif demo_files:
            default_file = sorted(demo_files)[0]
    
    st.session_state["df_raw"] = smart_load_any(default_file)
    st.session_state["data_name"] = default_file
    st.session_state["pending_clear"] = False
    st.rerun()

if st.session_state.get("pending_reset_filters", False):
    for col in FILTER_COLUMNS_ORDER:
        st.session_state[f"f_{col}"] = []
    st.session_state["pending_reset_filters"] = False
    st.rerun()

if st.session_state.get("pending_reset_controls", False):
    for k in ["shift_months", "scale_mode", "pct_val", "scale_abs",
              "start_mm_sel", "end_mm_sel", "es_mode", "es_start", "es_end",
              "yoy_input", "addproj_name", "addproj_start"]:
        st.session_state.pop(k, None)
    st.session_state["pending_reset_controls"] = False
    st.rerun()

# ============================================================
# Mobile Sidebar Toggle & Configuration
# ============================================================
# Add custom CSS for mobile sidebar toggle
st.markdown("""
<style>
/* Mobile sidebar toggle button */
.mobile-toggle-btn {
    display: none;
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 1000;
    background: #0078d4;
    color: white;
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    font-size: 20px;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
}

.mobile-toggle-btn:hover {
    background: #106ebe;
    transform: scale(1.1);
}

/* Show toggle button only on mobile */
@media (max-width: 768px) {
    .mobile-toggle-btn {
        display: block;
    }
    
    /* Hide default Streamlit sidebar on mobile */
    .css-1d391kg {
        display: none !important;
    }
    
    /* Show sidebar when mobile menu is open */
    .css-1d391kg.mobile-open {
        display: block !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        z-index: 999 !important;
        background: rgba(0,0,0,0.9) !important;
        overflow-y: auto !important;
    }
    
    /* Mobile-friendly button sizing */
    .stButton > button {
        height: 44px !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        min-height: 44px !important;
    }
    
    /* Mobile-friendly filter buttons */
    .slicer-button {
        padding: 8px 12px !important;
        font-size: 12px !important;
        min-width: 60px !important;
        height: 44px !important;
    }
    
    /* Mobile-friendly form elements */
    .stSelectbox > div > div {
        min-height: 44px !important;
    }
    
    .stTextInput > div > div {
        min-height: 44px !important;
    }
}

/* Desktop sidebar remains unchanged */
@media (min-width: 769px) {
    .css-1d391kg {
        min-width: 320px !important;
    }
}

/* Ensure consistent button styling across devices */
.stButton > button {
    white-space: nowrap !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# Mobile sidebar toggle button (only visible on mobile)
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("☰", key="mobile_sidebar_toggle", help="Toggle sidebar on mobile", 
                 use_container_width=False, type="secondary"):
        st.session_state["mobile_sidebar_open"] = not st.session_state.get("mobile_sidebar_open", False)

# Add JavaScript for mobile sidebar toggle
st.markdown("""
<script>
// Mobile sidebar toggle functionality
function toggleMobileSidebar() {
    const sidebar = document.querySelector('.css-1d391kg');
    const toggleBtn = document.querySelector('[data-testid="baseButton-secondary"]');
    
    if (sidebar) {
        if (sidebar.classList.contains('mobile-open')) {
            sidebar.classList.remove('mobile-open');
            if (toggleBtn) toggleBtn.textContent = '☰';
        } else {
            sidebar.classList.add('mobile-open');
            if (toggleBtn) toggleBtn.textContent = '✕';
        }
    }
}

// Add click event to toggle button
document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.querySelector('[data-testid="baseButton-secondary"]');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleMobileSidebar);
    }
});

// Close sidebar when clicking outside on mobile
document.addEventListener('click', function(e) {
    const sidebar = document.querySelector('.css-1d391kg.mobile-open');
    const toggleBtn = document.querySelector('[data-testid="baseButton-secondary"]');
    
    if (sidebar && !sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
        sidebar.classList.remove('mobile-open');
        if (toggleBtn) toggleBtn.textContent = '☰';
    }
});

// Handle mobile touch events
document.addEventListener('touchstart', function(e) {
    const sidebar = document.querySelector('.css-1d391kg.mobile-open');
    const toggleBtn = document.querySelector('[data-testid="baseButton-secondary"]');
    
    if (sidebar && !sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
        sidebar.classList.remove('mobile-open');
        if (toggleBtn) toggleBtn.textContent = '☰';
    }
});
</script>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar: Data + Filters + Change Options (restored layout)
# ============================================================
with st.sidebar:
    # Top status/caption
    st.caption(
        f"Loaded: **{st.session_state['data_name']}**  •  "
        f"<span class='demo-mode'>Demo data mode.</span>",
        unsafe_allow_html=True
    )

    # ---------- Data Source & Loader (TOP OF SIDEBAR) ----------
    st.markdown("### Data")
    src_changed = False

    # Find all Demo Data files in the current directory
    demo_data_files = []
    try:
        for file in os.listdir("."):
            if file.startswith("Demo Data") and file.endswith(".csv"):
                demo_data_files.append(file)
        demo_data_files.sort()  # Sort alphabetically
    except Exception:
        # Fallback if directory listing fails
        demo_data_files = ["Demo Data_Post Production (Small - 50 lines, 12 months).csv"]
    
    # Always default to "Demo Data_Post Production (Small - 50 lines, 12 months).csv"
    default_demo_file = "Demo Data_Post Production (Small - 50 lines, 12 months).csv"
    # If the default file is not in the list, use it anyway (or find the closest match)
    if default_demo_file not in demo_data_files:
        # Try to find a Post Production file with "12 months" in the name
        post_prod_12m = [f for f in demo_data_files if "Post Production" in f and "12 months" in f]
        if post_prod_12m:
            default_demo_file = sorted(post_prod_12m)[0]
        elif demo_data_files:
            default_demo_file = demo_data_files[0]
    
    # Get current data name
    current_data_name = st.session_state.get("data_name", default_demo_file)
    
    # Check if current data is a demo data file
    is_demo_data = current_data_name in demo_data_files
    has_last = "last_upload_df" in st.session_state and st.session_state["last_upload_df"] is not None
    is_last_upload = has_last and current_data_name == st.session_state.get("last_upload_name", "")
    
    # Create dropdown for demo data files
    if demo_data_files:
        # Determine the index for the selectbox
        if current_data_name in demo_data_files:
            default_index = demo_data_files.index(current_data_name)
        else:
            default_index = demo_data_files.index(default_demo_file) if default_demo_file in demo_data_files else 0
        
        selected_demo = st.selectbox(
            "Select Demo Data",
            options=demo_data_files,
            index=default_index,
            key="demo_data_selector"
        )
        
        # If a different demo file was selected, load it
        if selected_demo != current_data_name:
            with st.status(f"Loading {selected_demo}…", expanded=False):
                st.session_state["df_raw"] = smart_load_any(selected_demo)
                st.session_state["data_name"] = selected_demo
            st.session_state["changes"] = []
            key = _file_sig(selected_demo)
            st.session_state["accel"] = build_accelerators(st.session_state["df_raw"], FILTER_COLUMNS_ORDER, key)
            src_changed = True

    if src_changed:
        st.rerun()

    # Memory-only uploader (no temp files), with user-visible status
    st.markdown("**Or upload your own data:**")
    up = st.file_uploader("Browse and upload CSV file", type=["csv"], accept_multiple_files=False, disabled=False)
    if up is not None:
        with st.status(f"Loading {up.name}…", expanded=True) as s:
            s.write("Reading file bytes…")
            content = up.getvalue() if hasattr(up, "getvalue") else up.read()

            s.write("Parsing CSV…")
            # Load directly from memory; use pyarrow if present, else pandas default
            try:
                new_df = pd.read_csv(io.BytesIO(content), engine="pyarrow")
            except Exception:
                new_df = pd.read_csv(io.BytesIO(content))

            # Normalize types exactly like smart_load_any
            for c in DATE_COLUMNS:
                if c in new_df.columns:
                    new_df[c] = pd.to_datetime(new_df[c], errors="coerce")
            for c in FILTER_COLUMNS_ORDER:
                if c in new_df.columns:
                    new_df[c] = new_df[c].astype("string").fillna("").astype("category")
            tcols = [c for c in new_df.columns if _is_yyyymm(c)]
            if tcols:
                new_df[tcols] = new_df[tcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

            s.write("Rebuilding accelerators…")
            cache_key = _filehash_from_bytes(content)
            st.session_state["df_raw"] = new_df
            st.session_state["data_name"] = up.name
            st.session_state["changes"] = []
            st.session_state["accel"] = build_accelerators(new_df, FILTER_COLUMNS_ORDER, cache_key)

            # Remember as "Last Upload" so you can toggle back anytime
            st.session_state["last_upload_df"] = new_df.copy()
            st.session_state["last_upload_name"] = up.name
            st.session_state["last_upload_bytes"] = content

            s.update(label="Loaded ✔", state="complete")
        st.rerun()

    st.markdown("---")  # divider above the filters
    
    st.subheader("Filters")
    
    # Reset filters button at top of filters
    if st.button("Reset Filters", use_container_width=True):
        for col in FILTER_COLUMNS_ORDER:
            st.session_state[f"f_{col}"] = []
        st.session_state["pending_reset_filters"] = True
        st.rerun()

    # ---------- Dynamic Hierarchical Filters ----------
    selected = {}
    
    # Get current selections first (for dynamic filtering)
    current_selections = {}
    for col in FILTER_COLUMNS_ORDER:
        if col in df_raw.columns:
            current_selections[col] = st.session_state.get(f"f_{col}", [])
    
    # Get dynamic options based on current selections
    dynamic_options = get_dynamic_filter_options(df_raw, FILTER_COLUMNS_ORDER, current_selections)
    
    # Map column names to display names
    display_names = {
        "Resource_Type": "Resource Type",
        "Resource_Name": "Resource Name",
        "Department": "Department",
        "Team": "Team",
        "Project": "Project",
        "Workstream": "Workstream",
        "Account_Code": "Account"
    }
    
    for col in FILTER_COLUMNS_ORDER:
        if col in df_raw.columns:
            opts = dynamic_options.get(col, [])
            default = current_selections.get(col, [])
            
            # Filter out any default values that are no longer available
            valid_defaults = [v for v in default if v in opts]
            
            # Use display name for the label
            display_name = display_names.get(col, col)
            # Only pass default if the key doesn't exist in session state to avoid the error
            widget_key = f"f_{col}"
            if widget_key in st.session_state:
                # Key exists, don't pass default parameter
                selected_vals = st.multiselect(display_name, options=opts, key=widget_key)
            else:
                # Key doesn't exist, use default
                selected_vals = st.multiselect(display_name, options=opts, default=valid_defaults, key=widget_key)
            selected[col] = selected_vals

    # Scope based on filters
    mask_scope = scope_mask(df_raw, selected)
    df_current = df_raw[mask_scope].copy()
    df_current_spend = df_current[~is_headcount_row(df_current)].copy()

    st.markdown("---")

    # ---------- Change Options ----------
    st.subheader("Change Options")

    # Help text
    st.markdown(
        """
        <div class="co-help">
          <ul style="margin-top:0; margin-bottom:0.6rem; padding-left: 1.1rem;">
            <li><b>Shift</b>: moves the full curve for the filtered areas</li>
            <li><b>Scale</b>: increases or decreases filtered areas between the given start and end</li>
            <li><b>Extend</b>: copies selected date range and adds it at a chosen start location</li>
            <li><b>Remove</b>: removes data from selected date range</li>
            <li><b>Add Project</b>: copies an existing project and starts its curve at your chosen month</li>
            <li><b>Add Team</b>: copies an existing team and starts its curve at your chosen month</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Reset Options button (clears all change controls back to defaults) ----
    def _reset_change_options_state():
        for k in [
            "change_pick",
            "shift_months",
            "scale_mode", "pct_val", "scale_abs",
            "start_mm_sel", "end_mm_sel",
            "es_mode", "es_start", "es_end", "es_copy_start", "es_copy_end", "es_paste_start",
            "es_start_idx", "es_end_idx", "es_copy_start_idx", "es_copy_end_idx", "es_paste_start_idx",
            "addproj_name", "addproj_start",
            "addteam_name", "addteam_start",
        ]:
            st.session_state.pop(k, None)

    if st.button("Reset Options", use_container_width=True):
        _reset_change_options_state()
        st.toast("Change options reset", icon="✅")
        st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ---- Exactly one change type must be selected ----
    change_pick = st.radio(
        "Select one change to apply:",
        ["Shift", "Scale", "Extend", "Remove", "Add Project", "Add Team"],
        horizontal=False,
        key="change_pick",
        index=None,
    )

    # Small util for month formatting
    def _fmt_mm(mm: str) -> str:
        try:
            # Handle YYYY-MM format (including extended months like 2026-24)
            if '-' in mm:
                parts = mm.split('-')
                if len(parts) == 2:
                    year, month = int(parts[0]), int(parts[1])
                    # Handle extended periods: months 13-24 = months 1-12 of following year
                    if month > 12:
                        year += (month - 1) // 12
                        month = ((month - 1) % 12) + 1
                    return pd.to_datetime(f"{year}-{month:02d}-01", format="%Y-%m-%d").strftime("%m/%Y")
            # Handle YYYYMM format
            else:
                if len(mm) == 6 and mm.isdigit():
                    year, month = int(mm[:4]), int(mm[4:])
                    # Handle extended periods: months 13-24 = months 1-12 of following year
                    if month > 12:
                        year += (month - 1) // 12
                        month = ((month - 1) % 12) + 1
                    return pd.to_datetime(f"{year}{month:02d}01", format="%Y%m%d").strftime("%m/%Y")
        except Exception:
            return mm
    
    def create_time_range_selector(time_cols: List[str], start_key: str, end_key: str, label: str):
        """
        Create a time range selector using selectboxes with formatted labels.
        More user-friendly than scrolling through long dropdowns.
        """
        if not time_cols:
            return time_cols[0] if time_cols else "", time_cols[0] if time_cols else ""
        
        # Create formatted labels for display
        formatted_options = [_fmt_mm(mm) for mm in time_cols]
        
        # Get current values or defaults
        current_start_idx = st.session_state.get(f"{start_key}_idx", 0)
        current_end_idx = st.session_state.get(f"{end_key}_idx", len(time_cols) - 1)
        
        # Ensure current values are within bounds
        current_start_idx = max(0, min(current_start_idx, len(time_cols) - 1))
        current_end_idx = max(current_start_idx, min(current_end_idx, len(time_cols) - 1))
        
        st.markdown(f"**{label}**")
        
        # Create two columns for the selectors
        col1, col2 = st.columns(2)
        
        with col1:
            start_idx = st.selectbox(
                "Start",
                options=range(len(time_cols)),
                index=current_start_idx,
                key=f"{start_key}_selector",
                format_func=lambda x: formatted_options[x] if 0 <= x < len(formatted_options) else "Invalid"
            )
        
        with col2:
            end_idx = st.selectbox(
                "End",
                options=range(len(time_cols)),
                index=current_end_idx,
                key=f"{end_key}_selector",
                format_func=lambda x: formatted_options[x] if 0 <= x < len(formatted_options) else "Invalid"
            )
        
        # Ensure start <= end
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        # Store the indices for next time
        st.session_state[f"{start_key}_idx"] = start_idx
        st.session_state[f"{end_key}_idx"] = end_idx
        
        # Get the actual month values
        start_mm = time_cols[start_idx]
        end_mm = time_cols[end_idx]
        
        # Show the selected range
        st.caption(f"Selected: {_fmt_mm(start_mm)} to {_fmt_mm(end_mm)}")
        
        return start_mm, end_mm

    # ---- Conditionally render controls ONLY for the selected change ----

    # SHIFT
    if change_pick == "Shift":
        shift_months = st.slider(
            "Shift (months) — negative = left, positive = right",
            min_value=-12, max_value=12, value=0, step=1, key="shift_months"
        )

    # SCALE
    elif change_pick == "Scale":
        scale_mode = st.radio("Scaling mode", ["% change", "$ absolute"], horizontal=True, key="scale_mode")
        if scale_mode == "% change":
            pct_val = st.slider(
                "Percent change (−100% ➜ 0, +100% ➜ 2×)",
                min_value=-100, max_value=100, value=0, step=1, key="pct_val"
            )
        else:
            st.session_state["pct_val"] = 0
            scale_abs = st.number_input(
                "Absolute $ amount (distributed across selected months)",
                value=0.0, step=1000.0, format="%.2f", key="scale_abs"
            )

        # Get dynamic time options based on current filter selections
        dynamic_time_cols = get_dynamic_time_options(df_raw, selected, time_cols)
        if not dynamic_time_cols:
            dynamic_time_cols = time_cols  # fallback to all time columns
        
        # Use the new time range selector instead of dropdowns
        start_mm, end_mm = create_time_range_selector(
            dynamic_time_cols, 
            "start_mm_sel", 
            "end_mm_sel", 
            "Scale Window"
        )
        
        # Validation: Check if scale amount is specified
        if scale_mode == "% change" and st.session_state.get("pct_val", 0) == 0:
            st.info("ℹ️ Scale percentage is 0%. This will have no effect.")
        elif scale_mode == "$ absolute" and st.session_state.get("scale_abs", 0.0) == 0.0:
            st.info("ℹ️ Scale amount is $0. This will have no effect.")
        
        # Validation: Check if time range is valid
        if start_mm == end_mm:
            st.info("ℹ️ Start and end dates are the same. This will affect only one month.")

    # EXTEND
    elif change_pick == "Extend":
        # Get dynamic time options based on current filter selections
        dynamic_time_cols = get_dynamic_time_options(df_raw, selected, time_cols)
        if not dynamic_time_cols:
            dynamic_time_cols = time_cols  # fallback to all time columns
        
        st.markdown("**Select date range to copy:**")
        es_copy_start, es_copy_end = create_time_range_selector(
            dynamic_time_cols, 
            "es_copy_start", 
            "es_copy_end", 
            "Copy Range"
        )
        # Store indices for later use
        if es_copy_start in dynamic_time_cols:
            st.session_state["es_copy_start_idx"] = dynamic_time_cols.index(es_copy_start)
        if es_copy_end in dynamic_time_cols:
            st.session_state["es_copy_end_idx"] = dynamic_time_cols.index(es_copy_end)
        
        st.markdown("**Select where to paste (start date):**")
        # Get index for paste start (should be after copy end)
        copy_end_idx = dynamic_time_cols.index(es_copy_end) if es_copy_end in dynamic_time_cols else len(dynamic_time_cols) - 1
        # Ensure paste start is after copy end, but within valid bounds
        min_paste_idx = min(copy_end_idx + 1, len(dynamic_time_cols) - 1)
        saved_paste_idx = st.session_state.get("es_paste_start_idx", min_paste_idx)
        # Ensure the index is within valid bounds
        paste_start_idx = max(min_paste_idx, min(saved_paste_idx, len(dynamic_time_cols) - 1))
        # Final safety check: ensure index is valid
        paste_start_idx = max(0, min(paste_start_idx, len(dynamic_time_cols) - 1))
        es_paste_start = st.selectbox(
            "Paste Start Date",
            options=dynamic_time_cols,
            index=paste_start_idx,
            key="es_paste_start",
            format_func=_fmt_mm
        )
        st.session_state["es_paste_start_idx"] = dynamic_time_cols.index(es_paste_start)
        
        # Store for later use
        st.session_state["es_mode"] = "Extend"
        st.session_state["es_start"] = es_copy_start
        st.session_state["es_end"] = es_copy_end
        
        # Validation
        if es_copy_start == es_copy_end:
            st.info("ℹ️ Copy start and end dates are the same. This will copy only one month.")
    
    # REMOVE
    elif change_pick == "Remove":
        # Get dynamic time options based on current filter selections
        dynamic_time_cols = get_dynamic_time_options(df_raw, selected, time_cols)
        if not dynamic_time_cols:
            dynamic_time_cols = time_cols  # fallback to all time columns
        
        st.markdown("**Select date range to remove:**")
        es_start, es_end = create_time_range_selector(
            dynamic_time_cols, 
            "es_start", 
            "es_end", 
            "Remove Range"
        )
        
        # Store for later use
        st.session_state["es_mode"] = "Remove"
        
        # Validation
        if es_start == es_end:
            st.info("ℹ️ Start and end dates are the same. This will remove only one month.")

    # ADD PROJECT
    elif change_pick == "Add Project":
        proj_opts = sorted(df_raw["Project"].dropna().astype(str).unique().tolist()) if "Project" in df_raw.columns else []
        add_proj_name = st.selectbox("Project to copy", options=proj_opts, index=0 if proj_opts else None, key="addproj_name")
        
        # Get dynamic time options based on current filter selections
        dynamic_time_cols = get_dynamic_time_options(df_raw, selected, time_cols)
        if not dynamic_time_cols:
            dynamic_time_cols = time_cols  # fallback to all time columns
        
        add_proj_start = st.selectbox("Start Month", options=dynamic_time_cols, index=0, key="addproj_start", format_func=_fmt_mm)

    # ADD TEAM
    elif change_pick == "Add Team":
        team_opts = sorted(df_raw["Team"].dropna().astype(str).unique().tolist()) if "Team" in df_raw.columns else []
        add_team_name = st.selectbox("Team to copy", options=team_opts, index=0 if team_opts else None, key="addteam_name")
        
        # Get dynamic time options based on current filter selections
        dynamic_time_cols = get_dynamic_time_options(df_raw, selected, time_cols)
        if not dynamic_time_cols:
            dynamic_time_cols = time_cols  # fallback to all time columns
        
        add_team_start = st.selectbox("Start Month", options=dynamic_time_cols, index=0, key="addteam_start", format_func=_fmt_mm)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ---- Apply button: builds a single change dict according to 'change_pick' ----
    apply_clicked = st.button("Apply Change", type="primary", use_container_width=True, key="btn_apply")

# ============================================================
# Change Model & Apply
# ============================================================
@dataclass
class Change:
    id: str
    filters: Dict[str, List[str]]
    start_mm: str
    end_mm: str
    shift_months: int
    scale_pct: float
    scale_abs: float
    active: bool
    note: str
    es_mode: str = "None"
    es_start: str = ""
    es_end: str = ""
    # for Add Project and Add Team
    kind: str = "normal"
    addproj_name: str = ""
    addproj_start: str = ""
    addteam_name: str = ""
    addteam_start: str = ""

def month_slice_indices(time_cols_local: List[str], start_mm_local: str, end_mm_local: str) -> Tuple[int, int]:
    s = time_cols_local.index(start_mm_local)
    e = time_cols_local.index(end_mm_local)
    if s > e:
        s, e = e, s
    return s, e

def apply_shift(df: pd.DataFrame, row_mask: pd.Series, time_cols_local: List[str], shift: int) -> None:
    if shift == 0 or row_mask.sum() == 0:
        return
    block = df.loc[row_mask, time_cols_local].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")
    new_block = np.zeros_like(block)
    n = block.shape[1]
    if shift > 0:
        new_block[:, shift:n] = block[:, 0:n-shift]
    else:
        k = -shift
        new_block[:, 0:n-k] = block[:, k:n]
    df.loc[row_mask, time_cols_local] = new_block

def shift_dates(df: pd.DataFrame, row_mask: pd.Series, months: int) -> None:
    if months == 0 or row_mask.sum() == 0:
        return
    for c in DATE_COLUMNS:
        if c in df.columns:
            df.loc[row_mask, c] = pd.to_datetime(df.loc[row_mask, c], errors="coerce") + pd.DateOffset(months=months)

def apply_scale(df: pd.DataFrame, row_mask: pd.Series, time_cols_local: List[str],
                start_mm_local: str, end_mm_local: str, pct: float, abs_amount: float) -> None:
    if row_mask.sum() == 0: return
    s_idx, e_idx = month_slice_indices(time_cols_local, start_mm_local, end_mm_local)
    cols = time_cols_local[s_idx:e_idx+1]
    block = df.loc[row_mask, cols].to_numpy(dtype="float32")
    if pct != 0.0:
        block *= (1.0 + pct)
    if abs_amount != 0.0 and len(cols) > 0:
        block += (abs_amount / len(cols))
    df.loc[row_mask, cols] = block

def apply_extend(df: pd.DataFrame, row_mask: pd.Series, time_cols_local: List[str],
                 win_start: str, win_end: str, paste_start: str = None) -> None:
    if row_mask.sum() == 0: return
    s_idx, e_idx = month_slice_indices(time_cols_local, win_start, win_end)
    width = e_idx - s_idx + 1
    if width <= 0: return
    src_cols = time_cols_local[s_idx:e_idx+1]
    # Use paste_start if provided, otherwise default to after win_end
    if paste_start and paste_start in time_cols_local:
        dst_start = time_cols_local.index(paste_start)
    else:
        dst_start = e_idx + 1
    dst_end = min(dst_start + width - 1, len(time_cols_local)-1)
    if dst_start >= len(time_cols_local): return
    dst_cols = time_cols_local[dst_start:dst_end+1]
    
    debug_log(f"🔍 Debug: apply_extend - copying from {src_cols} to {dst_cols}")
    debug_log(f"🔍 Debug: apply_extend - source data sum: {df.loc[row_mask, src_cols].sum().sum():,.0f}")
    debug_log(f"🔍 Debug: apply_extend - destination data sum before: {df.loc[row_mask, dst_cols].sum().sum():,.0f}")
    
    src = df.loc[row_mask, src_cols].to_numpy(dtype="float32")[:, :len(dst_cols)]
    df.loc[row_mask, dst_cols] = df.loc[row_mask, dst_cols].to_numpy(dtype="float32") + src
    
    debug_log(f"🔍 Debug: apply_extend - destination data sum after: {df.loc[row_mask, dst_cols].sum().sum():,.0f}")
    
    # Adjust PRQ dates for extended data (move to new end of extended data)
    if "PRQ Date" in df.columns:
        # Get the last month of the extended data
        last_extended_month = time_cols_local[dst_end]
        last_extended_dt = pd.to_datetime(last_extended_month + "01", format="%Y%m%d")
        
        # Update PRQ dates for rows that were extended
        # Only update if the new end date is later than the current PRQ date
        current_prq_dates = pd.to_datetime(df.loc[row_mask, "PRQ Date"], errors="coerce")
        mask_update = current_prq_dates < last_extended_dt
        df.loc[row_mask & mask_update, "PRQ Date"] = last_extended_dt

def apply_shorten(df: pd.DataFrame, row_mask: pd.Series, time_cols_local: List[str],
                  win_start: str, win_end: str) -> None:
    if row_mask.sum() == 0: return
    s_idx, e_idx = month_slice_indices(time_cols_local, win_start, win_end)
    cols = time_cols_local[s_idx:e_idx+1]
    
    # Debug: Print what we're doing
    debug_log(f"🔍 Debug: Shortening columns {cols} from {win_start} to {win_end}")
    debug_log(f"🔍 Debug: Affecting {row_mask.sum()} rows")
    debug_log(f"🔍 Debug: Input df shape: {df.shape}")
    
    # Check what the data looks like BEFORE shortening
    before_sum = df.loc[row_mask, cols].sum().sum()
    debug_log(f"🔍 Debug: Total spending in target period BEFORE shorten: ${before_sum:,.0f}")
    
    # Set the specified columns to zero
    df.loc[row_mask, cols] = 0.0
    
    # Check what the data looks like AFTER shortening
    after_sum = df.loc[row_mask, cols].sum().sum()
    debug_log(f"🔍 Debug: Total spending in target period AFTER shorten: ${after_sum:,.0f}")
    
    # Debug: Check if the data was actually zeroed out
    if after_sum > 0:
        debug_log(f"🔍 Debug: WARNING - Data was not zeroed out! Still ${after_sum:,.0f}")
    else:
        debug_log(f"🔍 Debug: SUCCESS - Data was zeroed out correctly")
    
    # Debug: Check the overall impact on the dataframe
    overall_before = df[cols].sum().sum()
    debug_log(f"🔍 Debug: Overall spending in target period BEFORE shorten: ${overall_before:,.0f}")
    overall_after = df[cols].sum().sum()
    debug_log(f"🔍 Debug: Overall spending in target period AFTER shorten: ${overall_after:,.0f}")
    
    # Adjust TI/PO/PRQ dates for shortened data
    for date_col in DATE_COLUMNS:
        if date_col in df.columns:
            # Convert dates to datetime for comparison
            dates = pd.to_datetime(df.loc[row_mask, date_col], errors="coerce")
            win_start_dt = pd.to_datetime(win_start + "01", format="%Y%m%d")
            win_end_dt = pd.to_datetime(win_end + "01", format="%Y%m%d")
            
            if date_col == "TI Date":
                # Move TI dates inward to the new start of data (after the shortened period)
                mask_overlap = (dates >= win_start_dt) & (dates <= win_end_dt)
                df.loc[row_mask & mask_overlap, date_col] = win_end_dt + pd.DateOffset(months=1)
            elif date_col == "PRQ Date":
                # For shorten, move PRQ dates that are at or after the shortened period
                mask_overlap = (dates >= win_start_dt)  # Changed: include dates at or after start
                if mask_overlap.any():
                    debug_log(f"Adjusting PRQ dates for {mask_overlap.sum()} rows from {win_start_dt} to {win_start_dt - pd.DateOffset(months=1)}")
                df.loc[row_mask & mask_overlap, date_col] = win_start_dt - pd.DateOffset(months=1)
            elif date_col == "PO Date":
                # Move PO dates to the nearest location outside the removed area
                mask_overlap = (dates >= win_start_dt) & (dates <= win_end_dt)
                # Find the nearest valid date outside the shortened period
                for idx in df[row_mask & mask_overlap].index:
                    po_date = dates.loc[idx]
                    if pd.notna(po_date):
                        # Find the nearest date outside the shortened period
                        if po_date < win_start_dt:
                            # PO is before shortened period, keep as is
                            pass
                        else:
                            # PO is in or after shortened period, move to end of shortened period
                            df.loc[idx, date_col] = win_end_dt + pd.DateOffset(months=1)
    
    # Adjust Start_Month and End_Month for shortened data
    if "Start_Month" in df.columns:
        for idx in df[row_mask].index:
            start_mm = df.loc[idx, "Start_Month"]
            if pd.notna(start_mm):
                start_dt = yyyymm_to_dt(str(start_mm).strip())
                if pd.notna(start_dt):
                    win_start_dt = yyyymm_to_dt(win_start)
                    win_end_dt = yyyymm_to_dt(win_end)
                    # If start is in the removed range, move it to before the removed range
                    if win_start_dt <= start_dt <= win_end_dt:
                        # Move to month before removed range
                        new_start_dt = win_start_dt - pd.DateOffset(months=1)
                        df.loc[idx, "Start_Month"] = new_start_dt.strftime("%Y-%m")
                    elif start_dt > win_end_dt:
                        # If start is after removed range, shift it back by the removed duration
                        months_removed = (win_end_dt.year - win_start_dt.year) * 12 + (win_end_dt.month - win_start_dt.month) + 1
                        new_start_dt = start_dt - pd.DateOffset(months=months_removed)
                        df.loc[idx, "Start_Month"] = new_start_dt.strftime("%Y-%m")
    
    if "End_Month" in df.columns:
        for idx in df[row_mask].index:
            end_mm = df.loc[idx, "End_Month"]
            if pd.notna(end_mm):
                end_dt = yyyymm_to_dt(str(end_mm).strip())
                if pd.notna(end_dt):
                    win_start_dt = yyyymm_to_dt(win_start)
                    win_end_dt = yyyymm_to_dt(win_end)
                    # If end is in the removed range, move it to before the removed range
                    if win_start_dt <= end_dt <= win_end_dt:
                        # Move to month before removed range
                        new_end_dt = win_start_dt - pd.DateOffset(months=1)
                        df.loc[idx, "End_Month"] = new_end_dt.strftime("%Y-%m")
                    elif end_dt > win_end_dt:
                        # If end is after removed range, shift it back by the removed duration
                        months_removed = (win_end_dt.year - win_start_dt.year) * 12 + (win_end_dt.month - win_start_dt.month) + 1
                        new_end_dt = end_dt - pd.DateOffset(months=months_removed)
                        df.loc[idx, "End_Month"] = new_end_dt.strftime("%Y-%m")
                    elif end_dt < win_start_dt:
                        # If end is before removed range, keep as is
                        pass

def recalculate_project_dates_from_data(df: pd.DataFrame, time_cols: List[str]) -> None:
    """
    Recalculate Start_Month and End_Month for each row based on where actual non-zero data starts and ends.
    This is useful after removing data without filters, as it ensures project dates reflect the actual data range.
    """
    if "Start_Month" not in df.columns or "End_Month" not in df.columns:
        return
    
    # Convert time columns to numeric
    time_data = df[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    for idx in df.index:
        row_data = time_data.loc[idx]
        
        # Find first and last non-zero months
        non_zero_mask = row_data > 0.0001  # Use small threshold to account for floating point
        non_zero_indices = row_data.index[non_zero_mask].tolist()
        
        if non_zero_indices:
            # Get the first and last time columns with data
            first_month = non_zero_indices[0]
            last_month = non_zero_indices[-1]
            
            # Convert to datetime and format as YYYY-MM
            try:
                first_dt = yyyymm_to_dt(first_month)
                last_dt = yyyymm_to_dt(last_month)
                if pd.notna(first_dt):
                    df.loc[idx, "Start_Month"] = first_dt.strftime("%Y-%m")
                if pd.notna(last_dt):
                    df.loc[idx, "End_Month"] = last_dt.strftime("%Y-%m")
            except:
                pass

def apply_add_project(df_base: pd.DataFrame, df_work: pd.DataFrame,
                      project_name: str, start_mm: str, time_cols_local: List[str]) -> pd.DataFrame:
    debug_log(f"🔍 Debug: apply_add_project called with project: {project_name}, start: {start_mm}")
    debug_log(f"🔍 Debug: df_base has Project column: {'Project' in df_base.columns}")
    debug_log(f"🔍 Debug: df_base Project values: {df_base['Project'].astype(str).unique()[:5].tolist() if 'Project' in df_base.columns else 'N/A'}")
    
    if "Project" not in df_base.columns or not project_name:
        debug_log(f"🔍 Debug: Missing Project column or project name")
        return df_work
    src = df_base[df_base["Project"].astype(str) == project_name].copy()
    debug_log(f"🔍 Debug: Found {len(src)} rows for project '{project_name}'")
    if src.empty:
        debug_log(f"🔍 Debug: No rows found for project '{project_name}'")
        return df_work

    vals = src[time_cols_local].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")
    debug_log(f"🔍 Debug: Project time values sum: {vals.sum():,.0f}")
    if vals.sum() == 0.0:
        debug_log(f"🔍 Debug: Project has no time values, returning unchanged")
        return df_work
    col_nonzero = (vals.sum(axis=0) > 0.0)
    debug_log(f"🔍 Debug: Non-zero columns: {col_nonzero.sum()} out of {len(col_nonzero)}")
    try:
        src_first_idx = list(col_nonzero).index(True)
        debug_log(f"🔍 Debug: First non-zero column index: {src_first_idx}")
    except ValueError:
        debug_log(f"🔍 Debug: No non-zero columns found")
        return df_work

    tgt_idx = time_cols_local.index(start_mm)
    shift = tgt_idx - src_first_idx
    debug_log(f"🔍 Debug: Target index: {tgt_idx}, shift: {shift} months")

    shifted = np.zeros_like(vals)
    n = vals.shape[1]
    if shift > 0:
        shifted[:, shift:n] = vals[:, 0:n-shift]
        debug_log(f"🔍 Debug: Shifting right by {shift} months")
    elif shift < 0:
        k = -shift
        shifted[:, 0:n-k] = vals[:, k:n]
        debug_log(f"🔍 Debug: Shifting left by {k} months")
    else:
        shifted = vals.copy()
        debug_log(f"🔍 Debug: No shift needed")
    src[time_cols_local] = shifted

    for c in DATE_COLUMNS:
        if c in src.columns:
            src[c] = pd.to_datetime(src[c], errors="coerce") + pd.DateOffset(months=shift)

    debug_log(f"🔍 Debug: Concatenating {len(src)} rows to existing {len(df_work)} rows")
    result = pd.concat([df_work, src], ignore_index=True)
    debug_log(f"🔍 Debug: Final result shape: {result.shape}")
    return result

def apply_add_team(df_base: pd.DataFrame, df_work: pd.DataFrame,
                   team_name: str, start_mm: str, time_cols_local: List[str]) -> pd.DataFrame:
    debug_log(f"🔍 Debug: apply_add_team called with team: {team_name}, start: {start_mm}")
    debug_log(f"🔍 Debug: df_base has Team column: {'Team' in df_base.columns}")
    debug_log(f"🔍 Debug: df_base Team values: {df_base['Team'].astype(str).unique()[:5].tolist() if 'Team' in df_base.columns else 'N/A'}")
    
    if "Team" not in df_base.columns or not team_name:
        debug_log(f"🔍 Debug: Missing Team column or team name")
        return df_work
    src = df_base[df_base["Team"].astype(str) == team_name].copy()
    debug_log(f"🔍 Debug: Found {len(src)} rows for team '{team_name}'")
    if src.empty:
        debug_log(f"🔍 Debug: No rows found for team '{team_name}'")
        return df_work

    vals = src[time_cols_local].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")
    debug_log(f"🔍 Debug: Team time values sum: {vals.sum():,.0f}")
    if vals.sum() == 0.0:
        debug_log(f"🔍 Debug: Team has no time values, returning unchanged")
        return df_work
    col_nonzero = (vals.sum(axis=0) > 0.0)
    debug_log(f"🔍 Debug: Non-zero columns: {col_nonzero.sum()} out of {len(col_nonzero)}")
    try:
        src_first_idx = list(col_nonzero).index(True)
        debug_log(f"🔍 Debug: First non-zero column index: {src_first_idx}")
    except ValueError:
        debug_log(f"🔍 Debug: No non-zero columns found")
        return df_work

    tgt_idx = time_cols_local.index(start_mm)
    shift = tgt_idx - src_first_idx
    debug_log(f"🔍 Debug: Target index: {tgt_idx}, shift: {shift} months")

    shifted = np.zeros_like(vals)
    n = vals.shape[1]
    if shift > 0:
        shifted[:, shift:n] = vals[:, 0:n-shift]
        debug_log(f"🔍 Debug: Shifting right by {shift} months")
    elif shift < 0:
        k = -shift
        shifted[:, 0:n-k] = vals[:, k:n]
        debug_log(f"🔍 Debug: Shifting left by {k} months")
    else:
        shifted = vals.copy()
        debug_log(f"🔍 Debug: No shift needed")
    src[time_cols_local] = shifted

    for c in DATE_COLUMNS:
        if c in src.columns:
            src[c] = pd.to_datetime(src[c], errors="coerce") + pd.DateOffset(months=shift)

    debug_log(f"🔍 Debug: Concatenating {len(src)} rows to existing {len(df_work)} rows")
    result = pd.concat([df_work, src], ignore_index=True)
    debug_log(f"🔍 Debug: Final result shape: {result.shape}")
    return result

def describe_change(ch: Change) -> str:
    """
    Create detailed, specific change descriptions that include actual filter values and impact.
    """
    def format_filter_values(filters, key):
        """Format filter values for a specific key, showing up to 3 values"""
        vals = filters.get(key, [])
        if not vals:
            return ""
        if len(vals) == 1:
            return vals[0]
        elif len(vals) <= 3:
            return ", ".join(vals)
        else:
            return ", ".join(vals[:2]) + f" (+{len(vals)-2} more)"
    
    def get_scope_description(filters):
        """Create a detailed scope description from filters"""
        scope_parts = []
        
        # Priority order for scope description (most specific first)
        priority_keys = ["Project", "Group Initiative", "Cost Center", "CCRU", "Division", "Group", "Super Group"]
        
        for key in priority_keys:
            if key in filters and filters[key]:
                formatted_vals = format_filter_values(filters, key)
                if formatted_vals:
                    scope_parts.append(f"{key} {formatted_vals}")
                    # Stop after finding the most specific filter that has values
                    break
        
        # If no specific filters, show account if available
        if not scope_parts and "Account" in filters and filters["Account"]:
            formatted_vals = format_filter_values(filters, "Account")
            if formatted_vals:
                scope_parts.append(f"Account {formatted_vals}")
        
        # If still no scope, show any other filters
        if not scope_parts:
            for key in ["Account", "Category", "Forecast Status"]:
                if key in filters and filters[key]:
                    formatted_vals = format_filter_values(filters, key)
                    if formatted_vals:
                        scope_parts.append(f"{key} {formatted_vals}")
                        break
        
        return " and ".join(scope_parts) if scope_parts else "selected data"
    
    def format_date_range(start_mm, end_mm):
        """Format date range in a readable way"""
        try:
            start_date = pd.to_datetime(start_mm + "01", format="%Y%m%d").strftime("%b %Y")
            end_date = pd.to_datetime(end_mm + "01", format="%Y%m%d").strftime("%b %Y")
            if start_date == end_date:
                return start_date
            return f"{start_date} to {end_date}"
        except:
            return f"{start_mm} to {end_mm}"
    
    def calculate_impact_amount(df_base, filters, start_mm, end_mm, time_cols):
        """Calculate the dollar amount affected by this change"""
        try:
            # Create mask for the filtered data
            row_mask = fast_scope_mask(df_base, filters)
            if not row_mask.any():
                return 0.0, 0
            
            # Get the time columns in the specified range
            start_idx = time_cols.index(start_mm) if start_mm in time_cols else 0
            end_idx = time_cols.index(end_mm) if end_mm in time_cols else len(time_cols) - 1
            range_cols = time_cols[start_idx:end_idx + 1]
            
            # Calculate total amount in the filtered data for the specified range
            filtered_data = df_base[row_mask][range_cols]
            total_amount = filtered_data.apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum()
            rows_affected = row_mask.sum()
            
            return total_amount, rows_affected
        except Exception as e:
            debug_log(f"Error calculating impact amount: {e}")
            return 0.0, 0
    
    # Get detailed scope description
    scope_txt = get_scope_description(ch.filters)
    
    # Calculate impact amount for the change
    impact_amount = 0.0
    rows_affected = 0
    try:
        # Get the base dataframe from session state
        df_base = st.session_state.get("df_raw")
        if df_base is not None:
            impact_amount, rows_affected = calculate_impact_amount(df_base, ch.filters, ch.start_mm, ch.end_mm, time_cols)
    except Exception as e:
        debug_log(f"Error accessing df_raw for impact calculation: {e}")
    
    # Format impact amount
    impact_text = ""
    if impact_amount != 0:
        abs_amount = abs(impact_amount)
        if abs_amount >= 1_000_000:
            impact_text = f" (${abs_amount/1_000_000:.1f}M impacted, {rows_affected} rows)"
        else:
            impact_text = f" (${abs_amount:,.0f} impacted, {rows_affected} rows)"
    
    # Handle different change types with specific details
    if getattr(ch, "kind", "normal") == "add_project":
        # Handle both YYYYMM and YYYY-MM formats
        start_mm = str(ch.addproj_start).strip()
        try:
            if '-' in start_mm:
                start_date = pd.to_datetime(f"{start_mm}-01", format="%Y-%m-%d").strftime("%b %Y")
            else:
                start_date = pd.to_datetime(start_mm + "01", format="%Y%m%d").strftime("%b %Y")
        except:
            start_date = start_mm
        return f"Added project '{ch.addproj_name}' starting {start_date}"
    
    if getattr(ch, "kind", "normal") == "add_team":
        # Handle both YYYYMM and YYYY-MM formats
        start_mm = str(ch.addteam_start).strip()
        try:
            if '-' in start_mm:
                start_date = pd.to_datetime(f"{start_mm}-01", format="%Y-%m-%d").strftime("%b %Y")
            else:
                start_date = pd.to_datetime(start_mm + "01", format="%Y%m%d").strftime("%b %Y")
        except:
            start_date = start_mm
        return f"Added team '{ch.addteam_name}' starting {start_date}"
    
    # Build detailed description based on change type
    if ch.shift_months:
        direction = "right" if ch.shift_months > 0 else "left"
        months = abs(ch.shift_months)
        return f"Shifted {scope_txt} {direction} by {months} month{'s' if months != 1 else ''}{impact_text}"
    
    elif ch.scale_pct:
        date_range = format_date_range(ch.start_mm, ch.end_mm)
        pct_text = f"{int(ch.scale_pct*100):+d}%" if ch.scale_pct != 0 else "0%"
        return f"Scaled {scope_txt} by {pct_text} between {date_range}{impact_text}"
    
    elif ch.scale_abs:
        date_range = format_date_range(ch.start_mm, ch.end_mm)
        amount_text = f"${ch.scale_abs:,.0f}" if ch.scale_abs != 0 else "$0"
        return f"Scaled {scope_txt} by {amount_text} between {date_range}{impact_text}"
    
    elif ch.es_mode in ("Extend", "Remove"):
        date_range = format_date_range(ch.es_start or ch.start_mm, ch.es_end or ch.end_mm)
        if ch.es_mode == "Extend":
            paste_start = getattr(ch, "es_paste_start", ch.es_end)
            paste_date = _fmt_mm(paste_start) if hasattr(ch, "es_paste_start") else format_date_range(paste_start, paste_start)
            return f"Extended {scope_txt} by copying {date_range} to {paste_date}{impact_text}"
        else:  # Remove
            return f"Removed {scope_txt} data from {date_range}{impact_text}"
    
    # Fallback for any other cases
    date_range = format_date_range(ch.start_mm, ch.end_mm)
    return f"Modified {scope_txt} between {date_range}{impact_text}"

def fast_scope_mask(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.Series:
    """Fast scope mask function that can be used by describe_change"""
    acc = st.session_state.get("accel")
    if not acc or "mask_index" not in acc:
        m = pd.Series(True, index=df.index)
        for k, vals in filters.items():
            if k in df.columns and vals:
                m &= df[k].astype(str).isin(vals)
        return m
    base = np.ones(len(df), dtype=bool)
    for col, vals in filters.items():
        if not vals:
            continue
        col_map = acc["mask_index"].get(col, {})
        if not col_map:
            continue
        col_mask = np.zeros(len(df), dtype=bool)
        for v in vals:
            mv = col_map.get(str(v))
            if mv is not None:
                col_mask |= mv
        base &= col_mask
    return pd.Series(base, index=df.index)

def df_with_changes(df_base: pd.DataFrame, changes: List[Dict]) -> pd.DataFrame:
    df_n = df_base.copy()

    for ch in changes:
        if not ch.get("active", True):
            continue

        debug_log(f"🔍 Debug: Processing change: {ch.get('kind', 'unknown')} - {ch.get('es_mode', 'none')}")

        if ch.get("kind") == "add_project":
            debug_log(f"🔍 Debug: Processing Add Project change - project: {ch.get('addproj_name')}, start: {ch.get('addproj_start')}")
            debug_log(f"🔍 Debug: df_n shape before Add Project: {df_n.shape}")
            df_n = apply_add_project(df_base, df_n, ch.get("addproj_name", ""), ch.get("addproj_start", ""), time_cols)
            debug_log(f"🔍 Debug: df_n shape after Add Project: {df_n.shape}")
            continue
        
        if ch.get("kind") == "add_team":
            debug_log(f"🔍 Debug: Processing Add Team change - team: {ch.get('addteam_name')}, start: {ch.get('addteam_start')}")
            debug_log(f"🔍 Debug: df_n shape before Add Team: {df_n.shape}")
            df_n = apply_add_team(df_base, df_n, ch.get("addteam_name", ""), ch.get("addteam_start", ""), time_cols)
            debug_log(f"🔍 Debug: df_n shape after Add Team: {df_n.shape}")
            continue

        row_mask = fast_scope_mask(df_n, ch.get("filters", {}))

        # Apply shift operations
        shift_months = int(ch.get("shift_months", 0))
        if shift_months != 0:
            debug_log(f"🔍 Debug: Applying shift of {shift_months} months")
            apply_shift_months(df_n, row_mask, shift_months, time_cols)
            shift_phase_dates(df_n, row_mask, shift_months)

        # Apply scale operations
        scale_pct = float(ch.get("scale_pct", 0.0))
        scale_abs = float(ch.get("scale_abs", 0.0))
        if scale_pct != 0.0 or scale_abs != 0.0:
            debug_log(f"🔍 Debug: Applying scale: {scale_pct*100}% or ${scale_abs:,.0f}")
            apply_scale(df_n, row_mask, time_cols,
                        ch.get("start_mm", time_cols[0]), ch.get("end_mm", time_cols[-1]),
                        scale_pct, scale_abs)

        # Handle Extend/Shorten operations
        es_mode = ch.get("es_mode")
        debug_log(f"🔍 Debug: Change object keys: {list(ch.keys())}")
        debug_log(f"🔍 Debug: es_mode = '{es_mode}' (type: {type(es_mode)})")
        debug_log(f"🔍 Debug: es_mode == 'Extend': {es_mode == 'Extend'}")
        debug_log(f"🔍 Debug: es_mode == 'Shorten': {es_mode == 'Shorten'}")
        
        if es_mode == "Extend":
            debug_log(f"🔍 Debug: Applying Extend operation")
            paste_start = ch.get("es_paste_start") or ch.get("es_end") or time_cols[-1]
            apply_extend(df_n, row_mask, time_cols,
                         ch.get("es_start") or ch.get("start_mm", time_cols[0]),
                         ch.get("es_end")   or ch.get("end_mm",   time_cols[-1]),
                         paste_start=paste_start)
        elif es_mode == "Remove":
            debug_log(f"🔍 Debug: Applying Remove operation")
            apply_shorten(df_n, row_mask, time_cols,
                          ch.get("es_start") or ch.get("start_mm", time_cols[0]),
                          ch.get("es_end")   or ch.get("end_mm",   time_cols[-1]))
            # If no filters are set (removing all data), recalculate Start_Month and End_Month from actual data
            filters = ch.get("filters", {})
            if not any(filters.values()):
                debug_log(f"🔍 Debug: No filters set, recalculating Start_Month and End_Month from actual data")
                recalculate_project_dates_from_data(df_n, time_cols)
        else:
            debug_log(f"🔍 Debug: No Extend/Shorten operation found - es_mode is: '{es_mode}'")
        
        # Debug: Check if any other operations might be interfering
        debug_log(f"🔍 Debug: After Extend/Shorten operation - checking for interference")
        # Check if there are any other changes that might be affecting the same data
        for other_ch in changes:
            if other_ch != ch and other_ch.get("active", True):
                debug_log(f"🔍 Debug: Found other active change: {other_ch.get('kind')} - {other_ch.get('es_mode', 'none')}")
    return df_n

# When Apply is clicked, record change then reset the controls area
if 'changes' not in st.session_state:
    st.session_state['changes'] = []

if apply_clicked:
    # Build a change dict from the active selection
    if change_pick is None:
        st.warning("Pick a change type first (Shift, Scale, Extend, Remove, Add Project, or Add Team).")
        st.stop()

    # Generate a unique change ID
    if "change_seq" not in st.session_state:
        st.session_state["change_seq"] = 0
    st.session_state["change_seq"] += 1
    change_id = f"chg_{st.session_state['change_seq']}"

    change = {
        "id": change_id,
        "active": True,
        "filters": selected,            # the current sidebar filters define the scope
        "kind": None,                   # will set below
        "desc": "",                     # human-friendly description for the change log
    }

    if change_pick == "Shift":
        k = int(st.session_state.get("shift_months", 0))
        if k == 0:
            st.info("Shift of 0 months has no effect.")
            st.stop()
        change.update({
            "kind": "shift",
            "shift_months": k,
        })
        # Create detailed description using the enhanced function
        temp_change = Change(
            id=change_id,
            filters=selected,
            start_mm=time_cols[0],
            end_mm=time_cols[-1],
            shift_months=k,
            scale_pct=0.0,
            scale_abs=0.0,
            active=True,
            note=""
        )
        change["desc"] = describe_change(temp_change)

    elif change_pick == "Scale":
        mode = st.session_state.get("scale_mode", "% change")
        start_mm = st.session_state.get("start_mm_sel", time_cols[0])
        end_mm   = st.session_state.get("end_mm_sel", time_cols[-1])
        if mode == "% change":
            pct = float(st.session_state.get("pct_val", 0)) / 100.0
            change.update({
                "kind": "scale_pct",
                "scale_pct": pct,
                "start_mm": start_mm,
                "end_mm": end_mm,
            })
            # Create detailed description using the enhanced function
            temp_change = Change(
                id=change_id,
                filters=selected,
                start_mm=start_mm,
                end_mm=end_mm,
                shift_months=0,
                scale_pct=pct,
                scale_abs=0.0,
                active=True,
                note=""
            )
            change["desc"] = describe_change(temp_change)
        else:
            abs_amt = float(st.session_state.get("scale_abs", 0.0))
            change.update({
                "kind": "scale_abs",
                "scale_abs": abs_amt,
                "start_mm": start_mm,
                "end_mm": end_mm,
            })
            # Create detailed description using the enhanced function
            temp_change = Change(
                id=change_id,
                filters=selected,
                start_mm=start_mm,
                end_mm=end_mm,
                shift_months=0,
                scale_pct=0.0,
                scale_abs=abs_amt,
                active=True,
                note=""
            )
            change["desc"] = describe_change(temp_change)

    elif change_pick == "Extend":
        # Get the copy range and paste start from session state
        dynamic_time_cols = get_dynamic_time_options(df_raw, selected, time_cols)
        if not dynamic_time_cols:
            dynamic_time_cols = time_cols
        
        es_copy_start_idx = st.session_state.get("es_copy_start_idx", 0)
        es_copy_end_idx = st.session_state.get("es_copy_end_idx", len(dynamic_time_cols) - 1)
        # Ensure indices are within valid bounds
        es_copy_start_idx = max(0, min(es_copy_start_idx, len(dynamic_time_cols) - 1))
        es_copy_end_idx = max(0, min(es_copy_end_idx, len(dynamic_time_cols) - 1))
        
        min_paste_idx = min(es_copy_end_idx + 1, len(dynamic_time_cols) - 1)
        es_paste_start_idx = st.session_state.get("es_paste_start_idx", min_paste_idx)
        es_paste_start_idx = max(min_paste_idx, min(es_paste_start_idx, len(dynamic_time_cols) - 1))
        es_paste_start_idx = max(0, min(es_paste_start_idx, len(dynamic_time_cols) - 1))
        
        es_copy_start = dynamic_time_cols[es_copy_start_idx]
        es_copy_end = dynamic_time_cols[es_copy_end_idx]
        es_paste_start = dynamic_time_cols[es_paste_start_idx]
        
        change.update({
            "kind": "extend",
            "es_mode": "Extend",
            "es_start": es_copy_start,
            "es_end": es_copy_end,
            "es_paste_start": es_paste_start,
            "start_mm": es_copy_start,
            "end_mm": es_copy_end,
        })
        
        # Create description
        temp_change = Change(
            id=change_id,
            filters=selected,
            start_mm=es_copy_start,
            end_mm=es_copy_end,
            shift_months=0,
            scale_pct=0.0,
            scale_abs=0.0,
            active=True,
            note="",
            es_mode="Extend",
            es_start=es_copy_start,
            es_end=es_copy_end
        )
        temp_change.es_paste_start = es_paste_start
        change["desc"] = describe_change(temp_change)
        
    elif change_pick == "Remove":
        # Get the remove range from session state
        dynamic_time_cols = get_dynamic_time_options(df_raw, selected, time_cols)
        if not dynamic_time_cols:
            dynamic_time_cols = time_cols
        
        es_start_idx = st.session_state.get("es_start_idx", 0)
        es_end_idx = st.session_state.get("es_end_idx", len(dynamic_time_cols) - 1)
        
        es_start = dynamic_time_cols[es_start_idx]
        es_end = dynamic_time_cols[es_end_idx]
        
        change.update({
            "kind": "remove",
            "es_mode": "Remove",
            "es_start": es_start,
            "es_end": es_end,
            "start_mm": es_start,
            "end_mm": es_end,
        })
        
        # Create description
        temp_change = Change(
            id=change_id,
            filters=selected,
            start_mm=es_start,
            end_mm=es_end,
            shift_months=0,
            scale_pct=0.0,
            scale_abs=0.0,
            active=True,
            note="",
            es_mode="Remove",
            es_start=es_start,
            es_end=es_end
        )
        change["desc"] = describe_change(temp_change)

    elif change_pick == "Add Project":
        pname = st.session_state.get("addproj_name", "")
        pstart = st.session_state.get("addproj_start", time_cols[0])
        if not pname:
            st.warning("Pick a project to copy for Add Project.")
            st.stop()
        change.update({
            "kind": "add_project",
            "addproj_name": pname,
            "addproj_start": pstart,
        })
        # Create detailed description using the enhanced function
        temp_change = Change(
            id=change_id,
            filters=selected,
            start_mm=pstart,
            end_mm=pstart,
            shift_months=0,
            scale_pct=0.0,
            scale_abs=0.0,
            active=True,
            note="",
            kind="add_project",
            addproj_name=pname,
            addproj_start=pstart
        )
        change["desc"] = describe_change(temp_change)
    
    elif change_pick == "Add Team":
        tname = st.session_state.get("addteam_name", "")
        tstart = st.session_state.get("addteam_start", time_cols[0])
        if not tname:
            st.warning("Pick a team to copy for Add Team.")
            st.stop()
        change.update({
            "kind": "add_team",
            "addteam_name": tname,
            "addteam_start": tstart,
        })
        # Create detailed description using the enhanced function
        temp_change = Change(
            id=change_id,
            filters=selected,
            start_mm=tstart,
            end_mm=tstart,
            shift_months=0,
            scale_pct=0.0,
            scale_abs=0.0,
            active=True,
            note="",
            kind="add_team",
            addteam_name=tname,
            addteam_start=tstart
        )
        change["desc"] = describe_change(temp_change)

    # Record and rebuild
    st.session_state["changes"].append(change)

    # (Optional) brief toast
    st.toast("Change added. Charts updated.", icon="✨")

    # Reset the option controls so the next change starts clean
    _reset_change_options_state()

    # Set a flag to show debug info after rerun
    st.session_state["show_chart_debug"] = True

    # Rerun to render charts with the new change
    st.rerun()

# Apply all changes to baseline (with performance optimization)
with st.spinner("Processing changes..."):
    df_new_all = df_with_changes(df_raw, st.session_state["changes"])
    mask_scope_new = scope_mask(df_new_all, selected)  # recompute mask on the modified frame
    df_new = df_new_all[mask_scope_new].copy()
    show_new = any(c["active"] for c in st.session_state["changes"])

# Debug: Check if df_new_all was modified correctly
if st.session_state.get("changes") and any(c.get("es_mode") in ["Extend", "Shorten"] for c in st.session_state["changes"]):
    # Check if the changes are actually in df_new_all
    if "Group Initiative" in df_new_all.columns:
        falcon_mask = df_new_all["Group Initiative"] == "GI1- Falcon Connect"
        if falcon_mask.any():
            falcon_data = df_new_all[falcon_mask]
            target_cols = ['202706', '202707', '202708', '202709', '202710', '202711', '202712', '202801', '202802', '202803', '202804', '202805', '202806']
            if all(col in falcon_data.columns for col in target_cols):
                falcon_sum = falcon_data[target_cols].sum().sum()
                debug_log(f"🔍 Debug: df_new_all creation - Falcon Connect target period sum: ${falcon_sum:,.0f}")
                
                # Also check the overall sum for these months
                overall_sum = df_new_all[target_cols].sum().sum()
                debug_log(f"🔍 Debug: df_new_all creation - Overall target period sum: ${overall_sum:,.0f}")

# Debug: Show what changes were applied
if st.session_state.get("changes") and any(c.get("es_mode") in ["Extend", "Shorten"] for c in st.session_state["changes"]):
    debug_log("Extend/Shorten changes detected. Check if they appear in charts below.")
    # Show a simple indicator that changes are applied
    st.success("✅ Changes applied successfully! Check the charts below to see the impact.")
    
    # Debug: Check what data the charts are using
    debug_log(f"df_new_all shape: {df_new_all.shape}")
    debug_log(f"df_new shape: {df_new.shape}")
    debug_log(f"show_new: {show_new}")
    
    # Check if the changes are actually in df_new_all
    if "Group Initiative" in df_new_all.columns:
        falcon_mask = df_new_all["Group Initiative"] == "GI1- Falcon Connect"
        if falcon_mask.any():
            falcon_data = df_new_all[falcon_mask]
            target_cols = ['202706', '202707', '202708', '202709', '202710', '202711', '202712', '202801', '202802', '202803', '202804', '202805', '202806']
            if all(col in falcon_data.columns for col in target_cols):
                falcon_sum = falcon_data[target_cols].sum().sum()
                debug_log(f"Falcon Connect data in df_new_all for target period: ${falcon_sum:,.0f}")
            else:
                debug_log("Target columns not found in Falcon Connect data")
        else:
            debug_log("Falcon Connect not found in df_new_all")

# Show chart debug info if flag is set
if st.session_state.get("show_chart_debug"):
    debug_log(f"AFTER RERUN - show_new: {show_new}")
    debug_log(f"AFTER RERUN - df_new_all shape: {df_new_all.shape}")
    # Clear the flag
    st.session_state["show_chart_debug"] = False

df_current_spend = df_current[~is_headcount_row(df_current)].copy()
df_new_spend = df_new[~is_headcount_row(df_new)].copy() if show_new else df_new

# Expose for Ask tab
st.session_state["df_current"] = df_current
st.session_state["df_new_all"] = df_new_all
st.session_state["mask_scope"] = mask_scope

# ============================================================
# Main Navigation Tabs
# ============================================================
tab1 = st.tabs(["📊 Dashboard"])[0]  # Only show Dashboard tab

with tab1:
    # ============================================================
    # Resource View Toggle (Top Center)
    # ============================================================
    col_left, col_center, col_right = st.columns([0.3, 0.4, 0.3])
    with col_center:
        resource_view = st.radio(
            "",
            ["Spend", "FTE"],
            horizontal=True,
            key="resource_view_toggle",
            index=0
        )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    
    # ============================================================
    # Buttons Row (YoY Trend left, buttons right)
    # ============================================================
    tg_left, spacer_mid, btn_right = st.columns([0.55, 0.20, 0.25])
    with tg_left:
        tg_c1, tg_c2 = st.columns([0.45, 0.55])
        with tg_c1:
            target_on = st.toggle("MoM Growth Trend", value=False)
            st.markdown('<div class="yoy-compact">', unsafe_allow_html=True)
            mom_whole = st.number_input("MoM %", min_value=-100, max_value=500, value=5, step=1, key="yoy_input")
            st.markdown('</div>', unsafe_allow_html=True)
        with tg_c2:
            st.empty()
        growth_rate = (mom_whole or 0) / 100.0

        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Clear WIF", use_container_width=True):
                st.session_state["pending_clear"] = True
                st.rerun()
        with b2:
            pop = st.popover("Save and Export", use_container_width=True)
            with pop:
                export_name = st.text_input("Name (no extension)", value="EasyWIF_Scenario")
                
                # Prepare data for exports
                df0 = st.session_state["df_raw"]
                dfN = df_new_all
                # df0 = original (df_raw), dfN = new (df_new_all) after applying changes
                a0 = df0[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                aN = dfN[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

                # Align on index & columns so shapes match even if row counts differ
                a0_aln, aN_aln = a0.align(aN, join="outer", axis=0, fill_value=0.0)

                # Compare for any month changed
                diffs = (aN_aln.values != a0_aln.values)
                changed_mask = diffs.any(axis=1)
                changed_idx = aN_aln.index[changed_mask]
                
                # Create full dataset with UPDATED column
                df_full_export = dfN.copy()
                df_full_export["UPDATED"] = df_full_export.index.isin(changed_idx)
                
                # Prepare change log for export
                changes = st.session_state.get("changes", [])
                change_log_data = []
                for idx, ch in enumerate(changes, 1):
                    change_log_data.append({
                        "Change #": idx,
                        "Description": ch.get("desc", ""),
                        "Active": ch.get("active", True),
                        "Note": ch.get("note", "")
                    })
                df_change_log = pd.DataFrame(change_log_data)
                
                st.markdown("---")
                st.markdown("**Download New Data**")
                if df_full_export.empty:
                    st.info("No data to export yet.")
                else:
                    try:
                        import openpyxl  # noqa: F401
                        xbuf = io.BytesIO()
                        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
                            # Sheet 1: Change Log
                            if not df_change_log.empty:
                                df_change_log.to_excel(writer, index=False, sheet_name="Change Log")
                            else:
                                pd.DataFrame({"Message": ["No changes recorded"]}).to_excel(writer, index=False, sheet_name="Change Log")
                            # Sheet 2: Full dataset with NEW DATA
                            df_full_export.to_excel(writer, index=False, sheet_name="Data")
                        xbuf.seek(0)
                        st.download_button("Download .xlsx", data=xbuf,
                            file_name=f"{export_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"Excel export unavailable ({e}). Use CSV below.")
                    
                    # CSV export (single file with change log as header comment, then data)
                    csv_parts = []
                    if not df_change_log.empty:
                        csv_parts.append("# Change Log\n")
                        csv_parts.append(df_change_log.to_csv(index=False))
                        csv_parts.append("\n# Full Dataset with NEW DATA\n")
                    csv_parts.append(df_full_export.to_csv(index=False))
                    csv_bytes = "".join(csv_parts).encode("utf-8")
                    st.download_button("Download .csv", data=csv_bytes,
                        file_name=f"{export_name}.csv", mime="text/csv",
                        use_container_width=True
                    )
                
                st.markdown("---")
                st.markdown("**Download Visual**")
                # Collect all available figures
                figs = {
                    "Total Resources by Month": st.session_state.get("fig_year"),
                    "5-Year Spending Profile": st.session_state.get("fig_area"),
                    "Gantt": st.session_state.get("fig_gantt"),
                    "Resources by Dimension": st.session_state.get("fig_dept"),
                }
                # Add Burden & GSD if available
                if st.session_state.get("fig_bgsd") is not None:
                    figs["Burden & GSD"] = st.session_state.get("fig_bgsd")
                html_parts = [f"<h1>EasyWIF — {export_name}</h1>", "<h3>Summary</h3>"]
                for title, fig in figs.items():
                    if fig is not None:
                        html_parts.append(f"<h4>{title}</h4>")
                        html_parts.append(pio.to_html(fig, include_plotlyjs='cdn', full_html=False))
                
                df_delta = st.session_state.get("df_delta")
                if df_delta is not None:
                    html_parts.append("<h3>Delta Table — Current vs New</h3>")
                    html_parts.append(df_delta.to_html(index=False))
                
                # Add Change Log to HTML
                html_parts.append("<h3>Change Log</h3>")
                if changes:
                    html_parts.append("<ul>")
                    for ch in changes:
                        desc = ch.get("desc", "")
                        note = ch.get("note", "")
                        active = ch.get("active", True)
                        status = "✓ Active" if active else "✗ Inactive"
                        if note:
                            html_parts.append(f"<li><strong>{status}:</strong> {desc} <em>({note})</em></li>")
                        else:
                            html_parts.append(f"<li><strong>{status}:</strong> {desc}</li>")
                    html_parts.append("</ul>")
                else:
                    html_parts.append("<p>No changes recorded.</p>")
                
                # Create comprehensive dark theme CSS
                dark_theme_css = """
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        margin: 20px; 
                        background-color: #000000;
                        color: #ffffff;
                    }
                    h1 { 
                        color: #ffffff; 
                        margin-top: 20px;
                        margin-bottom: 20px;
                    }
                    h3 { 
                        color: #ffffff; 
                        margin-top: 30px;
                        margin-bottom: 15px;
                    }
                    h4 { 
                        color: #ffffff; 
                        margin-top: 20px;
                        margin-bottom: 10px;
                    }
                    ul { 
                        line-height: 1.6;
                        color: #ffffff;
                    }
                    li {
                        color: #ffffff;
                        margin-bottom: 8px;
                    }
                    p {
                        color: #ffffff;
                    }
                    strong {
                        color: #ffffff;
                    }
                    em {
                        color: #e0e0e0;
                    }
                    /* Style tables with dark theme */
                    table {
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                        background-color: #1a1a1a;
                        color: #ffffff;
                    }
                    th {
                        background-color: #2a2a2a;
                        color: #ffffff;
                        padding: 12px;
                        text-align: left;
                        border: 1px solid #444444;
                        font-weight: bold;
                    }
                    td {
                        background-color: #1a1a1a;
                        color: #ffffff;
                        padding: 10px;
                        border: 1px solid #444444;
                    }
                    tr:nth-child(even) {
                        background-color: #252525;
                    }
                    tr:hover {
                        background-color: #333333;
                    }
                    /* Ensure plotly charts maintain their styling */
                    .js-plotly-plot {
                        background-color: transparent;
                    }
                </style>
                """
                report_html = f"<html><head><meta charset='utf-8'>{dark_theme_css}</head><body>" + "\n".join(html_parts) + "</body></html>"
                st.download_button("Download HTML", data=report_html.encode("utf-8"),
                                   file_name=f"{export_name}.html", mime="text/html", use_container_width=True)

    st.markdown("---")

    # ============================================================
    # Resource Profile
    # ============================================================
    debug_log(f"🔍 Debug: ENTERING PROFILE CHART SECTION")
    debug_log(f"🔍 Debug: About to generate profile chart - show_new: {show_new}")
    # Filter by Resource_Type based on resource_view toggle
    resource_view_profile = st.session_state.get("resource_view_toggle", "Spend")
    if "Resource_Type" in df_current.columns:
        if resource_view_profile == "Spend":
            df_current_profile = df_current[df_current["Resource_Type"] == "Spend"].copy()
            df_new_profile = df_new[df_new["Resource_Type"] == "Spend"].copy() if show_new else df_current_profile
            df_new_all_profile = df_new_all[df_new_all["Resource_Type"] == "Spend"].copy() if show_new else df_current_profile
        else:  # FTE - show Headcount data
            df_current_profile = df_current[df_current["Resource_Type"] == "Headcount"].copy()
            df_new_profile = df_new[df_new["Resource_Type"] == "Headcount"].copy() if show_new else df_current_profile
            df_new_all_profile = df_new_all[df_new_all["Resource_Type"] == "Headcount"].copy() if show_new else df_current_profile
    else:
        df_current_profile = df_current.copy()
        df_new_profile = df_new.copy() if show_new else df_current_profile
        df_new_all_profile = df_new_all.copy() if show_new else df_current_profile

    st.markdown('<div class="tight-title">Resource Profile</div>', unsafe_allow_html=True)

    # Get all available time columns (including extended data from changes)
    all_time_cols_profile = set(time_cols)
    if show_new and df_new_all_profile is not None:
        new_cols = [c for c in df_new_all_profile.columns if _is_yyyymm(c)]
        all_time_cols_profile.update(new_cols)
    if df_current_profile is not None:
        current_cols = [c for c in df_current_profile.columns if _is_yyyymm(c)]
        all_time_cols_profile.update(current_cols)
    all_time_cols_profile = sorted(list(all_time_cols_profile), key=lambda x: yyyymm_to_dt(x))

    cur_series_all = total_series(df_current_profile, all_time_cols_profile).astype(float)
    # Ensure cur_series_all is indexed by all_time_cols_profile
    if not cur_series_all.index.equals(pd.Index(all_time_cols_profile)):
        cur_series_all = cur_series_all.reindex(all_time_cols_profile, fill_value=0.0)

    all_months_sorted = pd.Series([yyyymm_to_dt(mm) for mm in all_time_cols_profile], index=all_time_cols_profile)
    valid_idx2 = all_months_sorted.notna()
    # Use numpy array indexing to avoid index alignment issues
    valid_mask = valid_idx2.values
    all_months_sorted = all_months_sorted.iloc[valid_mask]
    cur_series_all = pd.Series(cur_series_all.values[valid_mask], index=all_months_sorted.index)

    # Get new series data if available
    new_series_all = None
    if show_new:
        debug_log("Generating 'New' profile chart trace")
        # Use df_new_profile (filtered data) when filters are active, df_new_all_profile (full data) when no filters
        if any(selected.values()):  # If any filters are active
            new_series_all = total_series(df_new_profile, all_time_cols_profile).astype(float)
            debug_log("Using filtered data (df_new_profile) for profile chart")
        else:
            new_series_all = total_series(df_new_all_profile, all_time_cols_profile).astype(float)
            debug_log("Using full data (df_new_all_profile) for profile chart")
        
        # Ensure new_series_all is indexed by all_time_cols_profile
        if not new_series_all.index.equals(pd.Index(all_time_cols_profile)):
            new_series_all = new_series_all.reindex(all_time_cols_profile, fill_value=0.0)
        new_series_all = pd.Series(new_series_all.values[valid_mask], index=all_months_sorted.index)
        
        # Debug: Check what data the profile chart is using
        debug_log(f"Profile chart - new_series_all sum: ${new_series_all.sum():,.0f}")

    # Find the actual data range - include all months with non-zero data
    all_series_data = {}
    for i, month in enumerate(all_months_sorted):
        val = cur_series_all.iloc[i] if i < len(cur_series_all) else 0.0
        if val > 0 or (new_series_all is not None and i < len(new_series_all) and new_series_all.iloc[i] > 0):
            all_series_data[month] = val

    if new_series_all is not None:
        for i, month in enumerate(all_months_sorted):
            if i < len(new_series_all):
                val = new_series_all.iloc[i]
                if val > 0 or month in all_series_data:
                    all_series_data[month] = max(all_series_data.get(month, 0), val)

    # Get the actual date range from data (not just DISPLAY_YEARS)
    if all_series_data:
        data_months = sorted(all_series_data.keys())
        x_min_data = data_months[0]
        x_max_data = data_months[-1]
        # Add a small buffer (1 month on each side)
        x_min = x_min_data - pd.DateOffset(months=1)
        x_max = x_max_data + pd.DateOffset(months=1)
    else:
        x_min = all_months_sorted.min() if len(all_months_sorted) > 0 else X_MIN
        x_max = all_months_sorted.max() if len(all_months_sorted) > 0 else X_MAX

    # Filter series to only show months within the data range (but include all data, not just 5 years)
    data_range_mask = (all_months_sorted >= x_min) & (all_months_sorted <= x_max)
    cur_series = cur_series_all[data_range_mask]
    months_display = all_months_sorted[data_range_mask]

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=months_display, y=cur_series, fill='tozeroy', mode='lines',
        name='Current', line=dict(width=2, color=COLOR_CURRENT), line_shape='linear'
    ))
    if show_new and new_series_all is not None:
        new_series = new_series_all[data_range_mask]
        fig_area.add_trace(go.Scatter(
            x=months_display, y=new_series, fill='tozeroy', mode='lines',
            name='New', line=dict(width=2, color=COLOR_NEW, dash='dot'), line_shape='linear'
        ))
    else:
        debug_log("show_new is False, not generating 'New' profile chart trace")
    if target_on:
        debug_log(f"MoM Growth calculation - growth_rate: {growth_rate}")
        
        # Calculate MoM (Month-over-Month) trend - apply growth rate to each month
        if len(cur_series) > 0:
            first_month_val = float(cur_series.iloc[0])
            debug_log(f"MoM Growth calculation - first month value: {first_month_val}")
            
            if first_month_val > 0:
                # Calculate monthly targets based on MoM growth rate
                targets = [first_month_val]
                for i in range(1, len(months_display)):
                    targets.append(targets[-1] * (1.0 + growth_rate))
                
                debug_log(f"MoM Growth calculation - targets array length: {len(targets)}")
                
                if len(targets) > 0 and len(months_display) > 0:
                    fig_area.add_trace(go.Scatter(
                        x=months_display, y=targets,
                        mode="lines", name="MoM Trend",
                        line=dict(color=COLOR_TARGET, width=1.5),
                        hovertemplate="Target: %{y:.0f}<extra></extra>"
                    ))
                else:
                    debug_log("MoM Growth calculation - No valid targets")
            else:
                debug_log("MoM Growth calculation - First month value is 0, skipping MoM trend")
        else:
            debug_log("MoM Growth calculation - No data available, skipping MoM trend")

    # Set y-axis label based on resource_view
    y_axis_label_profile = "FTE" if resource_view_profile == "FTE" else "Spending ($)"

    # Auto-scale y-axis: find the actual data range from all displayed data
    all_data = list(cur_series.values)
    if show_new and new_series_all is not None:
        new_series_display = new_series_all[data_range_mask]
        all_data.extend(new_series_display.values)
    if all_data:
        y_max = max(all_data) * 1.1 if max(all_data) > 0 else 1.0
        y_min = 0
    else:
        y_max = 1.0
        y_min = 0

    fig_area.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title=None, yaxis_title=y_axis_label_profile,
        xaxis=dict(
            tickfont=dict(size=13), 
            tickformat="%b '%y",  # Format: Jan '26, Feb '26, etc
            dtick="M1",  # Show every month
            range=[x_min, x_max],
            tickangle=-45
        ),
        yaxis=dict(range=[y_min, y_max], tickfont=dict(size=13)),
        legend=dict(
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0
        ),
        font=dict(size=13),
        margin=dict(t=40, r=10, l=10, b=50),
        height=320
    )
    st.plotly_chart(fig_area, use_container_width=True)
    st.session_state["fig_area"] = fig_area

    st.markdown("---")

    # ============================================================
    # Project View
    # ============================================================
    st.markdown('<div class="tight-title">Project View</div>', unsafe_allow_html=True)

    need = {"Project", "Start_Month", "End_Month", "Workstream"}
    if not need.issubset(df_current.columns):
        st.info("Gantt needs columns: Project, Start_Month, End_Month, Workstream.")
    else:
        # Use df_new (filtered data) when filters are active, df_new_all (full data) when no filters
        if show_new:
            if any(selected.values()):  # If any filters are active
                source_df = df_new
                debug_log("Using filtered data (df_new) for Gantt chart")
            else:
                source_df = df_new_all
                debug_log("Using full data (df_new_all) for Gantt chart")
        else:
            source_df = df_current
        
        # Convert Start_Month and End_Month to datetime
        def parse_month_date(month_str):
            """Parse MM/YYYY or YYYY-MM format to datetime. Handles extended periods where months 13-24 represent next year."""
            if pd.isna(month_str):
                return pd.NaT
            month_str = str(month_str).strip()
            try:
                # Try MM/YYYY format first
                if '/' in month_str:
                    parts = month_str.split('/')
                    if len(parts) == 2:
                        month, year = int(parts[0]), int(parts[1])
                        # Handle extended periods: months 13-24 = months 1-12 of following year
                        if month > 12:
                            year += (month - 1) // 12
                            month = ((month - 1) % 12) + 1
                        return pd.to_datetime(f"{year}-{month:02d}-01", format="%Y-%m-%d", errors="coerce")
                # Try YYYY-MM format
                elif '-' in month_str:
                    parts = month_str.split('-')
                    if len(parts) == 2:
                        year, month = int(parts[0]), int(parts[1])
                        # Handle extended periods: months 13-24 = months 1-12 of following year
                        if month > 12:
                            year += (month - 1) // 12
                            month = ((month - 1) % 12) + 1
                        return pd.to_datetime(f"{year}-{month:02d}-01", format="%Y-%m-%d", errors="coerce")
            except:
                pass
            return pd.NaT
        
        gantt_df = source_df[["Project", "Start_Month", "End_Month", "Workstream"]].copy()
        gantt_df["Start_Date"] = gantt_df["Start_Month"].apply(parse_month_date)
        gantt_df["End_Date"] = gantt_df["End_Month"].apply(parse_month_date)
        
        # Group by Project and Workstream, taking min start and max end
        agg = (
            gantt_df.groupby(["Project", "Workstream"])
            .agg({"Start_Date": "min", "End_Date": "max"})
            .reset_index()
        )
        
        agg = agg.dropna(subset=["Start_Date", "End_Date"])
        if agg.empty:
            st.write("No projects match current filters.")
        else:
            agg = agg.sort_values("Start_Date", ascending=True)
            order = agg["Project"].tolist()
            all_workstreams = sorted(df_raw["Workstream"].dropna().astype(str).unique()) \
                              if "Workstream" in df_raw.columns else sorted(agg["Workstream"].astype(str).unique())
            palette = px.colors.qualitative.Set2
            color_map = {ws: palette[i % len(palette)] for i, ws in enumerate(all_workstreams)}
            
            # Auto-scale x-axis based on data
            x_min = agg["Start_Date"].min() if len(agg) > 0 else X_MIN
            x_max = agg["End_Date"].max() if len(agg) > 0 else X_MAX
            
            fig_g = px.timeline(
                agg, x_start="Start_Date", x_end="End_Date", y="Project",
                color="Workstream", template=PLOTLY_TEMPLATE,
                color_discrete_map=color_map
            )
            
            fig_g.update_layout(
                xaxis=dict(
                    range=[x_min, x_max], 
                    tickformat="%b '%y",  # Format: Jan '26, Feb '26, etc
                    dtick="M1",
                    title=None, 
                    type="date", 
                    tickfont=dict(size=12),
                    tickangle=-45
                ),
                yaxis=dict(
                    title=None, 
                    categoryorder="array", 
                    categoryarray=order, 
                    automargin=True,
                    tickfont=dict(size=13)
                ),
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="left", 
                    x=0.0,
                    title_text="Workstream", 
                    font=dict(size=13)
                ),
                template=PLOTLY_TEMPLATE,
                font=dict(size=13),
                height=max(400, min(30 * len(order), 1200)),
                margin=dict(t=60, r=20, b=80, l=10),
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.session_state["fig_gantt"] = fig_g

    st.markdown("---")

    # ============================================================
    # Summary Row (2 columns): Total Resources by Month, Resources by Dimension
    # ============================================================
    
    # Filter by Resource_Type based on resource_view toggle (for both charts)
    resource_view = st.session_state.get("resource_view_toggle", "Spend")
    if "Resource_Type" in df_current.columns:
        if resource_view == "Spend":
            df_current_filtered = df_current[df_current["Resource_Type"] == "Spend"].copy()
            df_new_filtered = df_new[df_new["Resource_Type"] == "Spend"].copy() if show_new else df_current_filtered
            df_new_all_filtered = df_new_all[df_new_all["Resource_Type"] == "Spend"].copy() if show_new else df_current_filtered
        else:  # FTE - show Headcount data
            df_current_filtered = df_current[df_current["Resource_Type"] == "Headcount"].copy()
            df_new_filtered = df_new[df_new["Resource_Type"] == "Headcount"].copy() if show_new else df_current_filtered
            df_new_all_filtered = df_new_all[df_new_all["Resource_Type"] == "Headcount"].copy() if show_new else df_current_filtered
    else:
        df_current_filtered = df_current.copy()
        df_new_filtered = df_new.copy() if show_new else df_current_filtered
        df_new_all_filtered = df_new_all.copy() if show_new else df_current_filtered
    
    # Helper function to get all time columns from dataframes (including extended data)
    def get_all_time_columns() -> List[str]:
        """Get all time columns from current and new data, including any extended columns"""
        all_time_cols = set(time_cols)  # Start with original time columns
        
        # Add any time columns from df_new_all that might have been added via Extend
        if show_new and df_new_all_filtered is not None:
            new_cols = [c for c in df_new_all_filtered.columns if _is_yyyymm(c)]
            all_time_cols.update(new_cols)
        
        # Also check df_current_filtered for any additional time columns
        if df_current_filtered is not None:
            current_cols = [c for c in df_current_filtered.columns if _is_yyyymm(c)]
            all_time_cols.update(current_cols)
        
        # Sort the columns chronologically
        all_time_cols_list = sorted(list(all_time_cols), key=lambda x: yyyymm_to_dt(x))
        return all_time_cols_list
    
    # Helper function to get time window column indices (defined here before use)
    def get_time_window_columns(time_window: str, time_cols_local: List[str], df_data: pd.DataFrame = None) -> List[str]:
        """Convert time window selection to list of column names"""
        if time_window == "Entire Horizon":
            # For Entire Horizon, use all available time columns including extended data
            return get_all_time_columns()
        
        # Get current month index (first column with data, or first column if no data check)
        current_idx = 0
        if df_data is not None and len(df_data) > 0:
            # Find first column with non-zero data
            for i, col in enumerate(time_cols_local):
                if col in df_data.columns:
                    col_sum = df_data[col].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
                    if col_sum > 0:
                        current_idx = i
                        break
        
        if time_window == "Current Month":
            return [time_cols_local[current_idx]] if current_idx < len(time_cols_local) else []
        elif time_window == "Next 3 Months":
            end_idx = min(current_idx + 3, len(time_cols_local))
            return time_cols_local[current_idx:end_idx]
        elif time_window == "Next 6 Months":
            end_idx = min(current_idx + 6, len(time_cols_local))
            return time_cols_local[current_idx:end_idx]
        elif time_window == "Next 12 Months":
            end_idx = min(current_idx + 12, len(time_cols_local))
            return time_cols_local[current_idx:end_idx]
        else:
            return time_cols_local
    
    # Get time columns for selected window (use combined data to find current month)
    df_combined = pd.concat([df_current_filtered, df_new_all_filtered if show_new else df_current_filtered], ignore_index=True)
    
    # Get all available time columns (including extended data)
    all_available_time_cols = get_all_time_columns()
    
    # Time Window Dropdown at top center (tight and clean)
    col_time_left, col_time_center, col_time_right = st.columns([0.4, 0.2, 0.4])
    with col_time_center:
        time_window = st.selectbox(
            "Select Time Window",
            ["Current Month", "Next 3 Months", "Next 6 Months", "Next 12 Months", "Entire Horizon"],
            key="time_window_selector",
            index=1,  # Default to "Next 3 Months"
            label_visibility="visible"
        )
    
    # Add spacing between time window dropdown and graph titles
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    window_cols = get_time_window_columns(time_window, all_available_time_cols, df_combined)
    
    sum_left, g1, sum_right = st.columns([0.49, 0.02, 0.49])

@st.cache_data(ttl=300)  # Cache for 5 minutes
def year_totals(df_: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    df_ = df_[~is_headcount_row(df_)]
    m = df_[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    months_all = pd.to_datetime([yyyymm_to_dt(mm) for mm in time_cols])
    s = pd.DataFrame({"month": months_all, "total": m.sum(axis=0).to_numpy()})
    s["year"] = s["month"].dt.year
    out = s[s["year"].isin(years)].groupby("year", as_index=False)["total"].sum()
    for y in years:
        if y not in out["year"].values:
            out = pd.concat([out, pd.DataFrame({"year":[y], "total":[0.0]})], ignore_index=True)
    
    # Debug: Check what the year_totals function is returning
    if any(c.get("es_mode") in ["Extend", "Shorten"] for c in st.session_state.get("changes", [])):
        debug_log(f"year_totals function - input df shape: {df_.shape}")
        debug_log(f"year_totals function - output: {out['total'].tolist()}")
        
        # Check specific months that should be affected
        target_months = ['202706', '202707', '202708', '202709', '202710', '202711', '202712', '202801', '202802', '202803', '202804', '202805', '202806']
        if all(col in df_.columns for col in target_months):
            target_sum = df_[target_months].sum().sum()
            debug_log(f"year_totals function - target months sum: ${target_sum:,.0f}")
    
    return out.sort_values("year")

# Helper function to get all years that have data in the dataframe
def get_all_years_with_data(df_: pd.DataFrame) -> List[int]:
    """Dynamically detect all years that have data in the dataframe"""
    m = df_[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    months_all = pd.to_datetime([yyyymm_to_dt(mm) for mm in time_cols])
    s = pd.DataFrame({"month": months_all, "total": m.sum(axis=0).to_numpy()})
    s["year"] = s["month"].dt.year
    years_with_data = s[s["total"] > 0]["year"].unique().tolist()
    return sorted(years_with_data) if years_with_data else DISPLAY_YEARS

# Helper function for year totals without headcount filtering (used when Resource_Type filtering is applied)
def year_totals_no_filter(df_: pd.DataFrame, years: List[int] = None) -> pd.DataFrame:
    if years is None:
        years = get_all_years_with_data(df_)
    m = df_[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    months_all = pd.to_datetime([yyyymm_to_dt(mm) for mm in time_cols])
    s = pd.DataFrame({"month": months_all, "total": m.sum(axis=0).to_numpy()})
    s["year"] = s["month"].dt.year
    out = s[s["year"].isin(years)].groupby("year", as_index=False)["total"].sum()
    for y in years:
        if y not in out["year"].values:
            out = pd.concat([out, pd.DataFrame({"year":[y], "total":[0.0]})], ignore_index=True)
    return out.sort_values("year")

with sum_left:
    st.markdown('<div class="tight-title">Total Resources by Month</div>', unsafe_allow_html=True)
    
    # Get month-over-month data for the selected time window
    # Current data
    cur_monthly = df_current_filtered[window_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    cur_dates = [yyyymm_to_dt(mm) for mm in window_cols]
    cur_df = pd.DataFrame({
        "Month": cur_dates,
        "Total": cur_monthly.values,
        "Series": "Current"
    })
    
    # New data
    if show_new:
        if any(selected.values()):
            new_monthly = df_new_filtered[window_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
        else:
            new_monthly = df_new_all_filtered[window_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
        new_dates = [yyyymm_to_dt(mm) for mm in window_cols]
        new_df = pd.DataFrame({
            "Month": new_dates,
            "Total": new_monthly.values,
            "Series": "New"
        })
        month_long = pd.concat([cur_df, new_df], ignore_index=True)
    else:
        month_long = cur_df
    
    # Format month labels for x-axis
    month_long["Month_Label"] = month_long["Month"].dt.strftime("%b '%y")
    
    # For FTE, don't divide; for Spend, divide by 1K
    if resource_view == "FTE":
        month_long = month_long.assign(Total_Display = month_long["Total"])
        text_template = "%{y:.1f}"
        y_axis_label = "FTE"
    else:
        month_long = month_long.assign(Total_Display = month_long["Total"] / 1_000.0)
        text_template = "%{y:.1f}K"
        y_axis_label = "$K"
    
    # Create line/area chart for month-over-month
    fig_month = px.bar(
        month_long, x="Month_Label", y="Total_Display", color="Series", barmode="group",
        template=PLOTLY_TEMPLATE, color_discrete_map={"Current": COLOR_CURRENT, "New": COLOR_NEW}
    )
    
    # Auto-scale y-axis
    y_max = month_long["Total_Display"].max() * 1.1 if len(month_long) > 0 else 1.0
    y_min = 0
    
    # Adjust data label font size based on number of months (smaller for Entire Horizon)
    num_months = len(month_long["Month_Label"].unique())
    if num_months > 24:  # Entire Horizon or very long range
        label_font_size = 8
        tick_font_size = 9
    elif num_months > 12:
        label_font_size = 9
        tick_font_size = 10
    else:
        label_font_size = 11
        tick_font_size = 11
    
    fig_month.update_layout(
        xaxis_title=None, yaxis_title=y_axis_label,
        xaxis=dict(tickfont=dict(size=tick_font_size), tickangle=-45, showgrid=False),
        yaxis=dict(tickfont=dict(size=13), showgrid=True, gridcolor='rgba(128,128,128,0.2)', range=[y_min, y_max]),
        legend=dict(
            title_text="",  # Remove legend title
            font=dict(size=12),
            orientation="h",
            yanchor="top",
            y=-0.35,  # Move legend further down to create more space
            xanchor="center",
            x=0.5
        ),
        font=dict(size=13),
        bargap=0.18,
        margin=dict(t=40, r=10, l=10, b=130),  # Increase bottom margin to accommodate legend
        height=310,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_month.update_traces(texttemplate=text_template, textposition="outside", cliponaxis=False, textfont=dict(size=label_font_size))

    st.plotly_chart(fig_month, use_container_width=True)
    st.session_state["fig_year"] = fig_month

with sum_right:
    # Dimension selector
    dimension_options = []
    if "Department" in df_current_filtered.columns:
        dimension_options.append("Department")
    if "Team" in df_current_filtered.columns:
        dimension_options.append("Team")
    if "Project" in df_current_filtered.columns:
        dimension_options.append("Project")
    
    if not dimension_options:
        st.info("No dimension columns (Department, Team, Project) found in data.")
    else:
        # Title with inline radio buttons for Group by
        title_col_left, title_col_right = st.columns([0.6, 0.4])
        with title_col_left:
            st.markdown('<div class="tight-title">Resources by Category</div>', unsafe_allow_html=True)
        with title_col_right:
            selected_dimension = st.radio(
                "",
                dimension_options,
                key="dimension_selector",
                index=0,
                horizontal=True,
                label_visibility="collapsed"
            )
        
        # Use only New scenario data (or Current if no changes)
        if show_new:
            if any(selected.values()):
                df_dim_data = df_new_filtered.copy()
            else:
                df_dim_data = df_new_all_filtered.copy()
        else:
            df_dim_data = df_current_filtered.copy()
        
        # Group by selected dimension and sum only the window columns
        dim_grouped = df_dim_data.groupby(selected_dimension)[window_cols].sum().sum(axis=1).sort_values(ascending=False)
        
        # Filter to only dimensions with data
        dims_with_data = dim_grouped[dim_grouped > 0].index.tolist()
        if dims_with_data:
            dim_grouped = dim_grouped[dim_grouped.index.isin(dims_with_data)]
        else:
            dim_grouped = pd.Series(dtype=float)
        
        if len(dim_grouped) > 0:
            dim_df = pd.DataFrame({
                selected_dimension: dim_grouped.index.tolist(),
                "Total": dim_grouped.values
            })
            
            # For FTE, don't divide; for Spend, divide by 1K
            if resource_view == "FTE":
                dim_df = dim_df.assign(Total_Display = dim_df["Total"].fillna(0))
                text_template = "%{y:.1f}"
                y_axis_label = "FTE"
            else:
                # Handle division and avoid NaN - ensure Total is numeric and fill NaN with 0
                dim_df["Total"] = pd.to_numeric(dim_df["Total"], errors="coerce").fillna(0)
                dim_df = dim_df.assign(Total_Display = dim_df["Total"] / 1_000.0)
                # Replace any NaN or inf values with 0
                dim_df["Total_Display"] = dim_df["Total_Display"].replace([np.inf, -np.inf, np.nan], 0.0)
                text_template = "%{y:.1f}K"
                y_axis_label = "$K"
            
            # Filter out any rows with invalid Total_Display values before creating chart
            dim_df = dim_df[dim_df["Total_Display"].notna() & (dim_df["Total_Display"] != np.inf) & (dim_df["Total_Display"] >= 0)]
            
            if len(dim_df) > 0:
                # Create horizontal bar chart with dashboard theme colors
                # Build color map for each category using a diverse color palette
                category_list = dim_df[selected_dimension].tolist()
                color_map = {}
                # Use a palette that provides distinct colors for each category
                # Mix of theme colors and complementary shades
                color_palette = [
                    COLOR_CURRENT,           # Blue
                    COLOR_NEW,               # Green
                    COLOR_CURRENT_BURDEN,    # Light Blue
                    COLOR_NEW_BURDEN,        # Light Green
                    "#ff7f0e",               # Orange
                    "#d62728",               # Red
                    "#9467bd",               # Purple
                    "#8c564b",               # Brown
                    "#e377c2",               # Pink
                    "#7f7f7f",               # Gray
                    "#bcbd22",               # Olive
                    "#17becf",               # Cyan
                ]
                for i, cat in enumerate(category_list):
                    # Cycle through the palette to ensure each category gets a unique color
                    color_map[cat] = color_palette[i % len(color_palette)]
                
                fig_dim = px.bar(
                    dim_df, x="Total_Display", y=selected_dimension, orientation='h',
                    template=PLOTLY_TEMPLATE,
                    color=selected_dimension,
                    color_discrete_map=color_map
                )
                
                # Auto-scale x-axis
                x_max = dim_df["Total_Display"].max() * 1.1 if len(dim_df) > 0 and dim_df["Total_Display"].max() > 0 else 1.0
                x_min = 0
                
                fig_dim.update_layout(
                    xaxis_title=y_axis_label, yaxis_title=None,
                    xaxis=dict(tickfont=dict(size=13), showgrid=True, gridcolor='rgba(128,128,128,0.2)', range=[x_min, x_max]),
                    yaxis=dict(tickfont=dict(size=12), showgrid=False),
                    showlegend=False,
                    font=dict(size=13),
                    margin=dict(t=40, r=10, l=10, b=50),
                    height=310,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                # Only show text labels for non-zero, non-NaN values to avoid NaNK
                # Ensure all values are valid numbers
                valid_labels = []
                for val in dim_df["Total_Display"]:
                    if pd.notna(val) and val > 0 and not np.isinf(val):
                        if resource_view == "FTE":
                            valid_labels.append(f"{val:.1f}")
                        else:
                            valid_labels.append(f"{val:.1f}K")
                    else:
                        valid_labels.append("")
                
                fig_dim.update_traces(
                    texttemplate=valid_labels,
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=11)
                )
            
            st.plotly_chart(fig_dim, use_container_width=True)
            st.session_state["fig_dept"] = fig_dim
        else:
            st.info(f"No data found for {selected_dimension} in the selected time window.")


    # Add spacing and horizontal separator before Change Log
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    st.markdown("---")

# ============================================================
# Change Log
# ============================================================

# Backfill IDs for any legacy changes missing 'id'
for idx, ch in enumerate(st.session_state.get("changes", [])):
    if "id" not in ch:
        if "change_seq" not in st.session_state:
            st.session_state["change_seq"] = 0
        st.session_state["change_seq"] += 1
        ch["id"] = f"chg_{st.session_state['change_seq']}"

with st.expander("📝 Change Log", expanded=True):
    if len(st.session_state.get("changes", [])) == 0:
        st.write("No changes yet. Use the sidebar to apply one.")
    else:
        remove_ids: List[str] = []
        for ch in st.session_state["changes"]:
            c1, c2, c3 = st.columns([0.75, 0.15, 0.10])
            with c1:
                st.write(ch.get("desc",""))
            with c2:
                ch["active"] = st.checkbox("Show/Hide", value=ch["active"], key=f"active_{ch['id']}")
            with c3:
                if st.button("✖ Remove", key=f"remove_{ch['id']}"):
                    remove_ids.append(ch["id"])
        if remove_ids:
            st.session_state["changes"] = [c for c in st.session_state["changes"] if c["id"] not in remove_ids]
            st.success("Removed selected change(s).")

# ============================================================
# AI Insights Tab and AI Chat - DISABLED
# ============================================================
# Note: AI Insights tab and AI chat functionality have been disabled
# but the code is preserved for future use