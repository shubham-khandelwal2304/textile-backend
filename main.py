"""
Dora Dori AI - Inventory Intelligence Backend
Python FastAPI service for data analysis and processing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dora Dori AI - Inventory Intelligence API",
    description="Data analysis service for inventory dashboard",
    version="1.0.0"
)

# CORS middleware to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InventoryDataRequest(BaseModel):
    """Request model for raw inventory data from n8n"""
    data: List[Dict[str, Any]]


class DashboardArrayRequest(BaseModel):
    """Request model for array of dashboard objects from n8n"""
    dashboards: Optional[List[Dict[str, Any]]] = None
    data: Optional[List[Dict[str, Any]]] = None  # Accept 'data' field for n8n compatibility


class DashboardResponse(BaseModel):
    """Response model for analyzed dashboard data"""
    overview: Dict[str, Any]
    top_skus: List[Dict[str, Any]]
    fabric_status: List[Dict[str, Any]]
    ad_performance: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]


def format_currency(value: float) -> str:
    """Format number as Indian Rupee currency"""
    return f"₹{value:,.0f}".replace(",", ",")


def calculate_overview_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overview metrics from inventory data
    Replicates SQL query logic using pandas
    """
    # Filter active styles only
    active_df = df[df.get('active_flag', True) == True].copy()
    
    if active_df.empty:
        return {
            "total_active_styles": 0,
            "total_active_styles_change": 0,
            "fabrics_that_require_replenishment": 0,
            "fabrics_that_require_replenishment_change": 0,
            "avg_days_of_cover": 0,
            "avg_days_of_cover_change": 0,
            "total_units_sold": 0,
            "total_units_sold_change": 0,
            "avg_return_rate": 0.0,
            "avg_return_rate_change": 0.0,
            "styles_need_replenishment": 0,
            "styles_need_replenishment_change": 0,
            "styles_broken": 0,
            "styles_broken_change": 0,
            "styles_out_of_stock": 0,
            "styles_out_of_stock_change": 0
        }
    
    # 1. Total Active Styles
    total_active_styles = active_df['style_id'].nunique()
    
    # 2. Fabrics that require replenishment (checking all 3 fabric types)
    fabrics_need_replenishment = active_df[
        (active_df.get('fabric_1_available_meters', 0) < active_df.get('fabric_1_reorder_point', 0)) |
        (active_df.get('fabric_2_available_meters', 0) < active_df.get('fabric_2_reorder_point', 0)) |
        (active_df.get('fabric_3_available_meters', 0) < active_df.get('fabric_3_reorder_point', 0))
    ]['style_id'].nunique()
    
    # 3. Avg Days of Cover
    active_df['total_available_qty'] = (
        active_df.get('myntra_available_qty', 0) + 
        active_df.get('nykaa_available_qty', 0)
    )
    active_df['daily_sales'] = active_df.get('total_units_sold', 0) / 30
    active_df['days_of_cover'] = np.where(
        (active_df['total_available_qty'] > 0) & (active_df['daily_sales'] > 0),
        active_df['total_available_qty'] / active_df['daily_sales'],
        0
    )
    avg_days_of_cover = round(active_df['days_of_cover'].mean(), 0) if not active_df['days_of_cover'].empty else 0
    
    # 4. Total Units Sold (30d)
    total_units_sold = round(active_df.get('total_units_sold', 0).sum(), 0)
    
    # 5. Avg Return Rate
    avg_return_rate = round(active_df.get('return_rate_avg', 0).mean(), 2) if not active_df.empty else 0.0
    
    # 6. Styles that need replenishment (using default reorder point of 10)
    styles_need_replenishment = active_df[
        (active_df.get('myntra_available_qty', 0) < 10) |
        (active_df.get('nykaa_available_qty', 0) < 10)
    ]['style_id'].nunique()
    
    # 7. Styles broken (Size Broken)
    styles_broken = active_df[
        (active_df.get('myntra_size_curve_status', '') == 'broken') |
        (active_df.get('nykaa_size_curve_status', '') == 'broken') |
        (active_df.get('broken_size_curve', False) == True)
    ]['style_id'].nunique()
    
    # 8. Styles out of stock (Nykaa or Myntra)
    styles_out_of_stock = active_df[
        (active_df.get('myntra_available_qty', 0) < 10) |
        (active_df.get('nykaa_available_qty', 0) < 10)
    ]['style_id'].nunique()
    
    # For now, trend changes are set to 0 (can be enhanced with historical data comparison)
    return {
        "total_active_styles": int(total_active_styles),
        "total_active_styles_change": 0,
        "fabrics_that_require_replenishment": int(fabrics_need_replenishment),
        "fabrics_that_require_replenishment_change": 0,
        "avg_days_of_cover": int(avg_days_of_cover),
        "avg_days_of_cover_change": 0,
        "total_units_sold": int(total_units_sold),
        "total_units_sold_change": 0,
        "avg_return_rate": float(avg_return_rate),
        "avg_return_rate_change": 0.0,
        "styles_need_replenishment": int(styles_need_replenishment),
        "styles_need_replenishment_change": 0,
        "styles_broken": int(styles_broken),
        "styles_broken_change": 0,
        "styles_out_of_stock": int(styles_out_of_stock),
        "styles_out_of_stock_change": 0
    }


def calculate_top_skus(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Calculate top performing SKUs by sell-through rate
    """
    # Filter styles with sales
    sales_df = df[df.get('total_units_sold', 0) > 0].copy()
    
    if sales_df.empty:
        return []
    
    # Sort by sell-through rate descending
    top_skus = sales_df.nlargest(limit, 'sell_through_rate', keep='all')
    
    result = []
    for _, row in top_skus.iterrows():
        result.append({
            "style_name": str(row.get('style_name', 'Unknown')),
            "ad_platform": str(row.get('ad_platform', 'Unknown')),
            "sell_through_rate": round(float(row.get('sell_through_rate', 0)), 0),
            "roas": round(float(row.get('roas', 0)), 1)
        })
    
    return result


def calculate_fabric_status(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calculate fabric status for all fabric types
    Replicates SQL query logic using pandas
    """
    fabric_status_list = []
    
    # Process fabric_1
    fabric_1_df = df[df.get('fabric_1_type', '').notna() & (df.get('fabric_1_type', '') != '')].copy()
    if not fabric_1_df.empty:
        for fabric_type in fabric_1_df['fabric_1_type'].unique():
            fabric_rows = fabric_1_df[fabric_1_df['fabric_1_type'] == fabric_type]
            available = float(fabric_rows['fabric_1_available_meters'].iloc[0]) if not fabric_rows.empty else 0
            reorder_point = float(fabric_rows['fabric_1_reorder_point'].iloc[0]) if not fabric_rows.empty else 0
            
            if reorder_point > 0:
                percent_consumed = round((available / reorder_point) * 100, 0)
            else:
                percent_consumed = 0
            
            # Determine status
            if available < reorder_point * 0.8:
                status = 'LOW'
            elif available < reorder_point:
                status = 'WARNING'
            else:
                status = 'GOOD'
            
            fabric_status_list.append({
                "fabric_type": str(fabric_type),
                "available": float(available),
                "reorder_point": float(reorder_point),
                "percent_consumed": float(percent_consumed),
                "status": status
            })
    
    # Process fabric_2 and fabric_3 similarly if needed
    # (Currently only fabric_1 is used in the SQL query)
    
    # Sort by status (LOW, WARNING, GOOD)
    status_order = {'LOW': 0, 'WARNING': 1, 'GOOD': 2}
    fabric_status_list.sort(key=lambda x: status_order.get(x['status'], 3))
    
    return fabric_status_list


def calculate_ad_performance(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calculate ad performance metrics by platform
    """
    # Filter rows with ad platform data
    ad_df = df[df.get('ad_platform', '').notna() & (df.get('ad_platform', '') != '')].copy()
    
    if ad_df.empty:
        return []
    
    # Group by ad platform
    ad_performance = ad_df.groupby('ad_platform').agg({
        'ad_spend': 'sum',
        'roas': 'mean'
    }).reset_index()
    
    # Sort by total spend descending
    ad_performance = ad_performance.sort_values('ad_spend', ascending=False)
    
    result = []
    for _, row in ad_performance.iterrows():
        result.append({
            "ad_platform": str(row['ad_platform']),
            "total_spend": round(float(row['ad_spend']), 0),
            "avg_roas": round(float(row['roas']), 2)
        })
    
    return result


def calculate_alerts(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Calculate alerts based on various conditions
    """
    active_df = df[df.get('active_flag', True) == True].copy()
    
    alerts = []
    
    for _, row in active_df.iterrows():
        alert_type = None
        severity = None
        message = None
        
        # Check for low stock on Myntra
        if row.get('alert_low_stock_myntra', False) == True:
            alert_type = 'Low Stock'
            severity = 'critical'
            message = row.get('alert_message_summary', 'Low stock alert on Myntra')
        
        # Check for fabric reorder
        elif row.get('alert_fabric_reorder', False) == True:
            alert_type = 'Fabric Reorder'
            severity = 'critical'
            message = row.get('alert_message_summary', 'Fabric reorder required')
        
        # Check for high return rate
        elif row.get('alert_high_return_rate', False) == True:
            alert_type = 'High Return'
            severity = 'warning'
            message = row.get('alert_message_summary', 'High return rate detected')
        
        if alert_type:
            alerts.append({
                "style_id": str(row.get('style_id', '')),
                "style_name": str(row.get('style_name', 'Unknown')),
                "platform": str(row.get('ad_platform', 'Unknown')),
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "date": (datetime.now() - timedelta(days=np.random.randint(0, 5))).strftime('%Y-%m-%d')
            })
    
    # Sort by severity (critical first) and date
    severity_order = {'critical': 0, 'warning': 1, 'stable': 2}
    alerts.sort(key=lambda x: (severity_order.get(x['severity'], 3), x['date']), reverse=True)
    
    return alerts[:limit]


def aggregate_dashboard_data(dashboard_array: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate an array of dashboard objects into a single consolidated dashboard using pandas
    This function performs data analysis and aggregation on multiple dashboard results
    """
    if not dashboard_array or len(dashboard_array) == 0:
        return {
            "overview": {},
            "top_skus": [],
            "fabric_status": [],
            "ad_performance": [],
            "alerts": []
        }
    
    # Extract all dashboard objects
    dashboards = [item.get('dashboard') for item in dashboard_array if item.get('dashboard')]
    
    if not dashboards:
        return {
            "overview": {},
            "top_skus": [],
            "fabric_status": [],
            "ad_performance": [],
            "alerts": []
        }
    
    logger.info(f"Aggregating {len(dashboards)} dashboard objects")
    
    # Convert to DataFrame for easier aggregation
    overviews_df = pd.DataFrame([d.get('overview', {}) for d in dashboards])
    
    # Aggregate Overview Metrics using pandas
    def safe_sum(series_name):
        if series_name in overviews_df.columns:
            return int(overviews_df[series_name].sum())
        return 0
    
    def safe_avg(series_name):
        if series_name in overviews_df.columns:
            valid_values = overviews_df[overviews_df[series_name] > 0][series_name]
            if len(valid_values) > 0:
                return valid_values.mean()
        return 0
    
    overview = {
        "total_active_styles": safe_sum('total_active_styles'),
        "total_active_styles_change": 0,
        "fabrics_that_require_replenishment": safe_sum('fabrics_that_require_replenishment'),
        "fabrics_that_require_replenishment_change": 0,
        "avg_days_of_cover": int(safe_avg('avg_days_of_cover')),
        "avg_days_of_cover_change": 0,
        "total_units_sold": safe_sum('total_units_sold'),
        "total_units_sold_change": 0,
        "avg_return_rate": round(safe_avg('avg_return_rate'), 2),
        "avg_return_rate_change": 0.0,
        "styles_need_replenishment": safe_sum('styles_need_replenishment'),
        "styles_need_replenishment_change": 0,
        "styles_broken": safe_sum('styles_broken'),
        "styles_broken_change": 0,
        "styles_out_of_stock": safe_sum('styles_out_of_stock'),
        "styles_out_of_stock_change": 0
    }
    
    # Aggregate Top SKUs - combine all and use pandas to sort and deduplicate
    all_top_skus = []
    for dashboard in dashboards:
        all_top_skus.extend(dashboard.get('top_skus', []))
    
    if all_top_skus:
        top_skus_df = pd.DataFrame(all_top_skus)
        if not top_skus_df.empty:
            # Remove duplicates by style_name and ad_platform, keep highest sell_through_rate
            top_skus_df = top_skus_df.sort_values('sell_through_rate', ascending=False)
            top_skus_df = top_skus_df.drop_duplicates(subset=['style_name', 'ad_platform'], keep='first')
            top_skus = top_skus_df.head(5).to_dict('records')
            # Round numeric values
            for sku in top_skus:
                if 'sell_through_rate' in sku:
                    sku['sell_through_rate'] = round(float(sku['sell_through_rate']), 0)
                if 'roas' in sku:
                    sku['roas'] = round(float(sku['roas']), 1)
        else:
            top_skus = []
    else:
        top_skus = []
    
    # Aggregate Fabric Status by fabric_type using pandas
    all_fabrics = []
    for dashboard in dashboards:
        all_fabrics.extend(dashboard.get('fabric_status', []))
    
    if all_fabrics:
        fabrics_df = pd.DataFrame(all_fabrics)
        if not fabrics_df.empty and 'fabric_type' in fabrics_df.columns:
            # Group by fabric_type and aggregate
            fabric_agg = fabrics_df.groupby('fabric_type').agg({
                'available': 'sum',
                'reorder_point': 'max'
            }).reset_index()
            
            # Calculate percent_consumed and status
            fabric_status = []
            for _, row in fabric_agg.iterrows():
                available = float(row['available'])
                reorder_point = float(row['reorder_point'])
                percent_consumed = round((available / reorder_point * 100), 1) if reorder_point > 0 else 0
                
                if available < reorder_point * 0.8:
                    status = 'LOW'
                elif available < reorder_point:
                    status = 'WARNING'
                else:
                    status = 'GOOD'
                
                fabric_status.append({
                    "fabric_type": str(row['fabric_type']),
                    "available": available,
                    "reorder_point": reorder_point,
                    "percent_consumed": percent_consumed,
                    "status": status
                })
            
            # Sort by status priority
            status_order = {'LOW': 0, 'WARNING': 1, 'GOOD': 2}
            fabric_status.sort(key=lambda x: status_order.get(x['status'], 3))
        else:
            fabric_status = []
    else:
        fabric_status = []
    
    # Aggregate Ad Performance by platform using pandas with weighted average
    all_ads = []
    for dashboard in dashboards:
        all_ads.extend(dashboard.get('ad_performance', []))
    
    if all_ads:
        ads_df = pd.DataFrame(all_ads)
        if not ads_df.empty and 'ad_platform' in ads_df.columns:
            # Group by platform and calculate weighted average ROAS
            ad_performance = []
            for platform in ads_df['ad_platform'].unique():
                platform_ads = ads_df[ads_df['ad_platform'] == platform]
                total_spend = float(platform_ads['total_spend'].sum())
                
                # Weighted average ROAS
                if total_spend > 0:
                    weighted_roas = (platform_ads['total_spend'] * platform_ads['avg_roas']).sum() / total_spend
                else:
                    weighted_roas = float(platform_ads['avg_roas'].mean())
                
                ad_performance.append({
                    "ad_platform": str(platform),
                    "total_spend": round(total_spend, 0),
                    "avg_roas": round(weighted_roas, 2)
                })
            
            # Sort by total_spend descending
            ad_performance.sort(key=lambda x: x['total_spend'], reverse=True)
        else:
            ad_performance = []
    else:
        ad_performance = []
    
    # Aggregate Alerts - combine all and use pandas for sorting
    all_alerts = []
    for dashboard in dashboards:
        all_alerts.extend(dashboard.get('alerts', []))
    
    if all_alerts:
        alerts_df = pd.DataFrame(all_alerts)
        if not alerts_df.empty:
            # Remove duplicates by style_id and alert_type if columns exist
            if 'style_id' in alerts_df.columns and 'alert_type' in alerts_df.columns:
                alerts_df = alerts_df.drop_duplicates(subset=['style_id', 'alert_type'], keep='first')
            
            # Sort by severity and date
            severity_order = {'critical': 0, 'warning': 1, 'stable': 2}
            if 'severity' in alerts_df.columns:
                alerts_df['severity_order'] = alerts_df['severity'].map(lambda x: severity_order.get(x, 3))
            else:
                alerts_df['severity_order'] = 3
            
            if 'date' in alerts_df.columns:
                alerts_df['date_parsed'] = pd.to_datetime(alerts_df['date'], errors='coerce')
            else:
                alerts_df['date_parsed'] = pd.NaT
            
            # Sort and limit
            alerts_df = alerts_df.sort_values(['severity_order', 'date_parsed'], ascending=[True, False], na_position='last')
            alerts_df = alerts_df.head(10)
            
            # Remove helper columns
            if 'severity_order' in alerts_df.columns:
                alerts_df = alerts_df.drop('severity_order', axis=1)
            if 'date_parsed' in alerts_df.columns:
                alerts_df = alerts_df.drop('date_parsed', axis=1)
            
            alerts = alerts_df.to_dict('records')
        else:
            alerts = []
    else:
        alerts = []
    
    logger.info("Dashboard aggregation completed successfully")
    
    return {
        "overview": overview,
        "top_skus": top_skus,
        "fabric_status": fabric_status,
        "ad_performance": ad_performance,
        "alerts": alerts
    }


# Helpers for request normalization and type coercion
def _normalize_payload(data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize incoming payload to a list of row dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise HTTPException(status_code=422, detail="'data' must be an object or list of objects")


def _build_dataframe(data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame from payload and coerce types used in analysis."""
    records = _normalize_payload(data)
    df = pd.DataFrame(records)
    # Coerce numeric columns used in calculations
    numeric_cols = [
        "myntra_available_qty", "nykaa_available_qty", "total_units_sold", "return_rate_avg",
        "myntra_return_rate", "nykaa_return_rate", "sell_through_rate", "fabric_1_available_meters",
        "fabric_1_reorder_point", "fabric_2_available_meters", "fabric_2_reorder_point",
        "fabric_3_available_meters", "fabric_3_reorder_point", "mrp", "myntra_price", "nykaa_price",
        "myntra_discount_pct", "nykaa_discount_pct", "ad_spend", "roas", "clicks", "impressions",
        "planned_qty", "completed_qty", "fabric_consumed_meters", "fabric_yield",
        "myntra_S_qty", "myntra_M_qty", "myntra_L_qty", "myntra_XL_qty",
        "nykaa_S_qty", "nykaa_M_qty", "nykaa_L_qty", "nykaa_XL_qty"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Normalize booleans frequently used
    boolean_cols = [
        "active_flag", "alert_low_stock_myntra", "alert_fabric_reorder",
        "alert_high_return_rate", "broken_size_curve"
    ]
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("true", "1", "yes", "y"): return True
        if s in ("false", "0", "no", "n"): return False
        return v
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].map(_to_bool)
    # Lowercase status-like text
    for col in ["myntra_size_curve_status", "nykaa_size_curve_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
    return df


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Dora Dori AI - Inventory Intelligence API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_inventory_data(request: InventoryDataRequest):
    """
    Main endpoint for analyzing inventory data
    Receives raw data from n8n, processes it using pandas, and returns analyzed dashboard data
    """
    try:
        _records = request.data if isinstance(request.data, list) else [request.data]
        logger.info(f"Received data with {len(_records)} records")
        
        # Convert to pandas DataFrame with coercion
        df = _build_dataframe(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        
        logger.info(f"Processing DataFrame with shape: {df.shape}")
        
        # Perform all analyses
        overview = calculate_overview_metrics(df)
        top_skus = calculate_top_skus(df)
        fabric_status = calculate_fabric_status(df)
        ad_performance = calculate_ad_performance(df)
        alerts = calculate_alerts(df)
        
        # Construct response matching the expected frontend format
        dashboard_data = {
            "overview": overview,
            "top_skus": top_skus,
            "fabric_status": fabric_status,
            "ad_performance": ad_performance,
            "alerts": alerts
        }
        
        logger.info("Analysis completed successfully")
        return {"dashboard": dashboard_data}
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.post("/analyze/overview")
async def analyze_overview(request: InventoryDataRequest):
    """Analyze only overview metrics"""
    try:
        df = _build_dataframe(request.data)
        overview = calculate_overview_metrics(df)
        return {"overview": overview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/top-skus")
async def analyze_top_skus(request: InventoryDataRequest, limit: int = 5):
    """Analyze top performing SKUs"""
    try:
        df = _build_dataframe(request.data)
        top_skus = calculate_top_skus(df, limit)
        return {"top_skus": top_skus}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/fabric-status")
async def analyze_fabric_status(request: InventoryDataRequest):
    """Analyze fabric status"""
    try:
        df = _build_dataframe(request.data)
        fabric_status = calculate_fabric_status(df)
        return {"fabric_status": fabric_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/ad-performance")
async def analyze_ad_performance(request: InventoryDataRequest):
    """Analyze ad performance"""
    try:
        df = _build_dataframe(request.data)
        ad_performance = calculate_ad_performance(df)
        return {"ad_performance": ad_performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/alerts")
async def analyze_alerts(request: InventoryDataRequest, limit: int = 10):
    """Analyze alerts"""
    try:
        df = _build_dataframe(request.data)
        alerts = calculate_alerts(df, limit)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/aggregate", response_model=Dict[str, Any])
async def aggregate_dashboards(request: DashboardArrayRequest):
    """
    Aggregate multiple dashboard objects into a single consolidated dashboard
    Accepts an array of dashboard objects from n8n and performs data analysis using pandas
    Supports both 'dashboards' and 'data' field names for n8n compatibility
    
    If raw inventory data is sent (items with 'style_id', 'style_name', etc.), 
    it will first analyze each item to create dashboards, then aggregate them.
    If dashboard objects are sent (items with 'dashboard' key), it will aggregate directly.
    """
    try:
        # Support both 'dashboards' and 'data' field names
        input_array = request.dashboards or request.data
        
        if not input_array:
            raise HTTPException(
                status_code=422, 
                detail="Either 'dashboards' or 'data' field is required"
            )
        
        logger.info(f"Received {len(input_array)} items for processing")
        
        # Check if input is raw inventory data or dashboard objects
        # If first item has 'dashboard' key, it's already dashboard objects
        # Otherwise, it's raw inventory data that needs to be analyzed first
        if input_array and len(input_array) > 0:
            first_item = input_array[0]
            if isinstance(first_item, dict) and 'dashboard' in first_item:
                # Already dashboard objects - aggregate directly
                logger.info("Input appears to be dashboard objects, aggregating directly")
                aggregated_dashboard = aggregate_dashboard_data(input_array)
            else:
                # Raw inventory data - analyze all together as a batch (more efficient)
                logger.info("Input appears to be raw inventory data, analyzing as batch")
                try:
                    # Build dataframe from all items
                    df = _build_dataframe(input_array)
                    if df.empty:
                        raise HTTPException(
                            status_code=400,
                            detail="No valid data could be analyzed from input"
                        )
                    
                    # Create a single dashboard from all the data
                    aggregated_dashboard = {
                        "overview": calculate_overview_metrics(df),
                        "top_skus": calculate_top_skus(df),
                        "fabric_status": calculate_fabric_status(df),
                        "ad_performance": calculate_ad_performance(df),
                        "alerts": calculate_alerts(df)
                    }
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error analyzing raw inventory data: {str(e)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error analyzing inventory data: {str(e)}"
                    )
        else:
            raise HTTPException(
                status_code=400,
                detail="Input array is empty"
            )
        
        logger.info("Dashboard aggregation completed successfully")
        return {"dashboard": aggregated_dashboard}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error aggregating dashboard data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error aggregating dashboard data: {str(e)}")


@app.post("/analyze/aggregate-raw", response_model=Dict[str, Any])
async def aggregate_dashboards_raw(data: List[Dict[str, Any]]):
    """
    Aggregate multiple dashboard objects (raw array format)
    Accepts a raw array of dashboard objects directly from n8n
    """
    try:
        logger.info(f"Received {len(data)} dashboard objects for aggregation (raw format)")
        
        # Aggregate all dashboards
        aggregated_dashboard = aggregate_dashboard_data(data)
        
        logger.info("Dashboard aggregation completed successfully")
        return {"dashboard": aggregated_dashboard}
        
    except Exception as e:
        logger.error(f"Error aggregating dashboard data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error aggregating dashboard data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


