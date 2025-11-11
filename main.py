"""
Dora Dori AI - Inventory Intelligence Backend
Python FastAPI service for data analysis and processing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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
        logger.info(f"Received data with {len(request.data)} records")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
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
        df = pd.DataFrame(request.data)
        overview = calculate_overview_metrics(df)
        return {"overview": overview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/top-skus")
async def analyze_top_skus(request: InventoryDataRequest, limit: int = 5):
    """Analyze top performing SKUs"""
    try:
        df = pd.DataFrame(request.data)
        top_skus = calculate_top_skus(df, limit)
        return {"top_skus": top_skus}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/fabric-status")
async def analyze_fabric_status(request: InventoryDataRequest):
    """Analyze fabric status"""
    try:
        df = pd.DataFrame(request.data)
        fabric_status = calculate_fabric_status(df)
        return {"fabric_status": fabric_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/ad-performance")
async def analyze_ad_performance(request: InventoryDataRequest):
    """Analyze ad performance"""
    try:
        df = pd.DataFrame(request.data)
        ad_performance = calculate_ad_performance(df)
        return {"ad_performance": ad_performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/alerts")
async def analyze_alerts(request: InventoryDataRequest, limit: int = 10):
    """Analyze alerts"""
    try:
        df = pd.DataFrame(request.data)
        alerts = calculate_alerts(df, limit)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


