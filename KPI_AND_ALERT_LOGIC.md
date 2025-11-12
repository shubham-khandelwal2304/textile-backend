# KPI, Metrics, and Alert Logic Documentation

This document explains the implementation logic for all KPIs, metrics, and inventory alerts in the Dora Dori Vista dashboard.

## Table of Contents
1. [Dashboard KPIs](#dashboard-kpis)
2. [Additional Metrics](#additional-metrics)
3. [Inventory Alerts](#inventory-alerts)

---

## Dashboard KPIs

### 1. Total Active Styles
**Logic:**
```python
total_active_styles = active_df['style_id'].nunique()
```
- Counts distinct style IDs where `active_flag = True`
- Represents the total number of active product styles in inventory

### 2. Fabrics that Require Replenishments
**Logic:**
```python
fabrics_need_replenishment = active_df[
    (active_df.get('fabric_1_available_meters', 0) < active_df.get('fabric_1_reorder_point', 0)) |
    (active_df.get('fabric_2_available_meters', 0) < active_df.get('fabric_2_reorder_point', 0)) |
    (active_df.get('fabric_3_available_meters', 0) < active_df.get('fabric_3_reorder_point', 0))
]['style_id'].nunique()
```
- Checks all 3 fabric types (fabric_1, fabric_2, fabric_3)
- Counts styles where available meters < reorder point for any fabric type
- Returns count of distinct styles needing fabric replenishment

### 3. Average Days of Cover
**Logic:**
```python
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
avg_days_of_cover = active_df['days_of_cover'].mean()
```
- Calculates: `(Total Available Qty) / (Daily Sales Rate)`
- Daily sales = `total_units_sold / 30` (assuming 30-day period)
- Returns average days of inventory coverage across all styles

### 4. Average Return Rate
**Logic:**
```python
avg_return_rate = active_df.get('return_rate_avg', 0).mean()
```
- Averages the `return_rate_avg` field across all active styles
- Typically expressed as a percentage

### 5. Total Styles that Need Replenishment
**Logic:**
```python
styles_need_replenishment = active_df[
    (active_df.get('myntra_available_qty', 0) < 10) |
    (active_df.get('nykaa_available_qty', 0) < 10)
]['style_id'].nunique()
```
- Counts styles where available quantity < 10 units on either Myntra or Nykaa
- Uses default reorder point of 10 units
- Returns count of distinct styles needing replenishment

### 6. Total Units Sold
**Logic:**
```python
total_units_sold = active_df.get('total_units_sold', 0).sum()
```
- Sums `total_units_sold` across all active styles
- Represents total sales volume over the period (typically 30 days)

### 7. Styles that Need Replenishments Platform-wise (Myntra vs Nykaa)
**Logic:**
```python
# Myntra
styles_need_replenishment_myntra = active_df[
    active_df.get('myntra_available_qty', 0) < 10
]['style_id'].nunique()

# Nykaa
styles_need_replenishment_nykaa = active_df[
    active_df.get('nykaa_available_qty', 0) < 10
]['style_id'].nunique()
```
- Separately counts styles needing replenishment on each platform
- Myntra: styles with `myntra_available_qty < 10`
- Nykaa: styles with `nykaa_available_qty < 10`
- Allows platform-specific inventory management

### 8. Size Curve Broken
**Logic:**
```python
styles_broken = active_df[
    (active_df.get('myntra_size_curve_status', '') == 'broken') |
    (active_df.get('nykaa_size_curve_status', '') == 'broken') |
    (active_df.get('broken_size_curve', False) == True)
]['style_id'].nunique()
```
- Checks size curve status on both platforms
- Flags styles with broken size curves on Myntra, Nykaa, or both
- Returns count of distinct styles with broken size curves

---

## Additional Metrics

### 1. Top Performing SKU
**Logic:**
```python
# Filter styles with sales
sales_df = df[df.get('total_units_sold', 0) > 0].copy()

# Sort by sell-through rate descending
top_skus = sales_df.nlargest(limit, 'sell_through_rate', keep='all')
```
- Filters styles with `total_units_sold > 0`
- Sorts by `sell_through_rate` in descending order
- Returns top N SKUs (default: 5)
- Includes: style_name, ad_platform, sell_through_rate, ROAS

### 2. Fabric Status
**Logic:**
```python
# For each fabric type
available = fabric_rows['fabric_1_available_meters'].iloc[0]
reorder_point = fabric_rows['fabric_1_reorder_point'].iloc[0]
percent_consumed = (available / reorder_point) * 100

# Determine status
if available < reorder_point * 0.8:
    status = 'LOW'
elif available < reorder_point:
    status = 'WARNING'
else:
    status = 'GOOD'
```
- Groups by fabric type
- Calculates percent consumed: `(available / reorder_point) * 100`
- Status thresholds:
  - **LOW**: available < 80% of reorder point
  - **WARNING**: available < reorder point
  - **GOOD**: available >= reorder point
- Returns list of fabric statuses sorted by priority (LOW → WARNING → GOOD)

### 3. Ad Spend vs ROAS by Platform
**Logic:**
```python
# Group by platform
ad_performance = ad_df.groupby('ad_platform').agg({
    'ad_spend': 'sum',
    'roas': 'mean',
    'clicks': 'sum',
    'impressions': 'sum',
    'total_units_sold': 'sum'
}).reset_index()

# Calculate revenue
total_revenue = ad_spend * roas
```
- Groups ad data by `ad_platform`
- Aggregates:
  - Total ad spend (sum)
  - Average ROAS (mean)
  - Total clicks, impressions, units sold
- Calculates total revenue: `ad_spend * ROAS`
- Returns platform-wise performance metrics

### 4. Ad Performance Summary
**Enhanced Logic:**
```python
# Additional metrics calculated
ctr = (clicks / impressions) * 100  # Click-through rate
cpc = ad_spend / clicks  # Cost per click
conversion_rate = (total_units_sold / clicks) * 100
total_revenue = ad_spend * roas
```
- **CTR (Click-Through Rate)**: `(clicks / impressions) * 100`
- **CPC (Cost Per Click)**: `ad_spend / clicks`
- **Conversion Rate**: `(total_units_sold / clicks) * 100`
- **Total Revenue**: `ad_spend * ROAS`
- Provides comprehensive ad performance analysis by platform

---

## Inventory Alerts

All alerts are calculated for active styles (`active_flag = True`) and sorted by severity (critical → warning).

### 1. Low Stock on High Selling Style
**Logic:**
```python
high_selling_threshold = active_df.get('total_units_sold', 0).quantile(0.75)
if total_units_sold >= high_selling_threshold and total_qty < 20:
    # Alert triggered
```
- **Threshold**: Styles in top 25% by units sold (75th percentile)
- **Condition**: Total available quantity < 20 units
- **Severity**: Critical
- **Purpose**: Prevents stockouts on best-selling products

### 2. Fabric Reorder Needed
**Logic:**
```python
if (fabric_1_available < fabric_1_reorder and fabric_1_reorder > 0) or \
   (fabric_2_available < fabric_2_reorder and fabric_2_reorder > 0) or \
   (fabric_3_available < fabric_3_reorder and fabric_3_reorder > 0):
    # Alert triggered
```
- Checks all 3 fabric types
- Triggers when available meters < reorder point
- **Severity**: Critical
- **Purpose**: Ensures fabric availability for production

### 3. Broken Size Curve
**Logic:**
```python
myntra_broken = str(row.get('myntra_size_curve_status', '')).lower() == 'broken'
nykaa_broken = str(row.get('nykaa_size_curve_status', '')).lower() == 'broken'
broken_flag = bool(row.get('broken_size_curve', False))

if myntra_broken or nykaa_broken or broken_flag:
    # Alert triggered
```
- Checks size curve status on both platforms
- Flags broken size curves on Myntra, Nykaa, or both
- **Severity**: Warning
- **Purpose**: Identifies inventory distribution issues

### 4. High Return Rate
**Logic:**
```python
return_rate = float(row.get('return_rate_avg', 0))
if return_rate > 15.0:  # Threshold: >15%
    # Alert triggered
```
- **Threshold**: Return rate > 15%
- **Severity**: Warning
- **Purpose**: Identifies products with quality or fit issues

### 5. Planned vs Completed Production Gap
**Logic:**
```python
planned_qty = float(row.get('planned_qty', 0))
completed_qty = float(row.get('completed_qty', 0))
if planned_qty > 0:
    gap_pct = ((planned_qty - completed_qty) / planned_qty) * 100
    if gap_pct > 20:  # More than 20% gap
        # Alert triggered
```
- **Threshold**: Production gap > 20%
- Calculates: `((planned - completed) / planned) * 100`
- **Severity**: Warning
- **Purpose**: Tracks production efficiency and delays

### 6. Low Sell Through Rate
**Logic:**
```python
avg_sell_through = active_df.get('sell_through_rate', 0).mean()
sell_through = float(row.get('sell_through_rate', 0))
if sell_through > 0 and sell_through < (avg_sell_through * 0.5):
    # Alert triggered
```
- **Threshold**: Sell-through rate < 50% of average
- Compares individual style to portfolio average
- **Severity**: Warning
- **Purpose**: Identifies slow-moving inventory

### 7. Ad Spend Inefficiency (Low ROAS)
**Logic:**
```python
avg_roas = active_df.get('roas', 0).mean()
roas = float(row.get('roas', 0))
ad_spend = float(row.get('ad_spend', 0))
if ad_spend > 0 and roas > 0 and roas < (avg_roas * 0.7):
    # Alert triggered
```
- **Threshold**: ROAS < 70% of average ROAS
- Only triggers if ad spend > 0
- **Severity**: Warning
- **Purpose**: Identifies inefficient ad campaigns

### 8. Fabric Over Consumption
**Logic:**
```python
fabric_consumed = float(row.get('fabric_consumed_meters', 0))
fabric_yield = float(row.get('fabric_yield', 0))
expected_consumption = completed_qty / fabric_yield
over_consumption_pct = ((fabric_consumed - expected_consumption) / expected_consumption) * 100
if over_consumption_pct > 15:  # More than 15% over
    # Alert triggered
```
- **Threshold**: Over-consumption > 15%
- Calculates expected consumption: `completed_qty / fabric_yield`
- Compares actual vs expected consumption
- **Severity**: Warning
- **Purpose**: Identifies waste or production inefficiencies

### 9. High Discount Dependency
**Logic:**
```python
myntra_discount = float(row.get('myntra_discount_pct', 0))
nykaa_discount = float(row.get('nykaa_discount_pct', 0))
max_discount = max(myntra_discount, nykaa_discount)
if max_discount > 40:  # More than 40% discount
    # Alert triggered
```
- **Threshold**: Maximum discount > 40%
- Checks both Myntra and Nykaa discount percentages
- **Severity**: Warning
- **Purpose**: Identifies products requiring excessive discounts to sell

### 10. Fast Moving Product Out of Stock
**Logic:**
```python
daily_sales = total_units_sold / 30
if daily_sales >= 5 and total_qty <= 0:  # Fast moving and OOS
    # Alert triggered
```
- **Threshold**: Daily sales >= 5 units/day AND total quantity = 0
- Daily sales = `total_units_sold / 30`
- **Severity**: Critical
- **Purpose**: Prevents lost sales on high-demand products

---

## Data Requirements

### Required Fields for KPIs:
- `style_id`, `active_flag`
- `fabric_1_available_meters`, `fabric_1_reorder_point`
- `fabric_2_available_meters`, `fabric_2_reorder_point`
- `fabric_3_available_meters`, `fabric_3_reorder_point`
- `myntra_available_qty`, `nykaa_available_qty`
- `total_units_sold`, `return_rate_avg`
- `myntra_size_curve_status`, `nykaa_size_curve_status`, `broken_size_curve`

### Required Fields for Metrics:
- `sell_through_rate`, `roas`, `ad_platform`
- `ad_spend`, `clicks`, `impressions`
- `fabric_1_type`, `fabric_2_type`, `fabric_3_type`

### Required Fields for Alerts:
- All KPI fields plus:
- `planned_qty`, `completed_qty`
- `fabric_consumed_meters`, `fabric_yield`
- `myntra_discount_pct`, `nykaa_discount_pct`

---

## Thresholds Summary

| Metric/Alert | Threshold | Type |
|-------------|-----------|------|
| Reorder Point (Default) | 10 units | Fixed |
| High Selling (75th percentile) | Top 25% by sales | Dynamic |
| Low Stock Alert | < 20 units | Fixed |
| High Return Rate | > 15% | Fixed |
| Production Gap | > 20% | Fixed |
| Low Sell Through | < 50% of average | Dynamic |
| Low ROAS | < 70% of average | Dynamic |
| Fabric Over Consumption | > 15% | Fixed |
| High Discount | > 40% | Fixed |
| Fast Moving | >= 5 units/day | Fixed |

---

## Notes

1. **Dynamic Thresholds**: Some alerts use dynamic thresholds based on portfolio averages (e.g., low sell-through, low ROAS). This ensures alerts are relative to your business performance.

2. **Platform-Specific**: Many metrics and alerts can be calculated separately for Myntra and Nykaa platforms.

3. **Severity Levels**:
   - **Critical**: Requires immediate action (stockouts, fabric shortages)
   - **Warning**: Needs attention but not urgent (performance issues, inefficiencies)

4. **Data Period**: Most calculations assume a 30-day period for sales data. Adjust the divisor if your period differs.

5. **Null Handling**: All calculations use `.get()` with default values to handle missing data gracefully.


