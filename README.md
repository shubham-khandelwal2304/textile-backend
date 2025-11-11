# Dora Dori AI - Python Backend Service

Python FastAPI service for data analysis and processing of inventory data using pandas and other data analysis libraries.

## Architecture

```
n8n → PostgreSQL (Raw Data) → Python API (Analysis) → n8n → React Frontend
```

## Setup

### 1. Install Dependencies

```bash
cd python_backend
pip install -r requirements.txt
```

### 2. Run the Service

**Development mode:**
```bash
python main.py
```

**Production mode (using uvicorn):**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 3. API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Main Analysis Endpoint

**POST** `/analyze`

Receives raw inventory data from n8n and returns fully analyzed dashboard data.

**Request Body:**
```json
{
  "data": [
    {
      "style_id": "123",
      "style_name": "Style Name",
      "active_flag": true,
      "fabric_1_type": "Cotton",
      "fabric_1_available_meters": 500,
      "fabric_1_reorder_point": 200,
      "myntra_available_qty": 50,
      "nykaa_available_qty": 30,
      "total_units_sold": 100,
      "return_rate_avg": 5.2,
      "ad_platform": "Meta",
      "ad_spend": 10000,
      "roas": 3.5,
      "sell_through_rate": 85.5,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "dashboard": {
    "overview": {
      "total_active_styles": 150,
      "total_active_styles_change": 0,
      "fabrics_that_require_replenishment": 25,
      ...
    },
    "top_skus": [...],
    "fabric_status": [...],
    "ad_performance": [...],
    "alerts": [...]
  }
}
```

### Individual Analysis Endpoints

- **POST** `/analyze/overview` - Overview metrics only
- **POST** `/analyze/top-skus?limit=5` - Top performing SKUs
- **POST** `/analyze/fabric-status` - Fabric status analysis
- **POST** `/analyze/ad-performance` - Ad performance metrics
- **POST** `/analyze/alerts?limit=10` - Alerts analysis

### Health Check

- **GET** `/` - Service info
- **GET** `/health` - Health check

## n8n Integration

### Workflow Setup

1. **PostgreSQL Node**: Fetch raw inventory data
   ```sql
   SELECT * FROM inventory_data WHERE active_flag = TRUE;
   ```

2. **HTTP Request Node**: Send data to Python API
   - Method: POST
   - URL: `http://localhost:8000/analyze` (or your deployed URL)
   - Body: 
     ```json
     {
       "data": {{ $json }}
     }
     ```

3. **Return Node**: Return analyzed data to webhook
   - Return: `{{ $json.dashboard }}`

### Example n8n Workflow

```
[Webhook Trigger] 
  → [PostgreSQL: Get Raw Data] 
  → [HTTP Request: Analyze Data] 
  → [Return Response]
```

## Data Analysis Functions

### Overview Metrics
- Total Active Styles
- Fabrics Requiring Replenishment
- Average Days of Cover
- Total Units Sold (30d)
- Average Return Rate
- Styles Needing Replenishment
- Styles Broken (Size Curve)
- Styles Out of Stock

### Top SKUs
- Sorted by Sell-Through Rate
- Includes ROAS and Platform info

### Fabric Status
- Calculates percent consumed: `(available / reorder_point) * 100`
- Status classification:
  - **LOW**: available < 80% of reorder point
  - **WARNING**: available < 100% of reorder point
  - **GOOD**: available >= 100% of reorder point

### Ad Performance
- Groups by ad platform
- Calculates total spend and average ROAS

### Alerts
- Low Stock alerts
- Fabric Reorder alerts
- High Return Rate alerts

## Deployment

### Local Development
```bash
python main.py
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
- `PORT`: Server port (default: 8000)
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)

## Testing

Test the API using curl:

```bash
# Health check
curl http://localhost:8000/health

# Analyze data
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": [{"style_id": "123", "active_flag": true, ...}]}'
```

## Notes

- The service expects data in the same format as your PostgreSQL schema
- All calculations replicate the SQL query logic using pandas
- Trend calculations (change values) are currently set to 0 and can be enhanced with historical data comparison
- The service is stateless and processes data on-demand


