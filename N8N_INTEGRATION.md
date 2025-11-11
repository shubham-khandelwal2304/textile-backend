# n8n Integration Guide

This guide explains how to integrate the Python data analysis service with n8n workflows.

## Architecture Flow

```
[Webhook Trigger] 
  → [PostgreSQL: Fetch Raw Data] 
  → [HTTP Request: Send to Python API] 
  → [Python API: Analyze with Pandas] 
  → [Return: Analyzed Dashboard Data] 
  → [React Frontend]
```

## Step-by-Step n8n Workflow Setup

### Step 1: Webhook Trigger

1. Add a **Webhook** node
2. Set method to **GET** (or POST if preferred)
3. Set path: `inventory-dashboard`
4. Activate the workflow
5. Copy the webhook URL (e.g., `https://n8n-excollo.azurewebsites.net/webhook/inventory-dashboard`)

### Step 2: PostgreSQL - Fetch Raw Data

1. Add a **PostgreSQL** node
2. Connect to your database
3. Set operation to **Execute Query**
4. Use this query:

```sql
SELECT 
  style_id,
  style_name,
  active_flag,
  fabric_1_type,
  fabric_1_available_meters,
  fabric_1_reorder_point,
  fabric_2_type,
  fabric_2_available_meters,
  fabric_2_reorder_point,
  fabric_3_type,
  fabric_3_available_meters,
  fabric_3_reorder_point,
  myntra_available_qty,
  nykaa_available_qty,
  total_units_sold,
  return_rate_avg,
  ad_platform,
  ad_spend,
  roas,
  sell_through_rate,
  myntra_size_curve_status,
  nykaa_size_curve_status,
  broken_size_curve,
  alert_low_stock_myntra,
  alert_fabric_reorder,
  alert_high_return_rate,
  alert_message_summary,
  days_of_cover,
  reorder_point
FROM inventory_data
WHERE active_flag = TRUE;
```

**Note**: Adjust the SELECT fields based on your actual database schema.

### Step 3: Transform Data (Optional)

If your PostgreSQL node returns data in a different format, you may need a **Code** node to transform it:

```javascript
// Transform array of objects to match expected format
const items = $input.all();
return items.map(item => item.json);
```

### Step 4: HTTP Request - Call Python API

1. Add an **HTTP Request** node
2. Configure:
   - **Method**: POST
   - **URL**: `http://localhost:8000/analyze` (or your deployed Python service URL)
   - **Authentication**: None (or add if you implement auth)
   - **Body Content Type**: JSON
   - **Body**:
     ```json
     {
       "data": {{ $json }}
     }
     ```

**For Production**: Replace `localhost:8000` with your deployed Python service URL (e.g., `https://your-python-api.azurewebsites.net/analyze`)

### Step 5: Extract Dashboard Data

Add a **Code** node to extract the dashboard object:

```javascript
// Extract dashboard from response
const response = $input.first().json;
return {
  json: response.dashboard || response
};
```

### Step 6: Return Response

1. Add a **Respond to Webhook** node
2. Set **Response Code**: 200
3. Set **Response Body**: `{{ $json }}`

## Complete n8n Workflow JSON

Here's a sample workflow JSON structure (you can import this into n8n):

```json
{
  "name": "Inventory Dashboard with Python Analysis",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "GET",
        "path": "inventory-dashboard"
      },
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT * FROM inventory_data WHERE active_flag = TRUE"
      },
      "name": "PostgreSQL",
      "type": "n8n-nodes-base.postgres",
      "position": [450, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/analyze",
        "jsonParameters": true,
        "bodyParametersJson": "={\n  \"data\": {{ $json }}\n}"
      },
      "name": "Python API",
      "type": "n8n-nodes-base.httpRequest",
      "position": [650, 300]
    },
    {
      "parameters": {
        "jsCode": "const response = $input.first().json;\nreturn {\n  json: response.dashboard || response\n};"
      },
      "name": "Extract Dashboard",
      "type": "n8n-nodes-base.code",
      "position": [850, 300]
    },
    {
      "parameters": {
        "respondWith": "json",
        "responseBody": "={{ $json }}"
      },
      "name": "Respond to Webhook",
      "type": "n8n-nodes-base.respondToWebhook",
      "position": [1050, 300]
    }
  ],
  "connections": {
    "Webhook": {
      "main": [[{"node": "PostgreSQL"}]]
    },
    "PostgreSQL": {
      "main": [[{"node": "Python API"}]]
    },
    "Python API": {
      "main": [[{"node": "Extract Dashboard"}]]
    },
    "Extract Dashboard": {
      "main": [[{"node": "Respond to Webhook"}]]
    }
  }
}
```

## Alternative: Direct Analysis Endpoints

If you want to analyze specific sections separately, you can use individual endpoints:

### Overview Only
```json
{
  "url": "http://localhost:8000/analyze/overview",
  "body": {
    "data": {{ $json }}
  }
}
```

### Top SKUs
```json
{
  "url": "http://localhost:8000/analyze/top-skus?limit=5",
  "body": {
    "data": {{ $json }}
  }
}
```

## Error Handling

Add error handling in n8n:

1. Add an **IF** node after the HTTP Request node
2. Check if response status is 200
3. If error, use a **Code** node to format error response:

```javascript
const error = $input.first().json;
return {
  json: {
    error: true,
    message: error.detail || "Analysis failed",
    data: null
  }
};
```

## Testing the Workflow

1. **Test PostgreSQL Node**: Verify it returns data
2. **Test Python API**: Use a tool like Postman to test the Python API directly
3. **Test Full Workflow**: Trigger the webhook and verify the response

### Test Python API Directly

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "style_id": "123",
        "style_name": "Test Style",
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
        "sell_through_rate": 85.5
      }
    ]
  }'
```

## Production Deployment

### Python Service Deployment

Deploy your Python service to:
- **Azure App Service**
- **AWS Lambda** (with serverless framework)
- **Google Cloud Run**
- **Heroku**
- **Docker Container**

Update the n8n HTTP Request node URL to point to your deployed service.

### Environment Variables

Set these in your Python service deployment:
- `PORT`: Server port
- `CORS_ORIGINS`: Allowed origins for CORS

### Security Considerations

1. **Add Authentication**: Implement API keys or OAuth for the Python API
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **Input Validation**: Validate incoming data in Python service
4. **HTTPS**: Use HTTPS in production
5. **Error Logging**: Set up proper logging and monitoring

## Troubleshooting

### Python API Not Responding
- Check if the service is running: `curl http://localhost:8000/health`
- Check logs for errors
- Verify port is not blocked by firewall

### Data Format Issues
- Verify PostgreSQL query returns expected fields
- Check data types match expected format
- Use n8n's "Execute Workflow" to debug step-by-step

### Performance Issues
- Consider caching analyzed results
- Optimize pandas operations for large datasets
- Use database indexes for faster queries

## Next Steps

1. Deploy Python service to your preferred platform
2. Update n8n workflow with production URL
3. Test end-to-end flow
4. Monitor performance and optimize as needed

