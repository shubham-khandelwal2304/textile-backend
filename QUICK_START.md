# Quick Start Guide

Get the Python data analysis backend up and running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Step 1: Install Dependencies

```bash
cd python_backend
pip install -r requirements.txt
```

## Step 2: Start the Service

**Option A: Using Python directly**
```bash
python main.py
```

**Option B: Using uvicorn (recommended for development)**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Using the provided scripts**
- Windows: `run.bat`
- Linux/Mac: `chmod +x run.sh && ./run.sh`

## Step 3: Verify It's Working

Open your browser and visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

Or run the test script:
```bash
python test_api.py
```

## Step 4: Update n8n Workflow

1. Open your n8n workflow
2. Add an **HTTP Request** node after your PostgreSQL node
3. Configure:
   - Method: POST
   - URL: `http://localhost:8000/analyze`
   - Body: `{"data": {{ $json }}}`
4. Save and activate

## Step 5: Test End-to-End

1. Trigger your n8n webhook
2. Verify the response contains analyzed dashboard data
3. Check your React frontend displays the data correctly

## Troubleshooting

**Port already in use?**
```bash
# Change port in main.py or use:
uvicorn main:app --port 8001
```

**Import errors?**
```bash
# Make sure all dependencies are installed:
pip install -r requirements.txt
```

**Connection refused?**
- Make sure the service is running
- Check firewall settings
- Verify the port is correct

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Read [N8N_INTEGRATION.md](N8N_INTEGRATION.md) for n8n setup
- Deploy to production (Azure, AWS, etc.)

## Production Deployment

For production, consider:
- Using environment variables for configuration
- Adding authentication/API keys
- Setting up proper logging
- Using a process manager (PM2, systemd, etc.)
- Enabling HTTPS

