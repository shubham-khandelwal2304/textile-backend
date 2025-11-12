"""
Test script to verify aggregation endpoint output format
"""
import json
import requests

# Sample input (what you're sending)
sample_input = [
    {
        "dashboard": {
            "overview": {
                "total_active_styles": 1,
                "total_units_sold": 570
            },
            "top_skus": [{"style_name": "Style 1"}],
            "fabric_status": [],
            "ad_performance": [],
            "alerts": []
        }
    },
    {
        "dashboard": {
            "overview": {
                "total_active_styles": 1,
                "total_units_sold": 547
            },
            "top_skus": [{"style_name": "Style 2"}],
            "fabric_status": [],
            "ad_performance": [],
            "alerts": []
        }
    }
]

# Expected output format
expected_output = {
    "dashboard": {
        "overview": {
            "total_active_styles": 2,  # Sum of both
            "total_units_sold": 1117,  # Sum of both
            # ... other aggregated metrics
        },
        "top_skus": [
            # Top 5 SKUs from all dashboards combined
        ],
        "fabric_status": [
            # Aggregated fabric status
        ],
        "ad_performance": [
            # Aggregated ad performance by platform
        ],
        "alerts": [
            # Top 10 alerts from all dashboards
        ]
    }
}

print("Expected Output Format:")
print(json.dumps(expected_output, indent=2))

# Test the endpoint
# Uncomment to test:
# response = requests.post(
#     "http://localhost:8000/analyze/aggregate",
#     json={"dashboards": sample_input}
# )
# print("\nActual Response:")
# print(json.dumps(response.json(), indent=2))

