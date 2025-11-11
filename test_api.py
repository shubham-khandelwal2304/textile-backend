"""
Test script for Python API service
Run this to verify the API is working correctly
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Sample test data matching the inventory_data schema
test_data = {
    "data": [
        {
            "style_id": "STYLE001",
            "style_name": "Test Style 1",
            "active_flag": True,
            "fabric_1_type": "Cotton",
            "fabric_1_available_meters": 150,
            "fabric_1_reorder_point": 200,
            "fabric_2_type": None,
            "fabric_2_available_meters": 0,
            "fabric_2_reorder_point": 0,
            "fabric_3_type": None,
            "fabric_3_available_meters": 0,
            "fabric_3_reorder_point": 0,
            "myntra_available_qty": 50,
            "nykaa_available_qty": 30,
            "total_units_sold": 100,
            "return_rate_avg": 5.2,
            "ad_platform": "Meta",
            "ad_spend": 10000,
            "roas": 3.5,
            "sell_through_rate": 85.5,
            "myntra_size_curve_status": "good",
            "nykaa_size_curve_status": "good",
            "broken_size_curve": False,
            "alert_low_stock_myntra": False,
            "alert_fabric_reorder": True,
            "alert_high_return_rate": False,
            "alert_message_summary": "Fabric reorder required",
            "days_of_cover": 24,
            "reorder_point": 10
        },
        {
            "style_id": "STYLE002",
            "style_name": "Test Style 2",
            "active_flag": True,
            "fabric_1_type": "Polyester",
            "fabric_1_available_meters": 500,
            "fabric_1_reorder_point": 200,
            "fabric_2_type": None,
            "fabric_2_available_meters": 0,
            "fabric_2_reorder_point": 0,
            "fabric_3_type": None,
            "fabric_3_available_meters": 0,
            "fabric_3_reorder_point": 0,
            "myntra_available_qty": 80,
            "nykaa_available_qty": 50,
            "total_units_sold": 150,
            "return_rate_avg": 3.8,
            "ad_platform": "Google",
            "ad_spend": 15000,
            "roas": 4.2,
            "sell_through_rate": 92.0,
            "myntra_size_curve_status": "good",
            "nykaa_size_curve_status": "broken",
            "broken_size_curve": True,
            "alert_low_stock_myntra": False,
            "alert_fabric_reorder": False,
            "alert_high_return_rate": False,
            "alert_message_summary": None,
            "days_of_cover": 30,
            "reorder_point": 10
        },
        {
            "style_id": "STYLE003",
            "style_name": "Test Style 3",
            "active_flag": True,
            "fabric_1_type": "Cotton",
            "fabric_1_available_meters": 180,
            "fabric_1_reorder_point": 200,
            "fabric_2_type": None,
            "fabric_2_available_meters": 0,
            "fabric_2_reorder_point": 0,
            "fabric_3_type": None,
            "fabric_3_available_meters": 0,
            "fabric_3_reorder_point": 0,
            "myntra_available_qty": 5,
            "nykaa_available_qty": 3,
            "total_units_sold": 200,
            "return_rate_avg": 7.5,
            "ad_platform": "Meta",
            "ad_spend": 20000,
            "roas": 3.8,
            "sell_through_rate": 78.5,
            "myntra_size_curve_status": "good",
            "nykaa_size_curve_status": "good",
            "broken_size_curve": False,
            "alert_low_stock_myntra": True,
            "alert_fabric_reorder": False,
            "alert_high_return_rate": True,
            "alert_message_summary": "Low stock and high return rate",
            "days_of_cover": 12,
            "reorder_point": 10
        }
    ]
}


def test_health_check():
    """Test health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✓ Health check passed\n")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}\n")
        return False


def test_analyze_endpoint():
    """Test main analyze endpoint"""
    print("Testing /analyze endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            dashboard = result.get("dashboard", {})
            
            print("\nDashboard Data Structure:")
            print(f"  - Overview: {len(dashboard.get('overview', {}))} metrics")
            print(f"  - Top SKUs: {len(dashboard.get('top_skus', []))} items")
            print(f"  - Fabric Status: {len(dashboard.get('fabric_status', []))} fabrics")
            print(f"  - Ad Performance: {len(dashboard.get('ad_performance', []))} platforms")
            print(f"  - Alerts: {len(dashboard.get('alerts', []))} alerts")
            
            print("\nSample Overview Metrics:")
            overview = dashboard.get('overview', {})
            print(f"  - Total Active Styles: {overview.get('total_active_styles', 'N/A')}")
            print(f"  - Total Units Sold: {overview.get('total_units_sold', 'N/A')}")
            print(f"  - Avg Return Rate: {overview.get('avg_return_rate', 'N/A')}%")
            
            print("\nSample Fabric Status:")
            for fabric in dashboard.get('fabric_status', [])[:2]:
                print(f"  - {fabric.get('fabric_type')}: {fabric.get('status')} ({fabric.get('percent_consumed')}% consumed)")
            
            print("\n✓ Analyze endpoint passed\n")
            return True
        else:
            print(f"✗ Analyze endpoint failed: {response.text}\n")
            return False
    except Exception as e:
        print(f"✗ Analyze endpoint failed: {e}\n")
        return False


def test_individual_endpoints():
    """Test individual analysis endpoints"""
    endpoints = [
        ("/analyze/overview", "Overview"),
        ("/analyze/top-skus?limit=5", "Top SKUs"),
        ("/analyze/fabric-status", "Fabric Status"),
        ("/analyze/ad-performance", "Ad Performance"),
        ("/analyze/alerts?limit=10", "Alerts")
    ]
    
    print("Testing individual endpoints...")
    results = []
    
    for endpoint, name in endpoints:
        try:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                print(f"  ✓ {name}: OK")
                results.append(True)
            else:
                print(f"  ✗ {name}: Failed ({response.status_code})")
                results.append(False)
        except Exception as e:
            print(f"  ✗ {name}: Error - {e}")
            results.append(False)
    
    print()
    return all(results)


def main():
    """Run all tests"""
    print("=" * 60)
    print("Python API Service Test Suite")
    print("=" * 60)
    print()
    
    # Check if service is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to API service")
        print(f"  Make sure the service is running at {BASE_URL}")
        print("  Start it with: python main.py")
        return
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Analyze Endpoint", test_analyze_endpoint),
        ("Individual Endpoints", test_individual_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append(result)
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print()


if __name__ == "__main__":
    main()


