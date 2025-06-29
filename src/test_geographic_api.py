"""
Quick test of geographic API endpoints
"""
import requests
import json

def test_api_endpoints():
    base_url = "https://api.energidataservice.dk/dataset"
    
    # Test municipality capacity with minimal data
    print("Testing municipality capacity API...")
    try:
        response = requests.get(f"{base_url}/CapacityPerMunicipality", 
                              timeout=10, params={'limit': 5})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Got {len(data.get('records', []))} records")
            if data.get('records'):
                print(f"Sample record: {data['records'][0]}")
        else:
            print(f"❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\nTesting community production API...")
    try:
        response = requests.get(f"{base_url}/CommunityProduction", 
                              timeout=10, params={'limit': 5})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Got {len(data.get('records', []))} records")
            if data.get('records'):
                print(f"Sample record: {data['records'][0]}")
        else:
            print(f"❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_endpoints() 