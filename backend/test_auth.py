#!/usr/bin/env python3

import requests

# Test login
login_data = {
    "email": "ranajawadriaz.work@gmail.com", 
    "password": "password123"
}

try:
    response = requests.post(
        "http://localhost:8001/api/v1/auth/login",
        json=login_data
    )
    
    print(f"Login response: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        # Test auth/me with cookies
        cookies = response.cookies
        me_response = requests.get(
            "http://localhost:8001/api/v1/auth/me",
            cookies=cookies
        )
        print(f"Auth me response: {me_response.status_code}")
        print(f"Me data: {me_response.text}")
        
except Exception as e:
    print(f"Error: {e}")
