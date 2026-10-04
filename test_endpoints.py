import uvicorn
import threading
import time
import requests
import json
from app.main import app

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=8005, log_level='error')

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(3)

# Test login
r = requests.post('http://127.0.0.1:8005/auth/login', data={'username': 'i4c_admin', 'password': 'admin123'}, timeout=5)
token = r.json()['access_token']
headers = {'Authorization': 'Bearer ' + token}

# Test import endpoint
import_data = {
    'complaints': [
        {
            'transaction_id': 'REAL-001',
            'amount': 50000,
            'timestamp': '2026-10-04T10:00:00',
            'victim_bank': 'HDFC',
            'mode': 'UPI',
            'account_chain': [
                {'account_id': 'HDFC111', 'bank': 'HDFC', 'city': 'Mumbai', 'hop_position': 1, 'amount_received': 50000, 'amount_forwarded': 48000},
                {'account_id': 'SBI222', 'bank': 'SBI', 'city': 'Pune', 'hop_position': 2, 'amount_received': 48000, 'amount_forwarded': 0}
            ],
            'withdrawal_lat': 18.5204,
            'withdrawal_lon': 73.8567,
            'withdrawal_channel': 'ATM',
            'fraud_type': 'UPI Fraud',
            'victim_city': 'Mumbai',
            'victim_state': 'Maharashtra'
        }
    ],
    'source': 'CFCFRMS',
    'validate_only': False
}

r = requests.post('http://127.0.0.1:8005/import/complaints', json=import_data, headers=headers, timeout=10)
print('Import status:', r.status_code)
print('Import result:', r.json())

# Test retrain endpoint
r = requests.post('http://127.0.0.1:8005/model/retrain', json={'force_retrain': True}, headers=headers, timeout=120)
print('Retrain status:', r.status_code)
print('Retrain result:', r.json())

# Test real-graph metrics endpoint
r = requests.get('http://127.0.0.1:8005/metrics/real-graph', headers=headers, timeout=5)
print('Real-graph metrics status:', r.status_code)
if r.status_code == 200:
    print('Real-graph data:', r.json().get('note'))
else:
    print('Real-graph error:', r.json())