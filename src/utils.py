import requests
import os

def trigger_n8n_webhook(url, payload):
    response = requests.post(url, json=payload)
    return response.status_code, response.text

def get_role(user_type):
    if user_type.lower() == 'company':
        return 'Company'
    elif user_type.lower() == 'warehouse':
        return 'Warehouse'
    else:
        return 'Unknown'
