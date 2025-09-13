import numpy as np
import yaml

with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

THRESHOLDS = config['thresholds']

# Decision logic for allocation and urgent PO

def check_urgent(current_stock, avg_3):
    return current_stock < THRESHOLDS['urgent_stock_ratio'] * avg_3

def compute_allocation(latest_metrics, total_qty):
    # Create a copy to avoid SettingWithCopyWarning
    result = latest_metrics.copy()
    
    # Proportional allocation by avg_7, but ensure realistic results
    total_avg_7 = result['avg_7'].sum()
    
    if total_avg_7 == 0:  # If no sales data, distribute equally
        result['alloc_prop'] = 1.0 / len(result)
        result['alloc'] = (result['alloc_prop'] * total_qty).astype(int)
    else:
        result['alloc_prop'] = result['avg_7'] / total_avg_7
        result['alloc'] = (result['alloc_prop'] * total_qty).astype(int)
    
    # Ensure minimum allocation of 1 unit per city if total_qty allows
    if total_qty >= len(result):
        result['alloc'] = result['alloc'].apply(lambda x: max(1, x))
    
    # Rebalance if total exceeds available quantity
    total_allocated = result['alloc'].sum()
    if total_allocated > total_qty:
        # Reduce proportionally
        scale_factor = total_qty / total_allocated
        result['alloc'] = (result['alloc'] * scale_factor).astype(int)
    
    # Ensure we don't exceed total_qty
    while result['alloc'].sum() > total_qty:
        # Reduce largest allocation by 1
        max_idx = result['alloc'].idxmax()
        if result.loc[max_idx, 'alloc'] > 1:
            result.loc[max_idx, 'alloc'] -= 1
        else:
            break
    
    return result[['city_name','stock_quantity','avg_7','avg_3','alloc']].rename(columns={'alloc': 'recommended_allocation'})

def generate_webhook_payload(product_id, qty, priority, reason, eta=7):
    return {
        'action': 'raise_po',
        'product_id': product_id,
        'qty': int(qty),
        'priority': priority,
        'reason': reason,
        'suggested_eta_days': eta,
        'requested_by': 'warehouse_agent_v1'
    }
