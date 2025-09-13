import pandas as pd

def add_rolling_features(daily_full):
    daily_full = daily_full.sort_values(['product_id','city_name','date'])
    daily_full['avg_3'] = daily_full.groupby(['product_id','city_name'])['units_sold'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily_full['avg_7'] = daily_full.groupby(['product_id','city_name'])['units_sold'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    daily_full['avg_14'] = daily_full.groupby(['product_id','city_name'])['units_sold'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    return daily_full

def merge_inventory(daily_full, inv):
    latest_inv = inv.copy()
    # Ensure product_id and city_name are both string type in both DataFrames
    daily_full['product_id'] = daily_full['product_id'].astype(str)
    latest_inv['product_id'] = latest_inv['product_id'].astype(str)
    daily_full['city_name'] = daily_full['city_name'].astype(str)
    latest_inv['city_name'] = latest_inv['city_name'].astype(str)
    daily_full = daily_full.merge(latest_inv[['product_id','city_name','stock_quantity']], on=['product_id','city_name'], how='left')
    daily_full['stock_quantity'] = daily_full['stock_quantity'].fillna(0)
    return daily_full

def compute_days_of_stock(daily_full):
    daily_full['days_of_stock'] = daily_full['stock_quantity'] / (daily_full['avg_7'].replace(0, 1e-6))
    return daily_full

def get_latest_metrics(daily_full):
    latest = daily_full.groupby(['product_id','city_name']).apply(lambda df: df.sort_values('date').iloc[-1]).reset_index(drop=True)
    return latest

def get_high_demand_products(daily_full, lookback_days=7, top_n=5):
    last_date = daily_full['date'].max()
    week_ago = last_date - pd.Timedelta(days=lookback_days-1)
    recent = daily_full[daily_full['date'] >= week_ago]
    demand = recent.groupby('product_id')['units_sold'].sum().reset_index()
    demand = demand.sort_values('units_sold', ascending=False)
    return demand.head(top_n)
