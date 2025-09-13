import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from src.data_processing import load_sales_inventory, clean_sales, aggregate_daily, complete_date_range
from src.feature_engineering import add_rolling_features, merge_inventory, compute_days_of_stock, get_latest_metrics, get_high_demand_products
from src.model import prepare_lgbm_data, train_lgbm, predict_lgbm, prophet_forecast
from src.decision_engine import check_urgent, compute_allocation, generate_webhook_payload
from src.email_system import PurchaseOrderEmailSystem
import os
import json

st.set_page_config(page_title='Quick Commerce Agentic AI', layout='wide')
st.title('Quick Commerce Agentic AI Dashboard')

# Load config
with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
lookback_days = config['demand_analysis']['lookback_days']
top_n = config['demand_analysis']['top_n']

role = st.sidebar.selectbox('Select Role', ['Company', 'Warehouse'])
sales_file = st.sidebar.file_uploader('Upload Sales Data CSV', type=['csv'])
inv_file = st.sidebar.file_uploader('Upload Inventory Data CSV', type=['csv'])

if sales_file and inv_file:
    sales, inv = load_sales_inventory(sales_file, inv_file)
    sales = clean_sales(sales)
    daily = aggregate_daily(sales)
    daily_full = complete_date_range(daily)
    daily_full = add_rolling_features(daily_full)
    daily_full = merge_inventory(daily_full, inv)
    daily_full = compute_days_of_stock(daily_full)
    latest = get_latest_metrics(daily_full)

    # Create comprehensive product mapping from both sales and inventory data
    # Start with sales data (has product names)
    sales_products = sales.drop_duplicates(subset=['product_id'])[['product_id', 'product_name', 'category', 'sub_category']].copy()
    sales_products['product_id'] = sales_products['product_id'].astype(str)
    
    # Get all unique product IDs from both datasets
    all_sales_ids = set(str(x) for x in sales['product_id'].unique())
    all_inventory_ids = set(str(x) for x in inv['product_id'].unique())
    all_product_ids = all_sales_ids.union(all_inventory_ids)
    
    # Filter out suspicious date-like IDs (contains '-' and looks like a date)
    def is_date_like(pid):
        pid_str = str(pid)
        return '-' in pid_str and len(pid_str) > 6 and any(char.isdigit() for char in pid_str)
    
    valid_product_ids = {pid for pid in all_product_ids if not is_date_like(pid)}
    
    # Create mapping for all valid products
    product_mapping_dict = []
    for pid in valid_product_ids:
        if pid in all_sales_ids:
            # Product exists in sales data - use full info
            sales_info = sales_products[sales_products['product_id'] == pid].iloc[0]
            product_mapping_dict.append({
                'product_id': pid,
                'product_name': sales_info['product_name'],
                'category': sales_info['category'],
                'sub_category': sales_info['sub_category']
            })
        else:
            # Product only exists in inventory - use inventory data
            inv_info = inv.drop_duplicates(subset=['product_id'])
            inv_info['product_id'] = inv_info['product_id'].astype(str)
            inv_match = inv_info[inv_info['product_id'] == pid]
            if not inv_match.empty:
                inv_data = inv_match.iloc[0]
                product_mapping_dict.append({
                    'product_id': pid,
                    'product_name': inv_data['product_name'],
                    'category': inv_data['category'],
                    'sub_category': inv_data['sub_category']
                })
            else:
                # Fallback for any edge cases
                product_mapping_dict.append({
                    'product_id': pid,
                    'product_name': f'Product {pid}',
                    'category': 'Unknown',
                    'sub_category': 'Unknown'
                })
    
    # Save comprehensive mapping
    with open('product_mapping.json', 'w') as f:
        json.dump(product_mapping_dict, f, indent=2)
    
    # Create lookup dictionary with string keys
    product_map = {str(item['product_id']): item for item in product_mapping_dict}
    
    # Helper functions
    def get_product_name(pid):
        return product_map.get(str(pid), {}).get('product_name', 'Unknown')
    
    def get_product_info(pid):
        return product_map.get(str(pid), {'product_name': 'Unknown', 'category': 'Unknown', 'sub_category': 'Unknown'})
    
    def product_display(pid):
        info = get_product_info(pid)
        return f"{info['product_name']} ({pid})"

    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Product Insights", "Model Performance", "Action Log"])

    # --- Dashboard Tab ---
    with tab1:
        st.header('Overview')
        st.subheader(f'Top {top_n} Highly Demanded Products (Last {lookback_days} Days)')
        top_products = get_high_demand_products(daily_full, lookback_days=lookback_days, top_n=top_n)
        st.dataframe(top_products, use_container_width=True)
        st.subheader('Latest Metrics Snapshot')
        st.dataframe(latest.head(20), use_container_width=True)

    # --- Product Insights Tab ---
    with tab2:
        st.header('Product Insights')
        product_options = [(product_display(pid), pid) for pid in latest['product_id'].unique()]
        product_id = st.selectbox('Select Product (by Name or ID)', product_options, format_func=lambda x: x[0], key='prod_select')[1]
        product_info = get_product_info(product_id)
        product_latest = latest[latest['product_id'] == product_id]
        
        # Auto-send PO email if urgent stock detected
        urgent_cities = product_latest[product_latest.apply(lambda r: check_urgent(r['stock_quantity'], r['avg_3']), axis=1)]
        if len(urgent_cities) > 0:
            total_urgent_qty = urgent_cities['avg_7'].sum() * 7  # 7 days supply
            
            # Auto-send email
            if total_urgent_qty > 0:
                try:
                    email_system = PurchaseOrderEmailSystem()
                    po_data = email_system.create_po_data(
                        product_info=product_info,
                        quantity=int(total_urgent_qty),
                        priority='HIGH'
                    )
                    
                    # Send email automatically
                    success, message = email_system.send_po_email(po_data, "rathikmohamed786@gmail.com")
                    
                    if success:
                        st.success(f'🚀 **Auto-sent Purchase Order {po_data["po_number"]}** for urgent stock situation!')
                        st.info(f'📧 Email sent to rathikmohamed786@gmail.com for {int(total_urgent_qty)} units of {product_info["product_name"]}')
                    else:
                        st.warning(f'⚠️ Auto-send failed: {message}')
                        
                except Exception as e:
                    st.warning(f'⚠️ Auto-send error: {str(e)}')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Product Name", product_info['product_name'])
        with col2:
            st.metric("Product ID", product_id)
        with col3:
            st.metric("Category", product_info['category'])
        st.subheader('City-wise Metrics')
        urgent_cities = product_latest[product_latest.apply(lambda r: check_urgent(r['stock_quantity'], r['avg_3']), axis=1)]
        
        # Clean city metrics display
        city_metrics = product_latest[['city_name', 'units_sold', 'avg_3', 'avg_7', 'avg_14', 'stock_quantity', 'days_of_stock']].copy()
        city_metrics['Current Stock'] = city_metrics['stock_quantity'].astype(int)
        city_metrics['3-Day Avg'] = city_metrics['avg_3'].round(1)
        city_metrics['7-Day Avg'] = city_metrics['avg_7'].round(1)
        city_metrics['14-Day Avg'] = city_metrics['avg_14'].round(1)
        city_metrics['Days of Stock'] = city_metrics['days_of_stock'].round(1)
        city_metrics['Status'] = city_metrics['city_name'].apply(
            lambda x: '🚨 URGENT' if x in urgent_cities['city_name'].values else '✅ OK'
        )
        
        st.dataframe(city_metrics[['city_name', 'Current Stock', '3-Day Avg', '7-Day Avg', '14-Day Avg', 'Days of Stock', 'Status']], 
                    use_container_width=True, hide_index=True)

        st.markdown('---')
        st.subheader('🎯 Smart Allocation')
        alloc_qty = st.number_input('📦 Units to Allocate', min_value=1, value=1000, key='alloc_qty')
        if st.button('🚀 Generate Allocation Plan'):
            alloc_df = compute_allocation(product_latest.copy(), alloc_qty)
            
            # Business-friendly allocation display
            st.success(f"✅ Allocation plan for {alloc_qty} units of {product_info['product_name']}")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏙️ Cities Covered", len(alloc_df))
            with col2:
                st.metric("📦 Total Allocated", f"{alloc_df['recommended_allocation'].sum()} units")
            with col3:
                st.metric("⚖️ Allocation Balance", f"{alloc_qty - alloc_df['recommended_allocation'].sum()} units")
            
            # Clean allocation table
            alloc_display = alloc_df.copy()
            alloc_display['Current Stock'] = alloc_display['stock_quantity'].astype(int)
            alloc_display['7-Day Avg Sales'] = alloc_display['avg_7'].round(1)
            alloc_display['Recommended Units'] = alloc_display['recommended_allocation']
            alloc_display['New Stock Level'] = alloc_display['Current Stock'] + alloc_display['Recommended Units']
            
            st.dataframe(alloc_display[['city_name', 'Current Stock', '7-Day Avg Sales', 'Recommended Units', 'New Stock Level']], 
                        use_container_width=True, hide_index=True)
            
            # Business summary
            st.info(f"💡 **Allocation Strategy:** Units distributed based on 7-day average sales performance across {len(alloc_df)} cities.")
            
            with st.expander('🔧 Technical Details (for Developers)'):
                rationale = f"Allocated {alloc_qty} units of {product_info['product_name']} ({product_id}) across cities based on 7-day avg sales."
                decision_json = {
                    'product_id': product_id,
                    'product_name': product_info['product_name'],
                    'category': product_info['category'],
                    'sub_category': product_info['sub_category'],
                    'allocation_quantity': alloc_qty,
                    'allocation_summary': alloc_df.to_dict(orient='records'),
                    'rationale': rationale
                }
                st.json(decision_json)

        st.markdown('---')
        st.subheader('Sales Trend Overview (Last 30 Days)')
        # Show sales trend for all cities as overview
        all_cities_df = daily_full[(daily_full['product_id'] == product_id)]
        
        # Create chart with proper settings to show realistic variation
        fig_overview = px.line(all_cities_df.tail(30), x='date', y='units_sold', color='city_name', 
                              title=f'Sales Trend Overview - {product_info["product_name"]}')
        
        # Disable smoothing and ensure realistic display
        fig_overview.update_traces(
            line=dict(width=2),
            mode='lines+markers',  # Show both lines and markers for better visibility
            marker=dict(size=4)
        )
        
        # Ensure proper axis scaling
        fig_overview.update_layout(
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
        
        st.markdown('---')
        st.subheader('City-Specific Analysis')
        # City selection dropdown
        available_cities = product_latest['city_name'].unique()
        selected_city = st.selectbox('Select City for Detailed Analysis', available_cities, key='city_select')
        
        if selected_city:
            city_df = daily_full[(daily_full['product_id'] == product_id) & (daily_full['city_name'] == selected_city)]
            
            # Put sales trend and forecast side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f'📊 Sales Trend - {selected_city}')
                fig_city = px.line(city_df.tail(30), x='date', y='units_sold', 
                                 title=f'Sales Trend - {selected_city}',
                                 labels={'date': 'Date', 'units_sold': 'Units Sold'})
                
                # Ensure realistic display without smoothing
                fig_city.update_traces(
                    line=dict(width=2),
                    mode='lines+markers',
                    marker=dict(size=4)
                )
                
                fig_city.update_layout(
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_city, use_container_width=True)
            
            with col2:
                st.subheader(f'📈 7-Day Forecast - {selected_city}')
                try:
                    forecast = prophet_forecast(city_df)
                    
                    # Clean forecast chart with realistic display
                    fig_forecast = px.line(forecast, x='ds', y='yhat', 
                                         title=f'Demand Forecast - {selected_city}',
                                         labels={'ds': 'Date', 'yhat': 'Predicted Units'})
                    
                    # Ensure realistic forecast display
                    fig_forecast.update_traces(
                        line=dict(width=3),
                        mode='lines+markers',
                        marker=dict(size=5)
                    )
                    
                    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                           mode='lines', name='Upper Range', 
                                           line=dict(dash='dot', color='lightblue', width=1))
                    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                           mode='lines', name='Lower Range', 
                                           line=dict(dash='dot', color='lightblue', width=1))
                    
                    fig_forecast.update_layout(
                        showlegend=True,
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                except Exception as e:
                    st.error(f'❌ Forecast unavailable for {selected_city}.')
                    # Fallback to simple average
                    avg_demand = city_df['units_sold'].mean()
                    st.info(f"📊 Historical average: {int(avg_demand)} units/day")
            
            # Forecast summary metrics below the charts
            if 'forecast' in locals():
                st.subheader(f'📊 Forecast Summary - {selected_city}')
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Avg Daily", f"{int(forecast['yhat'].mean())} units")
                with col2:
                    st.metric("📈 Peak Demand", f"{int(forecast['yhat'].max())} units")
                with col3:
                    st.metric("📉 Min Demand", f"{int(forecast['yhat'].min())} units")
                with col4:
                    st.metric("📅 Total 7-Day", f"{int(forecast['yhat'].sum())} units")
                
                # Simple forecast table
                forecast_display = forecast.copy()
                forecast_display['Date'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
                forecast_display['Predicted Units'] = forecast_display['yhat'].astype(int)
                forecast_display['Range'] = forecast_display.apply(
                    lambda x: f"{int(x['yhat_lower'])}-{int(x['yhat_upper'])}", axis=1
                )
                st.dataframe(forecast_display[['Date', 'Predicted Units', 'Range']], 
                           use_container_width=True, hide_index=True)

        st.markdown('---')
        st.subheader('🚨 Urgent Purchase Orders')
        if not urgent_cities.empty:
            st.warning(f"⚠️ **{len(urgent_cities)} cities need urgent restocking!**")
            
            # Summary of urgent cities
            urgent_summary = urgent_cities[['city_name', 'stock_quantity', 'avg_3', 'avg_7']].copy()
            urgent_summary['Current Stock'] = urgent_summary['stock_quantity'].astype(int)
            urgent_summary['3-Day Avg Sales'] = urgent_summary['avg_3'].round(1)
            urgent_summary['7-Day Avg Sales'] = urgent_summary['avg_7'].round(1)
            urgent_summary['Recommended PO'] = (urgent_summary['avg_7'] * 7).astype(int)  # 7 days of stock
            urgent_summary['Priority'] = 'HIGH'
            
            st.dataframe(urgent_summary[['city_name', 'Current Stock', '3-Day Avg Sales', '7-Day Avg Sales', 'Recommended PO', 'Priority']], 
                        use_container_width=True, hide_index=True)
            
            # Business summary
            total_po_needed = urgent_summary['Recommended PO'].sum()
            st.info(f"💡 **Action Required:** {len(urgent_cities)} cities need immediate restocking. Total recommended PO: {total_po_needed} units.")
            
            with st.expander('🔧 Technical Details (for Developers)'):
                for _, row in urgent_cities.iterrows():
                    payload = generate_webhook_payload(row['product_id'], qty=int(row['avg_7'] * 7), priority='urgent', 
                                                    reason=f"stock_quantity {int(row['stock_quantity'])} < 20% of 3-day avg ({row['avg_3']:.1f})")
                    st.write(f"**City:** {row['city_name']}")
                    st.json(payload)
                    st.write("---")
            
            # Email Purchase Order Section
            st.markdown('---')
            st.subheader('📧 Send Purchase Order via Email')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                recipient_email = st.text_input(
                    'Recipient Email',
                    value='rathikmohamed786@gmail.com',
                    help='Email address to send the Purchase Order'
                )
                
                priority = st.selectbox(
                    'Priority Level',
                    options=['HIGH', 'MEDIUM', 'LOW'],
                    index=0,
                    help='Priority level for the Purchase Order'
                )
            
            with col2:
                st.markdown("### 📋 PO Summary")
                st.metric("Total Quantity", f"{total_po_needed} units")
                st.metric("Product", product_info['product_name'][:30] + "...")
                st.metric("Priority", priority)
            
            if st.button('📤 Send Purchase Order Email', type='primary', use_container_width=True):
                if recipient_email and total_po_needed > 0:
                    try:
                        # Initialize email system
                        email_system = PurchaseOrderEmailSystem()
                        
                        # Create PO data
                        po_data = email_system.create_po_data(
                            product_info=product_info,
                            quantity=total_po_needed,
                            priority=priority
                        )
                        
                        # Send email
                        with st.spinner('Sending Purchase Order email...'):
                            success, message = email_system.send_po_email(po_data, recipient_email)
                        
                        if success:
                            st.success(f'✅ Purchase Order {po_data["po_number"]} sent successfully to {recipient_email}!')
                            st.balloons()
                            
                            # Show PO details
                            with st.expander('📄 Purchase Order Details'):
                                st.json(po_data)
                        else:
                            st.error(f'❌ Failed to send email: {message}')
                            st.info('💡 Check the terminal/console for detailed error information.')
                            
                    except Exception as e:
                        st.error(f'❌ Error sending email: {str(e)}')
                        st.info('💡 Make sure to set up your email password in environment variables.')
                else:
                    st.warning('⚠️ Please enter a valid email address and ensure quantity > 0.')
        else:
            st.success('✅ **All cities have adequate stock levels.** No urgent PO required.')

    # --- Model Performance Tab ---
    with tab3:
        st.header('Model Performance')
        # Prepare data and train model (show metrics)
        target_col = 'units_sold'
        categorical_cols = ['product_id', 'city_name']
        feature_cols = ['product_id', 'city_name', 'avg_3', 'avg_7', 'avg_14', 'stock_quantity', 'days_of_stock']
        X, y = prepare_lgbm_data(daily_full, target_col, categorical_cols, feature_cols)
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = train_lgbm(X_train, y_train, categorical_cols)
        # Optionally, add metrics and feature importance if available
        st.subheader('Training/Validation Split')
        st.write(f'Training samples: {len(X_train)}, Validation samples: {len(X_val)}')
        # If you have metrics and feature importance, display them here
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        # Predict on validation set
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        # Fix MAPE: exclude zero actuals
        nonzero_idx = y_val != 0
        if np.any(nonzero_idx):
            mape = np.mean(np.abs((y_val[nonzero_idx] - y_pred[nonzero_idx]) / y_val[nonzero_idx])) * 100
        else:
            mape = np.nan
        # Add SMAPE
        smape = 100 * np.mean(2 * np.abs(y_val - y_pred) / (np.abs(y_val) + np.abs(y_pred) + 1e-8))
        st.subheader('Validation Metrics')
        st.metric('MAE', f'{mae:.2f}')
        st.metric('RMSE', f'{rmse:.2f}')
        st.metric('MAPE (%)', f'{mape:.2f}' if not np.isnan(mape) else 'N/A')
        st.metric('SMAPE (%)', f'{smape:.2f}')
        # Feature importance
        st.subheader('Feature Importance')
        import pandas as pd
        fi = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importance()})
        fi = fi.sort_values('importance', ascending=False)
        import plotly.express as px
        fig = px.bar(fi, x='importance', y='feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

    # --- Action Log Tab ---
    with tab4:
        st.header('Action Log')
        # Show product mapping file info
        st.subheader('Product Mapping')
        if os.path.exists('product_mapping.json'):
            with open('product_mapping.json', 'r') as f:
                mapping_data = json.load(f)
            st.write(f"Loaded {len(mapping_data)} products")
            st.dataframe(pd.DataFrame(mapping_data), use_container_width=True)
        else:
            st.info('Product mapping not found.')
        
        # Placeholder: In production, this would be loaded from a DB or log file
        if os.path.exists('action_log.json'):
            with open('action_log.json', 'r') as f:
                action_log = json.load(f)
            st.dataframe(pd.DataFrame(action_log), use_container_width=True)
        else:
            st.info('No actions logged yet.')
