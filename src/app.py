#!/usr/bin/env python3
"""
Chatbot-Enhanced Streamlit App for Quick Commerce Operations
Integrates natural language processing with your existing forecasting and email system
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Set Cohere API key
os.environ['COHERE_API_KEY'] = 'AcAOdKNntZDAYXtH7NrtfQOpgFyX9X9J2ORs9ZQK'

from langchain.llms import Cohere
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import load_sales_inventory, complete_date_range
from feature_engineering import add_rolling_features, merge_inventory, compute_days_of_stock
from model import train_lgbm, prophet_forecast, prepare_lgbm_data, train_lstm_for_range, lstm_forecast, lstm_metrics_json, train_lstm_full_and_save, load_lstm_model, lstm_forecast_with_model
from decision_engine import check_urgent, compute_allocation, generate_webhook_payload
from email_system import PurchaseOrderEmailSystem

# --- LangChain Cohere Integration ---
def get_cohere_insight(json_data, user_query):
    cohere = Cohere()
    prompt = PromptTemplate(
        input_variables=["json_data", "user_query"],
        template="""
        You are an expert business analyst. Given the following JSON data and user question, provide a smart, concise, and actionable insight for a business dashboard.\n
        JSON Data: {json_data}\n
        User Question: {user_query}\n
        Insight:
        """
    )
    chain = LLMChain(llm=cohere, prompt=prompt)
    result = chain.run({"json_data": json_data, "user_query": user_query})
    return result

class QuickCommerceChatbot:
    def __init__(self):
        self.sales_data = None
        self.inventory_data = None
        self.daily_full = None
        self.latest = None
        self.product_map = {}
        self.model = None
        self.email_system = PurchaseOrderEmailSystem()
        
    def load_data(self, sales_file, inv_file):
        """Load and process data"""
        try:
            self.sales_data, self.inventory_data = load_sales_inventory(sales_file, inv_file)
            self.daily_full = complete_date_range(self.sales_data)
            self.daily_full = add_rolling_features(self.daily_full)
            self.daily_full = merge_inventory(self.daily_full, self.inventory_data)
            self.daily_full = compute_days_of_stock(self.daily_full)
            # Create latest data with proper grouping
            if 'city_name' in self.daily_full.columns:
                self.latest = self.daily_full.groupby(['product_id', 'city_name'], group_keys=False).apply(lambda df: df.sort_values('date').iloc[-1]).reset_index(drop=True)
            else:
                self.latest = self.daily_full.groupby(['product_id'], group_keys=False).apply(lambda df: df.sort_values('date').iloc[-1]).reset_index(drop=True)
            self._create_product_mapping()
            # --- Load pre-trained LSTM model if available ---
            self.lstm_model, self.lstm_scaler = load_lstm_model('lstm_model.h5', 'lstm_scaler.pkl')
            if self.lstm_model is None or self.lstm_scaler is None:
                st.warning("LSTM model files not found. Please train the model first using: python src/model.py --sales data/sales.csv --inventory data/inventory.csv --sample_size 10000 --epochs 10 --batch_size 32 --model_path lstm_model.h5 --scaler_path lstm_scaler.pkl")
                self.lstm_metrics = None
            else:
                st.success("✅ Pre-trained LSTM model loaded successfully!")
                # Load metrics if available
                try:
                    with open('lstm_metrics.json', 'r') as f:
                        self.lstm_metrics = json.load(f)
                except:
                    self.lstm_metrics = None
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _create_product_mapping(self):
        """Create product ID to name mapping"""
        try:
            # Load existing mapping if available
            if os.path.exists('product_mapping.json'):
                with open('product_mapping.json', 'r') as f:
                    product_mapping_dict = json.load(f)
                self.product_map = {str(item['product_id']): item for item in product_mapping_dict}
            else:
                # Create basic mapping from sales data
                sales_products = self.sales_data.drop_duplicates(subset=['product_id'])[['product_id', 'product_name']].copy()
                self.product_map = {str(pid): {'product_name': name, 'category': 'Unknown', 'sub_category': 'Unknown'} 
                                  for pid, name in zip(sales_products['product_id'], sales_products['product_name'])}
        except Exception as e:
            st.warning(f"Could not load product mapping: {str(e)}")
            self.product_map = {}
    
    def get_product_name(self, product_id):
        """Get product name from ID"""
        return self.product_map.get(str(product_id), {}).get('product_name', f'Product {product_id}')
    
    def get_product_info(self, product_id):
        """Get full product info"""
        return self.product_map.get(str(product_id), {
            'product_name': f'Product {product_id}',
            'category': 'Unknown',
            'sub_category': 'Unknown'
        })
    
    def parse_query(self, query):
        """Parse natural language query to extract intent and parameters"""
        query_lower = query.lower()
        
        # Initialize result
        result = {
            'intent': 'unknown',
            'product_id': None,
            'product_name': None,
            'city': None,
            'quantity': None,
            'action': None,
            'start_date': None,
            'end_date': None,
            'city_1': None,
            'city_2': None,
            'periods': 7  # Default forecast periods
        }
        
        # Extract product information - improved matching
        best_match = None
        best_score = 0
        
        for pid, info in self.product_map.items():
            product_name = info['product_name'].lower()
            # Clean product name (remove price info and extra details)
            clean_product_name = product_name.split(' 1 unit - rs')[0].split(' 1.0 unit - rs')[0].strip()
            
            # Calculate match score
            score = 0
            
            # Exact product ID match
            if pid in query_lower:
                score += 100
            
            # Exact product name match
            if clean_product_name in query_lower:
                score += 80
            
            # Query in product name
            if query_lower in clean_product_name:
                score += 60
            
            # Word-by-word matching (more specific)
            query_words = [word for word in query_lower.split() if len(word) > 3]
            product_words = clean_product_name.split()
            
            # Count matching words
            matching_words = sum(1 for word in query_words if any(word in prod_word for prod_word in product_words))
            if matching_words > 0:
                score += matching_words * 20
            
            # Prefer exact matches over partial matches
            if score > best_score:
                best_match = (pid, info['product_name'])
                best_score = score
        
        if best_match and best_score > 0:
            result['product_id'] = best_match[0]
            result['product_name'] = best_match[1]
        
        # Extract city information
        if self.latest is not None and not self.latest.empty:
            # Check if city_name column exists, otherwise use city column
            city_col = 'city_name' if 'city_name' in self.latest.columns else 'city'
            if city_col in self.latest.columns:
                cities = self.latest[city_col].unique()
                for city in cities:
                    if city.lower() in query_lower:
                        result['city'] = city
                        break
        
        # Extract quantity and periods
        import re
        quantity_match = re.search(r'(\d+)\s*units?', query_lower)
        if quantity_match:
            result['quantity'] = int(quantity_match.group(1))
        
        # Extract forecast periods
        periods_match = re.search(r'(\d+)\s*days?', query_lower)
        if periods_match:
            result['periods'] = int(periods_match.group(1))
        
        # Extract date ranges
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})',
            r'from\s*(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})\s*to\s*(\d{1,2}/\d{1,2}/\d{4})'
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, query_lower)
            if date_match:
                result['start_date'] = date_match.group(1)
                result['end_date'] = date_match.group(2)
                break
        
        # --- LSTM Training Intent ---
        if 'train' in query_lower and 'lstm' in query_lower:
            result['intent'] = 'lstm_train'
            # Extract date range
            date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', query)
            if len(date_matches) >= 2:
                result['start_date'] = date_matches[0]
                result['end_date'] = date_matches[1]
            # Extract product/city if present
            for city in self.latest['city_name'].unique():
                if city.lower() in query_lower:
                    result['city'] = city
            for pid, info in self.product_map.items():
                if info['product_name'].lower() in query_lower:
                    result['product_id'] = pid
            return result
        # --- LSTM Prediction Intent ---
        if 'predict' in query_lower and 'lstm' in query_lower:
            result['intent'] = 'lstm_predict'
            for city in self.latest['city_name'].unique():
                if city.lower() in query_lower:
                    result['city'] = city
            for pid, info in self.product_map.items():
                if info['product_name'].lower() in query_lower:
                    result['product_id'] = pid
            return result
        # --- City Comparison Intent ---
        if 'compare' in query_lower and 'sales' in query_lower:
            result['intent'] = 'compare_cities'
            # Extract two cities
            cities = [city for city in self.latest['city_name'].unique() if city.lower() in query_lower]
            if len(cities) >= 2:
                result['city_1'] = cities[0]
                result['city_2'] = cities[1]
            return result
        # Determine intent (order matters - more specific intents first)
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'between']):
            result['intent'] = 'compare_cities'
        elif any(word in query_lower for word in ['forecast', 'predict', 'future', 'demand', 'sales prediction', 'forecasting']):
            result['intent'] = 'lstm_predict'
        elif any(word in query_lower for word in ['allocate', 'distribute', 'allocation', 'how to distribute']):
            result['intent'] = 'allocation'
            result['action'] = 'allocation'
        elif any(word in query_lower for word in ['send', 'email', 'notify']):
            result['intent'] = 'email'
            result['action'] = 'send_email'
        elif any(word in query_lower for word in ['urgent', 'low stock', 'reorder']) and 'email' not in query_lower:
            result['intent'] = 'urgent_stock'
            result['action'] = 'urgent_po'
        elif any(word in query_lower for word in ['purchase order', 'po']) and 'email' not in query_lower:
            result['intent'] = 'urgent_stock'
            result['action'] = 'urgent_po'
        elif any(word in query_lower for word in ['sales', 'performance', 'analysis', 'show me']):
            result['intent'] = 'analysis'
            result['action'] = 'analysis'
        
        return result
    
    def create_equal_allocation(self, product_id, quantity):
        """Create equal allocation for products without sales data"""
        # Get all cities from inventory data
        cities = self.inventory_data[self.inventory_data['product_id'] == product_id]['city_name'].unique()
        
        if len(cities) == 0:
            return "❌ No cities found for this product in inventory."
        
        # Equal distribution
        base_allocation = quantity // len(cities)
        remainder = quantity % len(cities)
        
        allocation_data = []
        for i, city in enumerate(cities):
            city_allocation = base_allocation + (1 if i < remainder else 0)
            # Get current stock for this city
            city_stock = self.inventory_data[
                (self.inventory_data['product_id'] == product_id) & 
                (self.inventory_data['city_name'] == city)
            ]['stock_quantity'].iloc[0] if len(self.inventory_data[
                (self.inventory_data['product_id'] == product_id) & 
                (self.inventory_data['city_name'] == city)
            ]) > 0 else 0
            
            allocation_data.append({
                'city_name': city,
                'recommended_allocation': city_allocation,
                'stock_quantity': city_stock,
                'avg_7': 0,  # No sales data
                'avg_3': 0   # No sales data
            })
        
        return pd.DataFrame(allocation_data)
    
    def process_allocation_query(self, parsed_query):
        """Process allocation requests"""
        if not parsed_query['product_id']:
            return "❌ Please specify a product name or ID for allocation."
        
        if not parsed_query['quantity']:
            return "❌ Please specify the quantity to allocate (e.g., 'allocate 1000 units of Product A')."
        
        product_id = parsed_query['product_id']
        quantity = parsed_query['quantity']
        product_info = self.get_product_info(product_id)
        
        # Get product data
        product_latest = self.latest[self.latest['product_id'] == product_id]
        
        if product_latest.empty:
            # Check if product exists in inventory but not in sales
            if product_id in self.inventory_data['product_id'].values:
                # Create equal allocation for products without sales data
                allocation = self.create_equal_allocation(product_id, quantity)
                
                # Create beautiful allocation visualization
                fig1, fig2 = self.create_allocation_visualization(allocation, product_info['product_name'], quantity, "Equal Distribution")
                
                # Store charts in session state for display
                st.session_state.allocation_charts = (fig1, fig2, allocation)
                
                # Format response
                response = f"📦 **Product Found in Inventory:** {product_info['product_name']}\n\n"
                response += f"⚠️ **No Sales History Available** - Using equal distribution strategy.\n\n"
                response += f"📦 **Total Quantity to Allocate:** {quantity:,} units\n\n"
                response += f"🏙️ **Distribution Summary:**\n"
                response += f"• **Cities:** {len(allocation)} locations\n"
                response += f"• **Strategy:** Equal distribution\n"
                response += f"• **Average per city:** {quantity // len(allocation):.0f} units\n\n"
                response += f"💡 **Recommendation:** Monitor sales performance to build forecasting data for future allocations."
                
                return response
            else:
                return f"❌ No data found for product: {product_info['product_name']}"
        
        # Compute allocation
        allocation = compute_allocation(product_latest, quantity)
        
        # Create beautiful allocation visualization
        fig1, fig2 = self.create_allocation_visualization(allocation, product_info['product_name'], quantity, "Smart Allocation")
        
        # Store charts in session state for display
        st.session_state.allocation_charts = (fig1, fig2, allocation)
        
        # Format response
        response = f"🎯 **Allocation Plan for {product_info['product_name']}**\n\n"
        response += f"📦 **Total Quantity to Allocate:** {quantity:,} units\n\n"
        response += f"🏙️ **Distribution Summary:**\n"
        response += f"• **Cities:** {len(allocation)} locations\n"
        response += f"• **Strategy:** Smart allocation based on 7-day sales average\n"
        response += f"• **Top 3 cities:** {', '.join(allocation.nlargest(3, 'recommended_allocation')['city_name'].tolist())}\n\n"
        
        # Check for urgent cities
        urgent_cities = allocation[allocation.apply(lambda r: check_urgent(r['stock_quantity'], r['avg_3']), axis=1)]
        if not urgent_cities.empty:
            response += f"🚨 **Urgent Stock Alert:** {len(urgent_cities)} cities need immediate restocking!\n"
            response += f"• **Most urgent:** {urgent_cities.iloc[0]['city_name']} (Only {int(urgent_cities.iloc[0]['stock_quantity'])} units left)\n\n"
        
        response += f"💡 **Next Steps:** Review the charts below for detailed city-wise breakdown and current stock levels."
        
        return response
    
    def process_forecast_query(self, parsed_query):
        """Process forecasting requests with beautiful visualizations"""
        if parsed_query['product_id']:
            # Single product forecast
            product_id = parsed_query['product_id']
            product_info = self.get_product_info(product_id)
            
            # Get product data
            product_data = self.daily_full[self.daily_full['product_id'] == product_id]
            
            if product_data.empty:
                return f"❌ No data found for product: {product_info['product_name']}"
            
            # Generate forecast
            forecast_data = []
            cities = product_data['city_name'].unique()
            
            for city in cities[:5]:  # Limit to top 5 cities
                city_data = product_data[product_data['city_name'] == city].sort_values('date')
                if len(city_data) >= 7:
                    forecast = prophet_forecast(city_data, periods=7)
                    if forecast is not None:
                        forecast_data.append({
                            'city': city,
                            'forecast': forecast
                        })
            
            if not forecast_data:
                return f"❌ Insufficient data for forecasting {product_info['product_name']}"
            
            # Format response
            response = f"🔮 **7-Day Sales Forecast for {product_info['product_name']}**\n\n"
            
            for data in forecast_data:
                city = data['city']
                forecast = data['forecast']
                total_forecast = forecast['yhat'].sum()
                
                response += f"🏙️ **{city}:**\n"
                response += f"   • Total 7-day forecast: {total_forecast:.0f} units\n"
                response += f"   • Daily average: {total_forecast/7:.1f} units/day\n\n"
            
            return response
        else:
            # All products forecast
            response = "🔮 **7-Day Sales Forecast for All Products**\n\n"
            
            # Get top 10 products by total sales
            top_products = self.latest.groupby('product_id')['units_sold'].sum().nlargest(10)
            
            forecast_summary = []
            total_forecast_all = 0
            
            for product_id, total_sales in top_products.items():
                product_info = self.get_product_info(product_id)
                product_data = self.daily_full[self.daily_full['product_id'] == product_id]
                
                if not product_data.empty:
                    # Get top city for this product
                    top_city = product_data.groupby('city_name')['units_sold'].sum().idxmax()
                    city_data = product_data[product_data['city_name'] == top_city].sort_values('date')
                    
                    if len(city_data) >= 7:
                        forecast = prophet_forecast(city_data, periods=7)
                        if forecast is not None:
                            total_forecast = forecast['yhat'].sum()
                            total_forecast_all += total_forecast
                            
                            forecast_summary.append({
                                'product_name': product_info['product_name'][:40] + '...' if len(product_info['product_name']) > 40 else product_info['product_name'],
                                'city': top_city,
                                'forecast': total_forecast,
                                'daily_avg': total_forecast/7
                            })
            
            if not forecast_summary:
                return "❌ Insufficient data for forecasting any products."
            
            # Sort by forecast amount
            forecast_summary.sort(key=lambda x: x['forecast'], reverse=True)
            
            response += f"📊 **Summary:** {len(forecast_summary)} products forecasted\n"
            response += f"📈 **Total 7-day forecast:** {total_forecast_all:,.0f} units\n\n"
            
            response += f"🏆 **Top 5 Products by Forecast:**\n"
            for i, item in enumerate(forecast_summary[:5], 1):
                response += f"**{i}. {item['product_name']}**\n"
                response += f"   🏙️ **Top City:** {item['city']}\n"
                response += f"   📦 **7-day forecast:** {item['forecast']:,.0f} units\n"
                response += f"   📊 **Daily average:** {item['daily_avg']:.1f} units/day\n\n"
            
            if len(forecast_summary) > 5:
                response += f"📋 **+ {len(forecast_summary) - 5} more products forecasted**\n\n"
            
            response += f"💡 **Recommendation:** Focus on top 3 products for inventory planning and allocation decisions."
            
            return response
    
    def process_urgent_stock_query(self, parsed_query):
        """Process urgent stock requests with beautiful UI"""
        if not parsed_query['product_id']:
            # Check all products for urgent stock
            urgent_products = []
            for product_id in self.latest['product_id'].unique():
                product_latest = self.latest[self.latest['product_id'] == product_id]
                urgent_cities = product_latest[product_latest.apply(lambda r: check_urgent(r['stock_quantity'], r['avg_3']), axis=1)]
                if not urgent_cities.empty:
                    product_info = self.get_product_info(product_id)
                    urgent_products.append({
                        'product_id': product_id,
                        'product_name': product_info['product_name'],
                        'urgent_cities': urgent_cities,
                        'total_urgent_qty': urgent_cities['avg_7'].sum() * 7
                    })
            
            if not urgent_products:
                return "✅ **All products have adequate stock levels.** No urgent restocking needed."
            
            # Sort by urgency (total quantity needed)
            urgent_products.sort(key=lambda x: x['total_urgent_qty'], reverse=True)
            
            # Create beautiful response with metrics
            response = "🚨 **URGENT STOCK ALERTS**\n\n"
            response += f"📊 **Summary:** {len(urgent_products)} products need immediate attention\n\n"
            
            # Top 5 most urgent products
            response += "🔥 **TOP 5 MOST URGENT PRODUCTS:**\n\n"
            for i, product in enumerate(urgent_products[:5], 1):
                response += f"**{i}. {product['product_name'][:50]}...**\n"
                response += f"   📦 **Total Needed:** {int(product['total_urgent_qty']):,} units\n"
                response += f"   🏙️ **Cities Affected:** {len(product['urgent_cities'])}\n"
                response += f"   🚨 **Priority:** {'CRITICAL' if product['total_urgent_qty'] > 1000 else 'HIGH' if product['total_urgent_qty'] > 500 else 'MEDIUM'}\n\n"
            
            if len(urgent_products) > 5:
                response += f"📋 **+ {len(urgent_products) - 5} more products need attention**\n\n"
            
            response += "💡 **Recommended Actions:**\n"
            response += "• Generate Purchase Orders for top 3 products immediately\n"
            response += "• Reallocate stock from overstocked cities\n"
            response += "• Set up automated alerts for future stockouts\n"
            
            return response
        else:
            # Check specific product
            product_id = parsed_query['product_id']
            product_info = self.get_product_info(product_id)
            product_latest = self.latest[self.latest['product_id'] == product_id]
            
            urgent_cities = product_latest[product_latest.apply(lambda r: check_urgent(r['stock_quantity'], r['avg_3']), axis=1)]
            
            if urgent_cities.empty:
                return f"✅ **{product_info['product_name']}** has adequate stock in all cities."
            
            response = f"🚨 **URGENT STOCK ALERT for {product_info['product_name']}**\n\n"
            total_urgent_qty = urgent_cities['avg_7'].sum() * 7
            
            for _, city in urgent_cities.iterrows():
                response += f"• **{city['city_name']}:** Only {int(city['stock_quantity'])} units left\n"
            
            response += f"\n💡 **Recommended Action:** Generate PO for {int(total_urgent_qty)} units"
            
            return response
    
    def process_analysis_query(self, parsed_query):
        """Process analysis requests with beautiful visualizations"""
        if parsed_query['city']:
            # City-specific analysis
            city = parsed_query['city']
            city_data = self.latest[self.latest['city_name'] == city]
            
            if city_data.empty:
                return f"❌ No data found for city: {city}"
            
            # Create beautiful city analysis visualization
            fig1, fig2 = self.create_city_analysis_visualization(city_data, city)
            
            # Store charts in session state for display
            st.session_state.analysis_charts = (fig1, fig2, city_data)
            
            total_sales = city_data['units_sold'].sum()
            total_stock = city_data['stock_quantity'].sum()
            unique_products = city_data['product_id'].nunique()
            avg_7_day = city_data['avg_7'].mean()
            
            # Top products in this city
            top_products = city_data.nlargest(5, 'units_sold')[['product_id', 'units_sold', 'stock_quantity']]
            
            response = f"🏙️ **Sales Performance Analysis for {city}**\n\n"
            response += f"📈 **Key Metrics:**\n"
            response += f"• Total Sales: {total_sales:,.0f} units\n"
            response += f"• Total Stock: {total_stock:,.0f} units\n"
            response += f"• Products Available: {unique_products}\n"
            response += f"• Average 7-day Sales: {avg_7_day:.1f} units/day\n\n"
            
            response += f"🏆 **Top 5 Products in {city}:**\n"
            for _, product in top_products.iterrows():
                product_name = self.get_product_name(product['product_id'])
                response += f"• {product_name[:40]}...: {int(product['units_sold'])} units sold, {int(product['stock_quantity'])} in stock\n"
            
            response += f"\n💡 **Next Steps:** Review the charts below for detailed product performance and stock analysis."
            
            return response
            
        elif parsed_query['product_id']:
            # Product-specific analysis
            product_id = parsed_query['product_id']
            product_info = self.get_product_info(product_id)
            product_latest = self.latest[self.latest['product_id'] == product_id]
            
            if product_latest.empty:
                return f"❌ No data found for product: {product_info['product_name']}"
            
            total_sales = product_latest['units_sold'].sum()
            total_stock = product_latest['stock_quantity'].sum()
            avg_7_day = product_latest['avg_7'].mean()
            
            # Top cities
            top_cities = product_latest.nlargest(5, 'units_sold')[['city_name', 'units_sold', 'stock_quantity']]
            
            response = f"📊 **Analysis for {product_info['product_name']}**\n\n"
            response += f"📈 **Key Metrics:**\n"
            response += f"• Total Sales: {total_sales:,.0f} units\n"
            response += f"• Total Stock: {total_stock:,.0f} units\n"
            response += f"• Average 7-day Sales: {avg_7_day:.1f} units/day\n\n"
            
            response += f"🏆 **Top Performing Cities:**\n"
            for _, city in top_cities.iterrows():
                response += f"• {city['city_name']}: {int(city['units_sold'])} units sold, {int(city['stock_quantity'])} in stock\n"
            
            return response
        else:
            # Overall analysis
            total_sales = self.latest['units_sold'].sum()
            total_stock = self.latest['stock_quantity'].sum()
            unique_products = self.latest['product_id'].nunique()
            unique_cities = self.latest['city_name'].nunique()
            
            # Top products
            top_products = self.latest.groupby('product_id')['units_sold'].sum().nlargest(5)
            
            response = "📊 **Overall Business Analysis**\n\n"
            response += f"📈 **Key Metrics:**\n"
            response += f"• Total Sales: {total_sales:,.0f} units\n"
            response += f"• Total Stock: {total_stock:,.0f} units\n"
            response += f"• Products: {unique_products}\n"
            response += f"• Cities: {unique_cities}\n\n"
            
            response += f"🏆 **Top Products:**\n"
            for product_id, sales in top_products.items():
                product_name = self.get_product_name(product_id)
                response += f"• {product_name}: {sales:,.0f} units\n"
            
            return response
    
    def process_email_query(self, parsed_query):
        """Process email sending requests"""
        if not parsed_query['product_id']:
            return "❌ Please specify a product for sending Purchase Order email."
        
        product_id = parsed_query['product_id']
        product_info = self.get_product_info(product_id)
        product_latest = self.latest[self.latest['product_id'] == product_id]
        
        urgent_cities = product_latest[product_latest.apply(lambda r: check_urgent(r['stock_quantity'], r['avg_3']), axis=1)]
        
        if urgent_cities.empty:
            return f"✅ **{product_info['product_name']}** has adequate stock. No urgent PO needed."
        
        # Calculate urgent quantity
        total_urgent_qty = urgent_cities['avg_7'].sum() * 7
        
        try:
            # Create PO data with product_id included
            po_data = self.email_system.create_po_data(
                product_info={
                    'product_id': product_id,
                    'product_name': product_info['product_name'],
                    'category': product_info.get('category', 'Unknown'),
                    'sub_category': product_info.get('sub_category', 'Unknown'),
                    'mrp': 100  # Default MRP
                },
                quantity=int(total_urgent_qty),
                priority='HIGH'
            )
            
            # Send email
            success, message = self.email_system.send_po_email(po_data, "rathikmohamed786@gmail.com")
            
            if success:
                response = f"✅ **Purchase Order Email Sent Successfully!**\n\n"
                response += f"📧 **Email Details:**\n"
                response += f"• PO Number: {po_data['po_number']}\n"
                response += f"• Product: {product_info['product_name']}\n"
                response += f"• Quantity: {int(total_urgent_qty)} units\n"
                response += f"• Priority: HIGH\n"
                response += f"• Recipient: rathikmohamed786@gmail.com\n\n"
                response += f"🎉 **Email sent successfully!**"
            else:
                response = f"❌ **Email sending failed:** {message}\n\n"
                response += f"💡 **Fallback:** Email content saved as HTML file for manual sending."
            
            return response
            
        except Exception as e:
            return f"❌ **Error sending email:** {str(e)}"
    
    def create_urgent_stock_visualization(self, urgent_products):
        """Create beautiful charts for urgent stock data"""
        if not urgent_products:
            return None
        
        # Prepare data for charts
        top_products = urgent_products[:10]  # Top 10 most urgent
        
        # Chart 1: Top Urgent Products by Quantity Needed
        product_names = [p['product_name'][:30] + '...' if len(p['product_name']) > 30 else p['product_name'] for p in top_products]
        quantities = [p['total_urgent_qty'] for p in top_products]
        
        fig1 = go.Figure(data=[
            go.Bar(
                x=product_names,
                y=quantities,
                marker_color=['#ff4444' if q > 1000 else '#ff8800' if q > 500 else '#ffaa00' for q in quantities],
                text=[f"{int(q):,}" for q in quantities],
                textposition='auto',
            )
        ])
        
        fig1.update_layout(
            title="🚨 Top 10 Most Urgent Products",
            xaxis_title="Products",
            yaxis_title="Units Needed",
            height=400,
            showlegend=False,
            xaxis={'tickangle': 45}
        )
        
        # Chart 2: Priority Distribution
        critical = len([p for p in urgent_products if p['total_urgent_qty'] > 1000])
        high = len([p for p in urgent_products if 500 < p['total_urgent_qty'] <= 1000])
        medium = len([p for p in urgent_products if p['total_urgent_qty'] <= 500])
        
        fig2 = go.Figure(data=[
            go.Pie(
                labels=['Critical (>1000)', 'High (500-1000)', 'Medium (<500)'],
                values=[critical, high, medium],
                marker_colors=['#ff4444', '#ff8800', '#ffaa00'],
                hole=0.3
            )
        ])
        
        fig2.update_layout(
            title="📊 Priority Distribution",
            height=400,
            showlegend=True
        )
        
        return fig1, fig2
    
    def create_allocation_visualization(self, allocation, product_name, total_qty, strategy):
        """Create beautiful charts for allocation data"""
        if allocation.empty:
            return None, None
        
        # Prepare data for charts
        top_cities = allocation.nlargest(15, 'recommended_allocation')  # Top 15 cities
        
        # Chart 1: Top Cities by Allocation
        city_names = [city[:15] + '...' if len(city) > 15 else city for city in top_cities['city_name']]
        allocations = top_cities['recommended_allocation'].tolist()
        current_stocks = top_cities['stock_quantity'].tolist()
        
        fig1 = go.Figure()
        
        # Add allocation bars
        fig1.add_trace(go.Bar(
            name='Allocated',
            x=city_names,
            y=allocations,
            marker_color='#2E86AB',
            text=[f"{int(a)}" for a in allocations],
            textposition='auto',
        ))
        
        # Add current stock bars
        fig1.add_trace(go.Bar(
            name='Current Stock',
            x=city_names,
            y=current_stocks,
            marker_color='#A23B72',
            text=[f"{int(s)}" for s in current_stocks],
            textposition='auto',
        ))
        
        fig1.update_layout(
            title=f"🏙️ Top 15 Cities - {strategy}",
            xaxis_title="Cities",
            yaxis_title="Units",
            height=500,
            barmode='group',
            xaxis={'tickangle': 45},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Chart 2: Stock vs Allocation Scatter
        fig2 = go.Figure(data=go.Scatter(
            x=allocation['stock_quantity'],
            y=allocation['recommended_allocation'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=allocation['recommended_allocation'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Allocation")
            ),
            text=allocation['city_name'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Current Stock: %{x}<br>' +
                         'Allocated: %{y}<br>' +
                         '<extra></extra>'
        ))
        
        fig2.update_layout(
            title="📊 Stock vs Allocation Analysis",
            xaxis_title="Current Stock",
            yaxis_title="Allocated Units",
            height=500,
            showlegend=False
        )
        
        return fig1, fig2
    
    def create_city_analysis_visualization(self, city_data, city_name):
        """Create beautiful charts for city analysis data"""
        if city_data.empty:
            return None, None
        
        # Prepare data for charts
        top_products = city_data.nlargest(10, 'units_sold')  # Top 10 products
        
        # Chart 1: Top Products by Sales
        product_names = [self.get_product_name(pid)[:25] + '...' if len(self.get_product_name(pid)) > 25 else self.get_product_name(pid) for pid in top_products['product_id']]
        sales = top_products['units_sold'].tolist()
        stocks = top_products['stock_quantity'].tolist()
        
        fig1 = go.Figure()
        
        # Add sales bars
        fig1.add_trace(go.Bar(
            name='Sales',
            x=product_names,
            y=sales,
            marker_color='#2E86AB',
            text=[f"{int(s)}" for s in sales],
            textposition='auto',
        ))
        
        # Add stock bars
        fig1.add_trace(go.Bar(
            name='Current Stock',
            x=product_names,
            y=stocks,
            marker_color='#A23B72',
            text=[f"{int(s)}" for s in stocks],
            textposition='auto',
        ))
        
        fig1.update_layout(
            title=f"🏙️ Top 10 Products in {city_name}",
            xaxis_title="Products",
            yaxis_title="Units",
            height=500,
            barmode='group',
            xaxis={'tickangle': 45},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Chart 2: Sales vs Stock Scatter
        fig2 = go.Figure(data=go.Scatter(
            x=city_data['stock_quantity'],
            y=city_data['units_sold'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=city_data['units_sold'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sales")
            ),
            text=[self.get_product_name(pid)[:15] + '...' if len(self.get_product_name(pid)) > 15 else self.get_product_name(pid) for pid in city_data['product_id']],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Current Stock: %{x}<br>' +
                         'Sales: %{y}<br>' +
                         '<extra></extra>'
        ))
        
        fig2.update_layout(
            title="📊 Sales vs Stock Analysis",
            xaxis_title="Current Stock",
            yaxis_title="Units Sold",
            height=500,
            showlegend=False
        )
        
        return fig1, fig2
    
    def train_lstm_on_range(self, start_date, end_date, product_id=None, city_name=None):
        # Filter data for product/city if specified
        df = self.daily_full.copy()
        if product_id:
            df = df[df['product_id'] == product_id]
        if city_name:
            df = df[df['city_name'] == city_name]
        result = train_lstm_for_range(df, start_date, end_date)
        if 'error' in result:
            return {'error': result['error']}
        self.lstm_model = result['model']
        self.lstm_scaler = result['scaler']
        self.lstm_metrics = result['metrics']
        return result['metrics']

    def lstm_predict(self, product_id=None, city_name=None, start_date=None, end_date=None, periods=7):
        df = self.daily_full.copy()
        
        # Filter by product and city if specified (but NOT by date range for training data)
        if product_id:
            df = df[df['product_id'] == product_id]
        if city_name:
            df = df[df['city_name'] == city_name]
        
        # Don't filter by date range here - we need historical data for LSTM
        # The date range is for the forecast period, not the training data
        
        if self.lstm_model is None or self.lstm_scaler is None:
            return {'error': 'LSTM model not loaded. Please train the model first.'}
        
        if len(df) < 14:  # Need at least lookback period
            return {'error': f'Insufficient data for prediction. Need at least 14 days, got {len(df)} days.'}
        
        # Calculate periods based on date range if provided
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            periods = (end_dt - start_dt).days + 1
        
        feature_cols = ['avg_3', 'avg_7', 'avg_14', 'stock_quantity', 'days_of_stock']
        try:
            yhat = lstm_forecast_with_model(df, self.lstm_model, self.lstm_scaler, feature_cols, periods=periods)
            return {
                'forecast': yhat,
                'product_id': product_id,
                'city_name': city_name,
                'periods': periods,
                'data_points_used': len(df),
                'forecast_start_date': start_date,
                'forecast_end_date': end_date
            }
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def process_query(self, query):
        parsed_query = self.parse_query(query)
        
        if parsed_query.get('intent') == 'lstm_predict':
            product_id = parsed_query.get('product_id')
            city_name = parsed_query.get('city')
            start_date = parsed_query.get('start_date')
            end_date = parsed_query.get('end_date')
            periods = parsed_query.get('periods', 7)
            
            result = self.lstm_predict(product_id, city_name, start_date, end_date, periods)
            json_out = json.dumps(result, indent=2)
            
            try:
                insight = get_cohere_insight(json_out, query)
                return json.dumps({"forecast": result, "insight": insight}, indent=2)
            except Exception as e:
                return json.dumps({"forecast": result, "insight": f"Cohere AI unavailable: {str(e)}"}, indent=2)
                
        elif parsed_query.get('intent') == 'allocation':
            result = self.handle_allocation(parsed_query)
            json_out = json.dumps(result, indent=2)
            try:
                insight = get_cohere_insight(json_out, query)
                return json.dumps({"allocation": result, "insight": insight}, indent=2)
            except Exception as e:
                return json.dumps({"allocation": result, "insight": f"Cohere AI unavailable: {str(e)}"}, indent=2)
                
        elif parsed_query.get('intent') == 'compare_cities':
            city1 = parsed_query.get('city_1')
            city2 = parsed_query.get('city_2')
            df = self.latest.copy()
            sales1 = df[df['city_name'] == city1]['units_sold'].sum()
            sales2 = df[df['city_name'] == city2]['units_sold'].sum()
            result = {
                'city_1': city1,
                'city_2': city2,
                'sales_1': int(sales1),
                'sales_2': int(sales2),
                'winner': city1 if sales1 > sales2 else city2 if sales2 > sales1 else 'Tie',
                'difference': int(abs(sales1 - sales2))
            }
            json_out = json.dumps(result, indent=2)
            
            try:
                insight = get_cohere_insight(json_out, query)
                return json.dumps({"comparison": result, "insight": insight}, indent=2)
            except Exception as e:
                return json.dumps({"comparison": result, "insight": f"Cohere AI unavailable: {str(e)}"}, indent=2)
        
        # Fallback to existing logic for other intents
        return "❓ I didn't understand your request. Please try rephrasing or use a supported command."

    def show_lstm_metrics(self, st, metrics_json):
        metrics = json.loads(metrics_json)
        st.subheader('LSTM Model Performance Metrics')
        for k, v in metrics.items():
            st.metric(k, f'{v:.2f}')
        st.json(metrics)

    def handle_allocation(self, parsed_query):
        """Process allocation requests"""
        if not parsed_query['product_id']:
            return {'error': 'Please specify a product name or ID for allocation.'}
        
        if not parsed_query['quantity']:
            return {'error': 'Please specify the quantity to allocate (e.g., "allocate 1000 units of Product A").'}
        
        product_id = parsed_query['product_id']
        quantity = parsed_query['quantity']
        product_info = self.get_product_info(product_id)
        
        if not product_info:
            return {'error': f'Product {product_id} not found.'}
        
        # Get current stock levels for all cities
        product_data = self.latest[self.latest['product_id'] == product_id]
        
        if product_data.empty:
            return {'error': f'No data found for product {product_id}.'}
        
        # Calculate allocation using the decision engine
        allocation_result = compute_allocation(product_data, quantity)
        
        return {
            'product_id': product_id,
            'product_name': product_info['product_name'],
            'total_quantity': quantity,
            'allocation': allocation_result.to_dict('records')
        }

    def show_ai_insight(self, st, insight):
        st.subheader('AI Insight (Cohere)')
        st.info(insight)

    def create_forecast_chart(self, forecast_data, product_id=None, city_name=None):
        """Create beautiful forecast visualization"""
        if 'error' in forecast_data:
            return None
        
        forecast_values = forecast_data.get('forecast', [])
        periods = forecast_data.get('periods', 7)
        
        # Create future dates
        if forecast_data.get('forecast_start_date') and forecast_data.get('forecast_end_date'):
            # Use the specified date range
            start_date = pd.to_datetime(forecast_data['forecast_start_date'])
            end_date = pd.to_datetime(forecast_data['forecast_end_date'])
            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            # Use default future dates from last data point
            if city_name and product_id:
                # Get last date from filtered data
                df = self.daily_full.copy()
                if product_id:
                    df = df[df['product_id'] == product_id]
                if city_name:
                    df = df[df['city_name'] == city_name]
                last_date = df['date'].max()
            else:
                last_date = self.daily_full['date'].max()
            
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        # Create chart with enhanced styling
        fig = go.Figure()
        
        # Add historical data for context (last 30 days)
        if city_name and product_id:
            df = self.daily_full.copy()
            if product_id:
                df = df[df['product_id'] == product_id]
            if city_name:
                df = df[df['city_name'] == city_name]
            df = df.sort_values('date').tail(30)  # Last 30 days
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['units_sold'],
                mode='lines+markers',
                name='📊 Historical Sales',
                line=dict(color='#A23B72', width=3),
                marker=dict(size=6, color='#A23B72'),
                opacity=0.8
            ))
        
        # Add forecast data
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_values,
            mode='lines+markers',
            name='🔮 LSTM Forecast',
            line=dict(color='#2E86AB', width=4),
            marker=dict(size=8, color='#2E86AB', symbol='diamond')
        ))
        
        # Add confidence interval (simplified)
        if len(forecast_values) > 0:
            upper_bound = [x * 1.2 for x in forecast_values]
            lower_bound = [x * 0.8 for x in forecast_values]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(46, 134, 171, 0.2)',
                name='Confidence Interval',
                hoverinfo='skip'
            ))
        
        # Enhanced layout
        title_text = f"📈 Sales Forecast"
        if city_name:
            title_text += f" - {city_name.title()}"
        if product_id:
            title_text += f" - Product {product_id}"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                font=dict(size=20, color='#2C3E50')
            ),
            xaxis=dict(
                title=dict(text="📅 Date", font=dict(size=14, color='#34495E')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title=dict(text="📦 Units Sold", font=dict(size=14, color='#34495E')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Add annotations
        if len(forecast_values) > 0:
            max_forecast = max(forecast_values)
            max_date = future_dates[forecast_values.index(max_forecast)]
            
            fig.add_annotation(
                x=max_date,
                y=max_forecast,
                text=f"Peak: {max_forecast:.0f} units",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#E74C3C',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E74C3C',
                borderwidth=1
            )
        
        return fig

    def create_comparison_chart(self, comparison_data):
        """Create beautiful city comparison chart"""
        if 'error' in comparison_data:
            return None
        
        city1 = comparison_data.get('city_1', 'City 1')
        city2 = comparison_data.get('city_2', 'City 2')
        sales1 = comparison_data.get('sales_1', 0)
        sales2 = comparison_data.get('sales_2', 0)
        winner = comparison_data.get('winner', '')
        difference = comparison_data.get('difference', 0)
        
        # Determine colors based on winner
        color1 = '#27AE60' if city1.lower() == winner.lower() else '#E74C3C'
        color2 = '#27AE60' if city2.lower() == winner.lower() else '#E74C3C'
        
        fig = go.Figure(data=[
            go.Bar(
                x=[city1.title(), city2.title()],
                y=[sales1, sales2],
                marker_color=[color1, color2],
                text=[f'{sales1:,}', f'{sales2:,}'],
                textposition='outside',
                textfont=dict(size=14, color='#2C3E50'),
                marker_line=dict(width=2, color='#34495E'),
                hovertemplate='<b>%{x}</b><br>Sales: %{y:,}<br><extra></extra>'
            )
        ])
        
        # Add difference annotation
        max_sales = max(sales1, sales2)
        fig.add_annotation(
            x=0.5,
            y=max_sales * 1.1,
            text=f"Difference: {difference:,} units",
            showarrow=False,
            font=dict(size=16, color='#2C3E50'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#34495E',
            borderwidth=1
        )
        
        fig.update_layout(
            title=dict(
                text=f"🏆 Sales Comparison: {city1.title()} vs {city2.title()}",
                x=0.5,
                font=dict(size=20, color='#2C3E50')
            ),
            xaxis=dict(
                title=dict(text="🏙️ Cities", font=dict(size=14, color='#34495E')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title=dict(text="💰 Total Sales", font=dict(size=14, color='#34495E')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            height=500,
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig

    def create_allocation_visualization(self, allocation_data):
        """Create allocation visualization charts"""
        if 'error' in allocation_data:
            return None
        
        allocation_df = pd.DataFrame(allocation_data['allocation'])
        
        # Chart 1: Current Stock vs Allocated
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=allocation_df['city_name'],
            y=allocation_df['stock_quantity'],
            name='Current Stock',
            marker_color='#E74C3C',
            text=allocation_df['stock_quantity'].astype(str),
            textposition='outside'
        ))
        
        fig1.add_trace(go.Bar(
            x=allocation_df['city_name'],
            y=allocation_df['recommended_allocation'],
            name='Recommended Allocation',
            marker_color='#27AE60',
            text=allocation_df['recommended_allocation'].astype(str),
            textposition='outside'
        ))
        
        fig1.update_layout(
            title=dict(text=f"📦 Stock Allocation: {allocation_data['product_name']}", x=0.5),
            xaxis=dict(title="Cities"),
            yaxis=dict(title="Units"),
            barmode='group',
            height=400,
            template='plotly_white'
        )
        
        # Chart 2: Allocation Distribution Pie Chart
        fig2 = go.Figure(data=[go.Pie(
            labels=allocation_df['city_name'],
            values=allocation_df['recommended_allocation'],
            hole=0.3,
            textinfo='label+percent+value'
        )])
        
        fig2.update_layout(
            title=dict(text=f"🎯 Allocation Distribution: {allocation_data['product_name']}", x=0.5),
            height=400,
            template='plotly_white'
        )
        
        return (fig1, fig2, allocation_df)

    def format_response_for_user(self, response):
        """Format the AI response into a user-friendly format"""
        try:
            # Try to parse as JSON first
            if isinstance(response, str):
                response_data = json.loads(response)
            else:
                response_data = response
            
            # Format comparison responses
            if 'comparison' in response_data:
                comparison = response_data['comparison']
                insight = response_data.get('insight', '')
                
                city1 = comparison.get('city_1', '').title()
                city2 = comparison.get('city_2', '').title()
                sales1 = comparison.get('sales_1', 0)
                sales2 = comparison.get('sales_2', 0)
                winner = comparison.get('winner', '').title()
                difference = comparison.get('difference', 0)
                
                # Calculate percentage difference
                percentage_diff = (difference / max(sales1, sales2)) * 100 if max(sales1, sales2) > 0 else 0
                
                formatted = f"""
## 🏆 **Sales Comparison Results**

### 📊 **Performance Summary**
- **{city1}**: {sales1:,} units sold
- **{city2}**: {sales2:,} units sold
- **Winner**: {winner} 🥇
- **Difference**: {difference:,} units ({percentage_diff:.1f}% higher)

### 📈 **Key Insights**
{insight}

---
*Data analyzed using advanced AI algorithms*
"""
                return formatted
            
            # Format forecast responses
            elif 'forecast' in response_data:
                forecast = response_data['forecast']
                insight = response_data.get('insight', '')
                
                if 'error' in forecast:
                    return f"❌ **Forecast Error**: {forecast['error']}"
                
                product_id = forecast.get('product_id', 'All Products')
                city_name = forecast.get('city_name', 'All Cities')
                periods = forecast.get('periods', 7)
                forecast_values = forecast.get('forecast', [])
                
                if forecast_values:
                    avg_forecast = sum(forecast_values) / len(forecast_values)
                    max_forecast = max(forecast_values)
                    min_forecast = min(forecast_values)
                    
                    # Safe city name formatting
                    location_display = city_name.title() if city_name and city_name != 'All Cities' else 'All Cities'
                    
                    formatted = f"""
## 🔮 **Sales Forecast Results**

### 📋 **Forecast Details**
- **Product**: {product_id}
- **Location**: {location_display}
- **Period**: {periods} days
- **Average Daily Sales**: {avg_forecast:.1f} units
- **Peak Sales**: {max_forecast:.1f} units
- **Minimum Sales**: {min_forecast:.1f} units

### 📊 **Daily Forecast**
"""
                    for i, value in enumerate(forecast_values, 1):
                        formatted += f"- **Day {i}**: {value:.1f} units\n"
                    
                    formatted += f"""
### 🧠 **AI Insights**
{insight}

---
*Powered by LSTM Neural Network*
"""
                    return formatted
            
            # Format allocation responses
            elif 'allocation' in response_data:
                allocation = response_data['allocation']
                insight = response_data.get('insight', '')
                
                if 'error' in allocation:
                    return f"❌ **Allocation Error**: {allocation['error']}"
                
                product_name = allocation.get('product_name', 'Unknown Product')
                total_quantity = allocation.get('total_quantity', 0)
                allocation_data = allocation.get('allocation', [])
                
                formatted = f"""
## 📦 **Allocation Results**

### 🎯 **Allocation Summary**
- **Product**: {product_name}
- **Total Quantity**: {total_quantity:,} units
- **Cities**: {len(allocation_data)} locations

### 🏙️ **City-wise Distribution**
"""
                for item in allocation_data:
                    city = item.get('city_name', '').title()
                    allocated = item.get('recommended_allocation', 0)
                    current_stock = item.get('stock_quantity', 0)
                    status = "🚨 URGENT" if current_stock < 10 else "⚠️ LOW" if current_stock < 50 else "✅ OK"
                    formatted += f"- **{city}**: {allocated:,} units (Current: {current_stock:,}) {status}\n"
                
                formatted += f"""
### 🧠 **AI Insights**
{insight}

---
*Optimized using advanced allocation algorithms*
"""
                return formatted
            
            # Fallback for other responses
            else:
                return response
                
        except (json.JSONDecodeError, KeyError, TypeError):
            # If parsing fails, return original response
            return response

def main():
    st.set_page_config(
        page_title="🤖 Quick Commerce AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for ChatGPT-style interface
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #4a90e2 0%, #6b73ff 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #4a90e2 0%, #6b73ff 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem 3rem;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.2);
        position: relative;
    }
    
    .ai-message {
        background: #f8f9fa;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 3rem 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #4a90e2;
        position: relative;
    }
    
    .message-label {
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    .example-button {
        background: linear-gradient(135deg, #4a90e2 0%, #6b73ff 100%);
        color: white;
        border: none;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.2);
        font-weight: 500;
        text-align: left;
        width: 100%;
        font-size: 1rem;
        line-height: 1.4;
    }
    
    .example-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        background: linear-gradient(135deg, #3d7bc6 0%, #5a5fcc 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #6b73ff 100%);
        color: white;
        border: none;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.2);
        font-size: 1rem;
        line-height: 1.4;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        background: linear-gradient(135deg, #3d7bc6 0%, #5a5fcc 100%);
    }
    
    
    .examples-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .examples-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Quick Commerce AI Operations Manager</h1>
        <p style="font-size: 1.2rem; margin: 0;">Chat with your AI operations manager using natural language!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = QuickCommerceChatbot()
        st.session_state.chat_history = []
    
    chatbot = st.session_state.chatbot
    
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Data Upload")
        sales_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'], key='sales_upload')
        inv_file = st.file_uploader("Upload Inventory Data (CSV)", type=['csv'], key='inv_upload')
        
        # --- JSON Input to Cohere AI ---
        if hasattr(st.session_state, 'last_response') and st.session_state.last_response:
            try:
                response_data = st.session_state.last_response
                if 'forecast' in response_data:
                    st.markdown("---")
                    st.subheader('🔍 JSON Input to Cohere AI')
                    st.json(response_data['forecast'])
                elif 'comparison' in response_data:
                    st.markdown("---")
                    st.subheader('🔍 JSON Input to Cohere AI')
                    st.json(response_data['comparison'])
                elif 'allocation' in response_data:
                    st.markdown("---")
                    st.subheader('🔍 JSON Input to Cohere AI')
                    st.json(response_data['allocation'])
            except Exception as e:
                st.error(f"Error displaying JSON: {e}")
                pass
        
        # --- LSTM Model Metrics (Hidden by default) ---
        if hasattr(chatbot, 'lstm_metrics') and chatbot.lstm_metrics:
            with st.expander("📊 LSTM Model Performance (Click to view)"):
                for metric, value in chatbot.lstm_metrics.items():
                    st.metric(metric, f"{value:.4f}")
        
        # --- Cohere AI Insight ---
        if hasattr(st.session_state, 'last_response') and st.session_state.last_response:
            try:
                response_data = st.session_state.last_response
                if 'insight' in response_data and response_data['insight']:
                    st.markdown("---")
                    st.subheader('🤖 AI Insight (Cohere)')
                    st.info(response_data['insight'])
            except Exception as e:
                st.error(f"Error displaying insight: {e}")
                pass
        
        if sales_file and inv_file:
            if st.button("🔄 Load Data", use_container_width=True):
                with st.spinner("Loading and processing data..."):
                    success = chatbot.load_data(sales_file, inv_file)
                    if success:
                        st.success("✅ Data loaded successfully!")
                        st.session_state.data_loaded = True
                    else:
                        st.error("❌ Failed to load data")
                        st.session_state.data_loaded = False
        else:
            st.info("📤 Please upload both sales and inventory CSV files to start")
            st.session_state.data_loaded = False

        # --- LSTM Model Status Section ---
        if st.session_state.get('data_loaded', False):
            st.markdown('---')
            st.header("🧠 LSTM Model Status")
            if hasattr(chatbot, 'lstm_metrics') and chatbot.lstm_metrics:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAE", f"{chatbot.lstm_metrics.get('MAE', 0):.2f}")
                    st.metric("RMSE", f"{chatbot.lstm_metrics.get('RMSE', 0):.2f}")
                with col2:
                    st.metric("MAPE", f"{chatbot.lstm_metrics.get('MAPE', 0):.2f}%")
                    st.metric("SMAPE", f"{chatbot.lstm_metrics.get('SMAPE', 0):.2f}%")
            else:
                st.info("No LSTM model metrics available. Train the model first using the CLI command shown above.")
    
    # Main chat interface
    if st.session_state.get('data_loaded', False):
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat history with ChatGPT-style bubbles
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                # Escape HTML content for user messages
                import html
                safe_content = html.escape(message['content'])
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-label">👤 You</div>
                    {safe_content}
                </div>
                """, unsafe_allow_html=True)
            else:
                # For AI messages, we need to handle markdown properly
                st.markdown(f"""
                <div class="ai-message">
                    <div class="message-label">🤖 AI Operations Manager</div>
                </div>
                """, unsafe_allow_html=True)
                # Display the AI response content as markdown (not HTML)
                st.markdown(message['content'])
                
                # Display charts for urgent stock queries
                if (i > 0 and 
                    st.session_state.chat_history[i-1]['type'] == 'user' and 
                    "urgent stock" in st.session_state.chat_history[i-1]['content'].lower() and 
                    "all products" in st.session_state.chat_history[i-1]['content'].lower() and
                    'urgent_charts' in st.session_state):
                    
                    st.markdown("---")
                    st.subheader("📊 **Visual Analysis**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(st.session_state.urgent_charts[0], use_container_width=True)
                    with col2:
                        st.plotly_chart(st.session_state.urgent_charts[1], use_container_width=True)
                    
                    # Clear charts after display
                    del st.session_state.urgent_charts
                
                # Display charts for allocation queries
                elif (i > 0 and 
                      st.session_state.chat_history[i-1]['type'] == 'user' and 
                      any(word in st.session_state.chat_history[i-1]['content'].lower() for word in ['allocate', 'distribute', 'allocation']) and
                      'allocation_charts' in st.session_state):
                    
                    st.markdown("---")
                    st.subheader("📊 **Allocation Analysis**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(st.session_state.allocation_charts[0], use_container_width=True)
                    with col2:
                        st.plotly_chart(st.session_state.allocation_charts[1], use_container_width=True)
                    
                    # Show detailed table
                    st.markdown("---")
                    st.subheader("📋 **Detailed City-wise Breakdown**")
                    
                    allocation_df = st.session_state.allocation_charts[2]
                    # Format the dataframe for better display
                    display_df = allocation_df.copy()
                    display_df['Current Stock'] = display_df['stock_quantity'].astype(int)
                    display_df['Allocated'] = display_df['recommended_allocation'].astype(int)
                    display_df['Status'] = display_df.apply(lambda row: '🚨 URGENT' if row['stock_quantity'] < 10 else '⚠️ LOW' if row['stock_quantity'] < 50 else '✅ OK', axis=1)
                    
                    # Select and rename columns for display
                    display_df = display_df[['city_name', 'Current Stock', 'Allocated', 'Status']].rename(columns={'city_name': 'City'})
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "City": st.column_config.TextColumn("🏙️ City", width="medium"),
                            "Current Stock": st.column_config.NumberColumn("📦 Current Stock", width="small"),
                            "Allocated": st.column_config.NumberColumn("🎯 Allocated", width="small"),
                            "Status": st.column_config.TextColumn("📊 Status", width="small")
                        }
                    )
                    
                    # Clear charts after display
                    del st.session_state.allocation_charts
                
                # Display charts for forecast queries
                if 'forecast_chart' in st.session_state:
                    st.markdown("---")
                    st.subheader("📊 **Sales Forecast Visualization**")
                    st.plotly_chart(st.session_state.forecast_chart, use_container_width=True)
                    del st.session_state.forecast_chart
                
                # Display charts for allocation queries
                elif 'allocation_charts' in st.session_state:
                    st.markdown("---")
                    st.subheader("📊 **Allocation Analysis**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(st.session_state.allocation_charts[0], use_container_width=True)
                    with col2:
                        st.plotly_chart(st.session_state.allocation_charts[1], use_container_width=True)
                    
                    # Show detailed table
                    st.markdown("---")
                    st.subheader("📋 **Detailed City-wise Breakdown**")
                    
                    allocation_df = st.session_state.allocation_charts[2]
                    # Format the dataframe for better display
                    display_df = allocation_df.copy()
                    display_df['Current Stock'] = display_df['stock_quantity'].astype(int)
                    display_df['Allocated'] = display_df['recommended_allocation'].astype(int)
                    display_df['Status'] = display_df.apply(lambda row: '🚨 URGENT' if row['stock_quantity'] < 10 else '⚠️ LOW' if row['stock_quantity'] < 50 else '✅ OK', axis=1)
                    
                    # Select and rename columns for display
                    display_df = display_df[['city_name', 'Current Stock', 'Allocated', 'Status']].rename(columns={'city_name': 'City'})
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "City": st.column_config.TextColumn("🏙️ City", width="medium"),
                            "Current Stock": st.column_config.NumberColumn("📦 Current Stock", width="small"),
                            "Allocated": st.column_config.NumberColumn("🎯 Allocated", width="small"),
                            "Status": st.column_config.TextColumn("📊 Status", width="small")
                        }
                    )
                    
                    del st.session_state.allocation_charts
                
                # Display charts for comparison queries
                elif 'comparison_chart' in st.session_state:
                    st.markdown("---")
                    st.subheader("📊 **City Comparison Visualization**")
                    st.plotly_chart(st.session_state.comparison_chart, use_container_width=True)
                    del st.session_state.comparison_chart
                
                # Display charts for analysis queries
                elif (i > 0 and 
                      st.session_state.chat_history[i-1]['type'] == 'user' and 
                      any(word in st.session_state.chat_history[i-1]['content'].lower() for word in ['analyze', 'analysis', 'performance', 'sales']) and
                      'analysis_charts' in st.session_state):
                    
                    st.markdown("---")
                    st.subheader("📊 **City Performance Analysis**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(st.session_state.analysis_charts[0], use_container_width=True)
                    with col2:
                        st.plotly_chart(st.session_state.analysis_charts[1], use_container_width=True)
                    
                    # Show detailed table
                    st.markdown("---")
                    st.subheader("📋 **Detailed Product Breakdown**")
                    
                    analysis_df = st.session_state.analysis_charts[2]
                    # Format the dataframe for better display
                    display_df = analysis_df.copy()
                    display_df['Product Name'] = display_df['product_id'].apply(lambda x: chatbot.get_product_name(x)[:50] + '...' if len(chatbot.get_product_name(x)) > 50 else chatbot.get_product_name(x))
                    display_df['Sales'] = display_df['units_sold'].astype(int)
                    display_df['Stock'] = display_df['stock_quantity'].astype(int)
                    display_df['7-Day Avg'] = display_df['avg_7'].round(1)
                    display_df['Status'] = display_df.apply(lambda row: '🚨 URGENT' if row['stock_quantity'] < 10 else '⚠️ LOW' if row['stock_quantity'] < 50 else '✅ OK', axis=1)
                    
                    # Select and rename columns for display
                    display_df = display_df[['Product Name', 'Sales', 'Stock', '7-Day Avg', 'Status']]
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Product Name": st.column_config.TextColumn("📦 Product", width="large"),
                            "Sales": st.column_config.NumberColumn("📈 Sales", width="small"),
                            "Stock": st.column_config.NumberColumn("📦 Stock", width="small"),
                            "7-Day Avg": st.column_config.NumberColumn("📊 7-Day Avg", width="small"),
                            "Status": st.column_config.TextColumn("📊 Status", width="small")
                        }
                    )
                    
                    # Clear charts after display
                    del st.session_state.analysis_charts
        
        # Clean input section
        user_input = st.text_input(
            "💬 Ask me anything about your operations:",
            placeholder="e.g., 'Allocate 1000 units of Gentle Baby Wash' or 'Show me urgent stock alerts'",
            key="chat_input",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("🚀 Send", use_container_width=True, key="send_button")
        
        # Process input when Enter is pressed or Send button is clicked
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input
            })
            
            # Process query
            with st.spinner("🤔 Thinking..."):
                response = chatbot.process_query(user_input)
                
                # Parse response and create visualizations
                try:
                    response_data = json.loads(response)
                    
                    # Store response data for display
                    st.session_state.last_response = response_data
                    
                    # Create charts based on response type
                    print(f"🔍 Response data keys: {list(response_data.keys())}")
                    
                    if 'forecast' in response_data:
                        print("📊 Creating forecast chart...")
                        forecast_data = response_data['forecast']
                        if 'error' not in forecast_data:
                            # Extract product and city from the forecast data
                            product_id = forecast_data.get('product_id')
                            city_name = forecast_data.get('city_name')
                            chart = chatbot.create_forecast_chart(forecast_data, product_id, city_name)
                            if chart:
                                st.session_state.forecast_chart = chart
                                print("✅ Forecast chart created and stored in session state")
                            else:
                                print("❌ Failed to create forecast chart")
                        else:
                            print(f"❌ Forecast error: {forecast_data['error']}")
                    
                    elif 'comparison' in response_data:
                        print("📊 Creating comparison chart...")
                        comparison_data = response_data['comparison']
                        if 'error' not in comparison_data:
                            chart = chatbot.create_comparison_chart(comparison_data)
                            if chart:
                                st.session_state.comparison_chart = chart
                                print("✅ Comparison chart created and stored in session state")
                            else:
                                print("❌ Failed to create comparison chart")
                        else:
                            print(f"❌ Comparison error: {comparison_data['error']}")
                    
                    elif 'allocation' in response_data:
                        print("📊 Creating allocation charts...")
                        allocation_data = response_data['allocation']
                        if 'error' not in allocation_data:
                            charts = chatbot.create_allocation_visualization(allocation_data)
                            if charts:
                                st.session_state.allocation_charts = charts
                                print("✅ Allocation charts created and stored in session state")
                            else:
                                print("❌ Failed to create allocation charts")
                        else:
                            print(f"❌ Allocation error: {allocation_data['error']}")
                                
                except json.JSONDecodeError:
                    # Handle non-JSON responses
                    pass
            
            # Format response for better user experience
            formatted_response = chatbot.format_response_for_user(response)
            
            # Add AI response to history
            st.session_state.chat_history.append({
                'type': 'ai',
                'content': formatted_response
            })
            
            st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Modern Example Queries Section
        st.markdown("""
        <div class="examples-section">
            <div class="examples-title">💡 What would you like to know?</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Real example questions that show actual capabilities
        examples = [
            "Compare sales between Delhi and Mumbai",
            "Predict sales for Product 445017 in Chennai for 7 days",
            "Forecast demand from 2025-01-01 to 2025-01-31",
            "Allocate 1000 units for Gentle Baby Wash"
        ]
        
        # Create 2x2 grid layout
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💬 Compare sales between Delhi and Mumbai", key="example_0", use_container_width=True):
                example = "Compare sales between Delhi and Mumbai"
                # Process the example query directly
                with st.spinner("🤔 Thinking..."):
                    response = chatbot.process_query(example)
                    
                    # Check if this is an urgent stock query and create visualizations
                    if "urgent stock" in example.lower() and "all products" in example.lower():
                        # Get urgent products data for visualization
                        urgent_products = []
                        for product_id in chatbot.latest['product_id'].unique():
                            product_latest = chatbot.latest[chatbot.latest['product_id'] == product_id]
                            urgent_cities = product_latest[product_latest.apply(lambda r: check_urgent(r['stock_quantity'], r['avg_3']), axis=1)]
                            if not urgent_cities.empty:
                                product_info = chatbot.get_product_info(product_id)
                                urgent_products.append({
                                    'product_id': product_id,
                                    'product_name': product_info['product_name'],
                                    'urgent_cities': urgent_cities,
                                    'total_urgent_qty': urgent_cities['avg_7'].sum() * 7
                                })
                        
                        if urgent_products:
                            urgent_products.sort(key=lambda x: x['total_urgent_qty'], reverse=True)
                            fig1, fig2 = chatbot.create_urgent_stock_visualization(urgent_products)
                            
                            # Store charts in session state for display
                            st.session_state.urgent_charts = (fig1, fig2)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': example
                })
                st.session_state.chat_history.append({
                    'type': 'ai',
                    'content': response
                })
                
                st.rerun()
            
            if st.button("💬 Predict sales for Product 445017 in Chennai for 7 days", key="example_1", use_container_width=True):
                example = "Predict sales for Product 445017 in Chennai for 7 days"
                # Process the example query directly
                with st.spinner("🤔 Thinking..."):
                    response = chatbot.process_query(example)
                    
                    # Check if this is an allocation query and create visualizations
                    if any(word in example.lower() for word in ['allocate', 'distribute', 'allocation']):
                        # Allocation charts are already created in the process_allocation_query method
                        pass
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': example
                })
                st.session_state.chat_history.append({
                    'type': 'ai',
                    'content': response
                })
                
                st.rerun()
        
        with col2:
            if st.button("💬 Forecast demand from 2025-01-01 to 2025-01-31", key="example_2", use_container_width=True):
                example = "Forecast demand from 2025-01-01 to 2025-01-31"
                # Process the example query directly
                with st.spinner("🤔 Thinking..."):
                    response = chatbot.process_query(example)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': example
                })
                st.session_state.chat_history.append({
                    'type': 'ai',
                    'content': response
                })
                
                st.rerun()
            
            if st.button("💬 Allocate 1000 units for Gentle Baby Wash", key="example_3", use_container_width=True):
                example = "Allocate 1000 units for Gentle Baby Wash"
                # Process the example query directly
                with st.spinner("🤔 Thinking..."):
                    response = chatbot.process_query(example)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': example
                })
                st.session_state.chat_history.append({
                    'type': 'ai',
                    'content': response
                })
                
                st.rerun()
        
        # Close chat container
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # No data loaded - show modern welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2 style="color: #4a90e2; margin-bottom: 1rem;">👆 Please upload your data files to start chatting!</h2>
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">Upload your sales and inventory CSV files in the sidebar to begin</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show modern example queries even without data
        st.markdown("""
        <div class="examples-section">
            <div class="examples-title">💡 What you can ask once data is loaded:</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Modern button layout for examples
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <h4 style="color: #4a90e2; margin-bottom: 1rem;">📦 Smart Allocation</h4>
                <p style="color: #666; margin: 0.5rem 0;">• "Allocate 1000 units of Gentle Baby Wash"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "Distribute 500 units across cities"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "Show allocation plan for Product 445017"</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <h4 style="color: #4a90e2; margin-bottom: 1rem;">🔮 AI Forecasting</h4>
                <p style="color: #666; margin: 0.5rem 0;">• "Generate 7-day forecast for all products"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "Show sales forecast for Caring Baby Wipes"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "Predict demand in Mumbai"</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <h4 style="color: #4a90e2; margin-bottom: 1rem;">🚨 Urgent Stock Alerts</h4>
                <p style="color: #666; margin: 0.5rem 0;">• "Check urgent stock for all products"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "What products need restocking?"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "Show low stock alerts"</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <h4 style="color: #4a90e2; margin-bottom: 1rem;">📧 Auto Email System</h4>
                <p style="color: #666; margin: 0.5rem 0;">• "Send PO email for Caring Baby Wipes"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "Generate purchase order for Product 445017"</p>
                <p style="color: #666; margin: 0.5rem 0;">• "Notify supplier about urgent stock"</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
