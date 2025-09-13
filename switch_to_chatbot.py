#!/usr/bin/env python3
"""
Switch to Chatbot Interface
Converts your existing system to use natural language queries like your friend's system
"""

import os
import shutil

def switch_to_chatbot():
    """Switch from dropdown interface to chatbot interface"""
    
    print("🤖 Switching to Chatbot Interface...")
    
    # Check if chatbot app exists
    if not os.path.exists("src/chatbot_app.py"):
        print("❌ Chatbot app not found!")
        return False
    
    # Backup original app
    if os.path.exists("src/app.py"):
        shutil.copy("src/app.py", "src/app_original.py")
        print("📁 Backed up original app to app_original.py")
    
    # Copy chatbot app as main app
    shutil.copy("src/chatbot_app.py", "src/app.py")
    print("✅ Switched to chatbot interface!")
    
    # Create usage guide
    create_usage_guide()
    
    print("\n🎉 **Chatbot Interface Ready!**")
    print("\n📋 **How to use:**")
    print("1. Run: streamlit run src/app.py")
    print("2. Upload your CSV files in the sidebar")
    print("3. Start chatting with natural language!")
    print("\n💬 **Example queries:**")
    print("• 'Allocate 1000 units of Gentle Baby Wash'")
    print("• 'Show me sales forecast for Product 445017'")
    print("• 'Check urgent stock for all products'")
    print("• 'Send PO email for Gentle Baby Wash'")
    print("• 'Analyze sales performance for Mumbai'")
    
    return True

def create_usage_guide():
    """Create a usage guide for the chatbot interface"""
    
    guide_content = """# 🤖 Chatbot Interface Usage Guide

## 🚀 Getting Started

1. **Run the app:**
   ```bash
   streamlit run src/app.py
   ```

2. **Upload your data:**
   - Upload sales CSV file in the sidebar
   - Upload inventory CSV file in the sidebar
   - Click "Load Data" button

3. **Start chatting:**
   - Type your questions in natural language
   - Click "Send" or press Enter
   - Get instant AI-powered responses

## 💬 Example Queries

### 📦 Allocation & Distribution
- "Allocate 1000 units of Gentle Baby Wash"
- "How should I distribute 500 units of Product 445017?"
- "Show allocation plan for Gentle Baby Wash across cities"

### 🔮 Forecasting & Predictions
- "Show me sales forecast for Gentle Baby Wash"
- "Predict demand for Product 445017 in Mumbai"
- "Generate 7-day forecast for all products"

### 🚨 Urgent Stock Management
- "Check urgent stock for all products"
- "What products need immediate restocking?"
- "Show me low stock alerts"

### 📧 Email & Notifications
- "Send PO email for Gentle Baby Wash"
- "Notify supplier about urgent stock"
- "Generate purchase order for Product 445017"

### 📊 Analysis & Insights
- "Analyze sales performance for Mumbai"
- "Show me top performing products"
- "What are the underperforming cities?"

## 🎯 Key Features

✅ **Natural Language Processing** - Ask questions like you're talking to a person
✅ **Auto-send PO Emails** - Automatically sends professional Purchase Orders
✅ **Smart Allocation** - Intelligent distribution based on city performance
✅ **Real-time Forecasting** - 7-day sales predictions using Prophet + LightGBM
✅ **Urgent Stock Alerts** - Instant notifications for low stock situations
✅ **Professional Email Templates** - Business-ready PO emails with all details

## 🔧 Technical Details

- **Backend:** Python + Streamlit
- **AI Models:** LightGBM + Prophet for forecasting
- **Email System:** SMTP with multiple fallbacks
- **Data Processing:** Pandas with advanced feature engineering
- **Natural Language:** Custom query parsing and intent recognition

## 🆘 Troubleshooting

**Q: The chatbot doesn't understand my query**
A: Try using the example queries as templates. Be specific about product names and quantities.

**Q: Email sending fails**
A: Check the console for detailed error messages. The system will save HTML files as fallback.

**Q: No data found for my product**
A: Make sure you've uploaded the correct CSV files and the product names match your data.

## 🎉 Enjoy Your AI Operations Manager!

Your chatbot can now handle complex operations queries just like ChatGPT, but specifically designed for your Quick Commerce business!
"""
    
    with open("CHATBOT_USAGE_GUIDE.md", "w", encoding='utf-8') as f:
        f.write(guide_content)
    
    print("📖 Created CHATBOT_USAGE_GUIDE.md")

if __name__ == "__main__":
    success = switch_to_chatbot()
    if success:
        print("\n🚀 Ready to chat with your AI operations manager!")
    else:
        print("\n❌ Switch failed. Please check the error messages above.")
