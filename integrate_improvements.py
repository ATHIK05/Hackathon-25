#!/usr/bin/env python3
"""
Integration script to enhance the existing EQ-Rev_AppXcel system
with our improved features:
- Enhanced email system with auto-send
- Better error handling and fallbacks
- Multiple SMTP configurations
- Professional PO templates
"""

import os
import shutil
import subprocess
import sys

def clone_and_enhance():
    """Clone the repository and integrate our improvements"""
    
    print("🚀 Integrating improvements with EQ-Rev_AppXcel...")
    
    # Clone the repository
    repo_url = "https://github.com/SanmathiSedhupathi/EQ-Rev_AppXcel.git"
    clone_dir = "EQ-Rev_AppXcel"
    
    try:
        # Clone if not exists
        if not os.path.exists(clone_dir):
            print(f"📥 Cloning repository from {repo_url}")
            subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
        else:
            print(f"📁 Repository already exists at {clone_dir}")
        
        # Copy our enhanced email system
        print("📧 Integrating enhanced email system...")
        shutil.copy("src/email_system.py", f"{clone_dir}/enhanced_email_system.py")
        
        # Copy our improved data processing
        print("🔄 Integrating improved data processing...")
        shutil.copy("src/data_processing.py", f"{clone_dir}/enhanced_data_processing.py")
        
        # Copy our model improvements
        print("🤖 Integrating enhanced model...")
        shutil.copy("src/model.py", f"{clone_dir}/enhanced_model.py")
        
        # Create integration guide
        create_integration_guide(clone_dir)
        
        print("✅ Integration complete!")
        print(f"📁 Enhanced files available in: {clone_dir}/")
        
    except Exception as e:
        print(f"❌ Integration failed: {str(e)}")
        return False
    
    return True

def create_integration_guide(clone_dir):
    """Create a guide for integrating the improvements"""
    
    guide_content = """# 🚀 Integration Guide: Enhanced Features

## 📧 Enhanced Email System

### Features Added:
- **Auto-send PO emails** when urgent stock detected
- **Multiple SMTP fallbacks** (ports 587, 465, 25)
- **Professional HTML templates** with business formatting
- **Error handling** with detailed logging
- **File fallback** if SMTP fails

### Integration Steps:

1. **Replace existing email system:**
   ```python
   # In your action_execution.py or main app
   from enhanced_email_system import PurchaseOrderEmailSystem
   
   # Initialize with your credentials
   email_system = PurchaseOrderEmailSystem()
   ```

2. **Update email configuration:**
   ```python
   # The system now uses:
   # - Sender: mohamedathikr.22msc@kongu.edu
   # - App Password: iopc hfuw ryic jypx
   # - Multiple SMTP servers for reliability
   ```

3. **Auto-send functionality:**
   ```python
   # Automatically sends PO emails when urgent stock detected
   if urgent_stock_detected:
       po_data = email_system.create_po_data(product_info, quantity, 'HIGH')
       success, message = email_system.send_po_email(po_data, recipient)
   ```

## 🔄 Enhanced Data Processing

### Improvements:
- **Better categorical handling** for product IDs
- **Realistic business patterns** in forecasting
- **Improved data augmentation** for sparse data
- **Consistent data types** across all operations

### Integration:
```python
# Replace your data_preprocessing.py imports with:
from enhanced_data_processing import load_sales_inventory, complete_date_range
```

## 🤖 Enhanced Model

### Features:
- **LightGBM + Prophet** hybrid approach
- **Realistic business patterns** (weekend effects, midweek boosts)
- **Better handling of sparse data**
- **Improved forecasting accuracy**

### Integration:
```python
# Use enhanced model for better predictions:
from enhanced_model import train_lgbm, prophet_forecast
```

## 🎯 Key Enhancements Summary:

1. **Auto-Send Emails**: PO emails sent automatically when urgent stock detected
2. **Multiple SMTP Fallbacks**: Reliable email delivery with multiple server attempts
3. **Professional Templates**: Business-ready HTML email templates
4. **Better Error Handling**: Detailed error messages and fallback options
5. **Enhanced Forecasting**: More realistic predictions with business patterns
6. **Improved Data Processing**: Better handling of categorical data and sparse datasets

## 🚀 Next Steps:

1. **Test the enhanced email system** with your existing data
2. **Integrate auto-send** into your Streamlit app
3. **Update your LSTM model** with the enhanced preprocessing
4. **Deploy with improved reliability**

## 📞 Support:

- Check console logs for detailed error information
- Email fallback files are saved if SMTP fails
- Multiple SMTP attempts ensure delivery reliability

---
**Enhanced by the Hackathon Team** 🏆
"""
    
    with open(f"{clone_dir}/INTEGRATION_GUIDE.md", "w", encoding='utf-8') as f:
        f.write(guide_content)
    
    print("📖 Integration guide created: INTEGRATION_GUIDE.md")

if __name__ == "__main__":
    success = clone_and_enhance()
    if success:
        print("\n🎉 Ready to enhance your teammate's amazing system!")
        print("📁 Check the EQ-Rev_AppXcel folder for integrated files")
    else:
        print("\n❌ Integration failed. Please check the error messages above.")
