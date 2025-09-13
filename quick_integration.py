#!/usr/bin/env python3
"""
Quick Integration Script for EQ-Rev_AppXcel
Integrates our enhanced email system with your teammate's amazing foundation
"""

import os
import sys

def integrate_enhanced_email():
    """Integrate our enhanced email system"""
    
    print("🚀 Integrating Enhanced Email System with EQ-Rev_AppXcel...")
    
    # Check if we're in the right directory
    if not os.path.exists("EQ-Rev_AppXcel"):
        print("❌ EQ-Rev_AppXcel directory not found!")
        print("💡 Make sure you're in the Hackathon directory")
        return False
    
    # Copy enhanced files
    enhanced_files = [
        ("src/email_system.py", "EQ-Rev_AppXcel/enhanced_email_system.py"),
        ("src/data_processing.py", "EQ-Rev_AppXcel/enhanced_data_processing.py"),
        ("src/model.py", "EQ-Rev_AppXcel/enhanced_model.py")
    ]
    
    for src, dst in enhanced_files:
        if os.path.exists(src):
            print(f"📧 Copying {src} to {dst}")
            with open(src, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(dst, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            print(f"⚠️ Source file {src} not found")
    
    # Create integration example
    create_integration_example()
    
    print("✅ Integration complete!")
    print("📁 Enhanced files are now in EQ-Rev_AppXcel/")
    print("📖 Check INTEGRATION_GUIDE.md for detailed instructions")
    
    return True

def create_integration_example():
    """Create a simple integration example"""
    
    example_code = '''# Enhanced Email Integration Example
# Add this to your action_execution.py or streamlit_app.py

from enhanced_email_system import PurchaseOrderEmailSystem

class EnhancedActionExecutor:
    def __init__(self):
        self.email_system = PurchaseOrderEmailSystem()
    
    def send_urgent_po_auto(self, product_data, urgent_cities):
        """Auto-send PO email when urgent stock detected"""
        if not urgent_cities:
            return False, "No urgent stock detected"
        
        # Calculate urgent quantity
        total_urgent_qty = sum(city['avg_7'] * 7 for city in urgent_cities)
        
        # Create PO data
        po_data = self.email_system.create_po_data(
            product_info=product_data,
            quantity=int(total_urgent_qty),
            priority='HIGH'
        )
        
        # Send email automatically
        success, message = self.email_system.send_po_email(
            po_data, 
            "rathikmohamed786@gmail.com"
        )
        
        return success, message

# Usage in your Streamlit app:
def check_and_auto_send():
    executor = EnhancedActionExecutor()
    
    # Your existing logic to detect urgent stock
    urgent_cities = detect_urgent_stock()  # Your existing function
    
    if urgent_cities:
        success, message = executor.send_urgent_po_auto(
            product_data, 
            urgent_cities
        )
        
        if success:
            st.success(f"🚀 Auto-sent PO email: {message}")
        else:
            st.warning(f"⚠️ Auto-send failed: {message}")
'''
    
    with open("EQ-Rev_AppXcel/email_integration_example.py", "w", encoding='utf-8') as f:
        f.write(example_code)
    
    print("📝 Created email_integration_example.py")

if __name__ == "__main__":
    success = integrate_enhanced_email()
    if success:
        print("\n🎉 Ready to enhance your teammate's system!")
        print("\n📋 Next steps:")
        print("1. cd EQ-Rev_AppXcel")
        print("2. pip install lightgbm prophet imblearn")
        print("3. Check email_integration_example.py")
        print("4. Integrate with your existing code")
        print("5. streamlit run streamlit_app.py")
    else:
        print("\n❌ Integration failed. Please check the error messages above.")
