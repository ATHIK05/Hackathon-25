import smtplib
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

class PurchaseOrderEmailSystem:
    def __init__(self):
        # Try multiple SMTP configurations
        self.smtp_configs = [
            {"server": "smtp.gmail.com", "port": 587},
            {"server": "smtp.gmail.com", "port": 465},
            {"server": "smtp.gmail.com", "port": 25}
        ]
        self.sender_email = "mohamedathikr.22msc@kongu.edu"
        # App password for mohamedathikr.22msc@kongu.edu
        self.sender_password = "iopc hfuw ryic jypx"
        
    def generate_po_template(self, po_data: Dict) -> str:
        """Generate professional HTML email template for Purchase Order"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Purchase Order - {po_data['po_number']}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #2c3e50, #3498db);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: bold;
                }}
                .company-info {{
                    margin-top: 15px;
                }}
                .company-logo {{
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .company-name {{
                    font-size: 18px;
                    margin-bottom: 3px;
                }}
                .company-address {{
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .po-details {{
                    background-color: #ecf0f1;
                    padding: 25px;
                    border-left: 4px solid #3498db;
                }}
                .details-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin-top: 20px;
                }}
                .supplier-info, .po-info {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 6px;
                    border: 1px solid #ddd;
                }}
                .section-title {{
                    background-color: #34495e;
                    color: white;
                    padding: 10px 15px;
                    margin: -20px -20px 15px -20px;
                    font-weight: bold;
                    border-radius: 6px 6px 0 0;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 8px;
                    padding: 5px 0;
                    border-bottom: 1px solid #eee;
                }}
                .info-label {{
                    font-weight: bold;
                    color: #555;
                }}
                .info-value {{
                    color: #333;
                }}
                .bill-ship-section {{
                    background-color: #f8f9fa;
                    padding: 25px;
                }}
                .bill-ship-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                }}
                .bill-to, .ship-to {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 6px;
                    border: 1px solid #ddd;
                }}
                .section-header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 12px 15px;
                    margin: -20px -20px 15px -20px;
                    font-weight: bold;
                    border-radius: 6px 6px 0 0;
                }}
                .payment-terms {{
                    background-color: #e8f5e8;
                    padding: 20px;
                    border-left: 4px solid #27ae60;
                    margin: 20px 0;
                }}
                .payment-terms h3 {{
                    margin-top: 0;
                    color: #27ae60;
                }}
                .items-table {{
                    margin: 25px 0;
                    overflow-x: auto;
                }}
                .items-table table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    border-radius: 6px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .items-table th {{
                    background-color: #3498db;
                    color: white;
                    padding: 15px 10px;
                    text-align: left;
                    font-weight: bold;
                }}
                .items-table td {{
                    padding: 12px 10px;
                    border-bottom: 1px solid #eee;
                }}
                .items-table tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .items-table tr:hover {{
                    background-color: #e3f2fd;
                }}
                .totals-section {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    text-align: right;
                }}
                .total-row {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    padding: 8px 0;
                }}
                .total-label {{
                    font-weight: bold;
                    color: #555;
                }}
                .total-value {{
                    font-weight: bold;
                    color: #333;
                }}
                .grand-total {{
                    border-top: 2px solid #3498db;
                    padding-top: 15px;
                    margin-top: 15px;
                    font-size: 18px;
                    color: #2c3e50;
                }}
                .terms-section {{
                    background-color: #fff3cd;
                    padding: 25px;
                    border-left: 4px solid #ffc107;
                    margin: 25px 0;
                }}
                .terms-section h3 {{
                    margin-top: 0;
                    color: #856404;
                }}
                .terms-list {{
                    margin: 15px 0;
                }}
                .terms-list li {{
                    margin-bottom: 8px;
                    line-height: 1.5;
                }}
                .footer {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .footer-signature {{
                    background-color: #34495e;
                    padding: 15px;
                    border-radius: 6px;
                    margin-top: 15px;
                    display: inline-block;
                }}
                .urgent-badge {{
                    background-color: #e74c3c;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                    display: inline-block;
                    margin-left: 10px;
                }}
                .priority-high {{
                    border-left: 4px solid #e74c3c;
                }}
                .priority-medium {{
                    border-left: 4px solid #f39c12;
                }}
                .priority-low {{
                    border-left: 4px solid #27ae60;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="header">
                    <h1>PURCHASE ORDER</h1>
                    <div class="company-info">
                        <div class="company-logo">🛒 Thendral</div>
                        <div class="company-name">Supermarket</div>
                        <div class="company-address">No 23/2, SBI Colony, Ragavendra Nagar, Chennai - 600124</div>
                    </div>
                </div>

                <!-- PO Details -->
                <div class="po-details">
                    <h2>Purchase Order Details</h2>
                    <div class="details-grid">
                        <div class="supplier-info">
                            <div class="section-title">Supplier Information</div>
                            <div class="info-row">
                                <span class="info-label">Supplier Name:</span>
                                <span class="info-value">{po_data['supplier_name']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Address:</span>
                                <span class="info-value">{po_data['supplier_address']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">GSTIN:</span>
                                <span class="info-value">{po_data['supplier_gstin']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Contact:</span>
                                <span class="info-value">{po_data['supplier_contact']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Email:</span>
                                <span class="info-value">{po_data['supplier_email']}</span>
                            </div>
                        </div>
                        
                        <div class="po-info">
                            <div class="section-title">Purchase Order Information</div>
                            <div class="info-row">
                                <span class="info-label">Supplier Code:</span>
                                <span class="info-value">{po_data['supplier_code']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">PO No:</span>
                                <span class="info-value">{po_data['po_number']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">PO Date:</span>
                                <span class="info-value">{po_data['po_date']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">PO Expiry Days:</span>
                                <span class="info-value">{po_data['expiry_days']} days</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Priority:</span>
                                <span class="info-value">{po_data['priority']} <span class="urgent-badge">URGENT</span></span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Bill To / Ship To -->
                <div class="bill-ship-section">
                    <div class="bill-ship-grid">
                        <div class="bill-to">
                            <div class="section-header">Bill To</div>
                            <div><strong>Thendral Supermarket</strong></div>
                            <div>No 23/2, SBI Colony.</div>
                            <div>Ragavendra Nagar, Chennai - 600124</div>
                            <div>GSTIN: 33APFSDF1ZV</div>
                            <div>Contact: +91-7869825463</div>
                            <div>Email: purchase-team@thendral.com</div>
                        </div>
                        
                        <div class="ship-to">
                            <div class="section-header">Ship To</div>
                            <div><strong>Thendral Supermarket</strong></div>
                            <div>No 23/2, SBI Colony.</div>
                            <div>Ragavendra Nagar, Chennai - 600124</div>
                            <div>GSTIN: 33APFSDF1ZV</div>
                            <div>Contact: +91-7869825463</div>
                            <div>Email: purchase-team@thendral.com</div>
                        </div>
                    </div>
                </div>

                <!-- Payment Terms -->
                <div class="payment-terms">
                    <h3>💳 Payment Terms</h3>
                    <p><strong>Payment Date:</strong> {po_data['expiry_days']} days from date of delivery</p>
                    <p><strong>Payment Terms:</strong> 100% against invoice</p>
                </div>

                <!-- Items Table -->
                <div class="items-table">
                    <table>
                        <thead>
                            <tr>
                                <th>S.No</th>
                                <th>Product Code</th>
                                <th>Product Name</th>
                                <th>HSN Code</th>
                                <th>Quantity</th>
                                <th>Units</th>
                                <th>Rate</th>
                                <th>Tax</th>
                                <th>Amount</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_items_rows(po_data['items'])}
                        </tbody>
                    </table>
                </div>

                <!-- Totals -->
                <div class="totals-section">
                    <div class="total-row">
                        <span class="total-label">Total:</span>
                        <span class="total-value">₹{po_data['total_amount']:,.2f}</span>
                    </div>
                    <div class="total-row">
                        <span class="total-label">Discounts:</span>
                        <span class="total-value">₹{po_data.get('discount', 0):,.2f}</span>
                    </div>
                    <div class="total-row grand-total">
                        <span class="total-label">Grand Total:</span>
                        <span class="total-value">₹{po_data['grand_total']:,.2f}</span>
                    </div>
                </div>

                <!-- Terms and Conditions -->
                <div class="terms-section">
                    <h3>📋 Terms and Conditions</h3>
                    <ol class="terms-list">
                        <li>We reserve the right to cancel the purchase order anytime before product shipment.</li>
                        <li>Invoice raised to us should contain the details of purchase order with date mentioned.</li>
                        <li>Adherence to agreed product specifications is a must. Any deviation during delivery will result in cancellation of PO.</li>
                        <li>Packing and shipping charges are to be borne by {po_data['supplier_name']}.</li>
                        <li>Delivery should be strictly done within {po_data['expiry_days']} days from the date of purchase order.</li>
                    </ol>
                    <p><strong>Mark any communications to purchase-team@thendral.com</strong></p>
                </div>

                <!-- Footer -->
                <div class="footer">
                    <div class="footer-signature">
                        <strong>For Thendral Supermarket</strong><br>
                        <em>Authorized signatory</em>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_items_rows(self, items: List[Dict]) -> str:
        """Generate HTML rows for items table"""
        rows = ""
        for i, item in enumerate(items, 1):
            rows += f"""
            <tr>
                <td>{i}</td>
                <td>{item['product_code']}</td>
                <td>{item['product_name']}</td>
                <td>{item['hsn_code']}</td>
                <td>{item['quantity']}</td>
                <td>{item['units']}</td>
                <td>₹{item['rate']:,.2f}</td>
                <td>{item['tax']}%</td>
                <td>₹{item['amount']:,.2f}</td>
            </tr>
            """
        return rows
    
    def create_po_data(self, product_info: Dict, quantity: int, priority: str = "HIGH") -> Dict:
        """Create PO data structure from product information"""
        
        # Generate PO number
        po_number = f"2024/PO-{datetime.now().strftime('%m%d%H%M')}"
        
        # Calculate amounts
        rate = product_info.get('mrp', 100)  # Use MRP as rate
        tax_rate = 18  # Standard GST rate
        amount = quantity * rate
        tax_amount = (amount * tax_rate) / 100
        total_amount = amount + tax_amount
        discount = total_amount * 0.01  # 1% discount
        grand_total = total_amount - discount
        
        po_data = {
            'po_number': po_number,
            'po_date': datetime.now().strftime('%d-%m-%Y'),
            'expiry_days': 7,
            'priority': priority,
            'supplier_name': 'SM traders',
            'supplier_address': '43, Kambar Street, Chennai - 600453',
            'supplier_gstin': '33AACCEPVS1ZH',
            'supplier_contact': '+91-9345678123',
            'supplier_email': 'purchase-sm@gmail.com',
            'supplier_code': 'VNDR-104',
            'items': [{
                'product_code': product_info['product_id'],
                'product_name': product_info['product_name'],
                'hsn_code': '34019011',  # Default HSN code
                'quantity': quantity,
                'units': 'units',  # Changed from 'nos' to 'units' for clarity
                'rate': rate,
                'tax': tax_rate,
                'amount': total_amount
            }],
            'total_amount': total_amount,
            'discount': discount,
            'grand_total': grand_total
        }
        
        return po_data
    
    def send_po_email(self, po_data: Dict, recipient_email: str) -> tuple[bool, str]:
        """Send Purchase Order email - returns (success, message)"""
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Purchase Order {po_data['po_number']} - {po_data['supplier_name']}"
        
        # Generate HTML content
        html_content = self.generate_po_template(po_data)
        
        # Create HTML part
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Try multiple SMTP configurations
        for i, config in enumerate(self.smtp_configs):
            try:
                print(f"Attempt {i+1}: Trying {config['server']}:{config['port']}")
                print(f"Sending from {self.sender_email} to {recipient_email}")
                
                if config['port'] == 465:
                    # Use SSL for port 465
                    import ssl
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(config['server'], config['port'], context=context) as server:
                        print("Connected to SMTP server (SSL)")
                        server.login(self.sender_email, self.sender_password)
                        print("Login successful")
                        server.send_message(msg)
                        print("Email sent successfully")
                else:
                    # Use TLS for other ports
                    with smtplib.SMTP(config['server'], config['port'], timeout=30) as server:
                        print("Connected to SMTP server")
                        server.starttls()
                        print("TLS started")
                        server.login(self.sender_email, self.sender_password)
                        print("Login successful")
                        server.send_message(msg)
                        print("Email sent successfully")
                
                return True, f"Email sent successfully via {config['server']}:{config['port']}"
                
            except smtplib.SMTPAuthenticationError as e:
                error_msg = f"Authentication failed on {config['server']}:{config['port']}: {str(e)}"
                print(error_msg)
                if i == len(self.smtp_configs) - 1:  # Last attempt
                    return False, f"Authentication failed on all servers. Check your email and app password."
                continue
                
            except smtplib.SMTPRecipientsRefused as e:
                error_msg = f"Recipient email refused on {config['server']}:{config['port']}: {str(e)}"
                print(error_msg)
                return False, error_msg
                
            except smtplib.SMTPServerDisconnected as e:
                error_msg = f"Server disconnected on {config['server']}:{config['port']}: {str(e)}"
                print(error_msg)
                if i == len(self.smtp_configs) - 1:  # Last attempt
                    return False, f"All SMTP servers disconnected. Check your network connection."
                continue
                
            except Exception as e:
                error_msg = f"Error on {config['server']}:{config['port']}: {str(e)}"
                print(error_msg)
                if i == len(self.smtp_configs) - 1:  # Last attempt
                    return False, f"All SMTP attempts failed. Last error: {str(e)}"
                continue
        
        # If all SMTP attempts fail, save email to file as fallback
        try:
            email_filename = f"po_email_{po_data['po_number'].replace('/', '_')}.html"
            with open(email_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return False, f"SMTP failed, but email saved to {email_filename}. You can send this manually."
        except Exception as e:
            return False, f"All SMTP attempts failed and file save failed: {str(e)}"
    
    def send_po_with_attachment(self, po_data: Dict, recipient_email: str, pdf_path: str = None) -> bool:
        """Send Purchase Order email with PDF attachment"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"Purchase Order {po_data['po_number']} - {po_data['supplier_name']}"
            
            # Generate HTML content
            html_content = self.generate_po_template(po_data)
            
            # Create HTML part
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add PDF attachment if provided
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(pdf_path)}'
                )
                msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Error sending email with attachment: {str(e)}")
            return False
