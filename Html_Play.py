import pandas as pd
import os
from datetime import datetime
import webbrowser

class NiftyOptionsAnalyzer:
    def __init__(self, lot_size=75):
        self.lot_size = lot_size
        self.strategies = []
        
    def read_csv(self, csv_file_path):
        """Read and process the CSV file"""
        try:
            # Read CSV with proper column names
            df = pd.read_csv(csv_file_path)
            
            # Clean column names (remove unnamed columns)
            df.columns = ['timestamp', 'symbol', 'price', 'action', 'reason', 'spot_price']
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
    
    def extract_strike_price(self, symbol):
        """Extract strike price from symbol"""
        import re
        match = re.search(r'(\d{5})(CE|PE)', symbol)
        return int(match.group(1)) if match else 0
    
    def analyze_trades(self, df):
        """Analyze trades and group them into strategies"""
        trades = df.to_dict('records')
        strategy_number = 1
        
        i = 0
        while i < len(trades):
            trade = trades[i]
            
            # Look for entry trades
            if 'Entry' in trade['reason']:
                # Find the next entry trade (should be the pair)
                if i + 1 < len(trades) and 'Entry' in trades[i + 1]['reason']:
                    entry_trade1 = trades[i]
                    entry_trade2 = trades[i + 1]
                    
                    # Find corresponding exit trades
                    exit_trade1 = None
                    exit_trade2 = None
                    
                    # Look for exit trades with matching symbols
                    for j in range(i + 2, len(trades)):
                        if 'Exit' in trades[j]['reason']:
                            if trades[j]['symbol'] == entry_trade1['symbol'] and not exit_trade1:
                                exit_trade1 = trades[j]
                            elif trades[j]['symbol'] == entry_trade2['symbol'] and not exit_trade2:
                                exit_trade2 = trades[j]
                            
                            if exit_trade1 and exit_trade2:
                                break
                    
                    # Determine strategy type
                    strategy_type = ""
                    if "Short Call Spread" in entry_trade1['reason'] or "Short Call Spread" in entry_trade2['reason']:
                        strategy_type = "Short Call Spread"
                    elif "Long Put Spread" in entry_trade1['reason'] or "Long Put Spread" in entry_trade2['reason']:
                        strategy_type = "Long Put Spread"
                    
                    # Calculate P&L if we have exit trades
                    if exit_trade1 and exit_trade2:
                        # Calculate net credit and debit
                        if entry_trade1['action'] == -1:  # Sell first
                            net_credit = entry_trade1['price'] - entry_trade2['price']
                            net_debit = exit_trade1['price'] - exit_trade2['price']
                        else:  # Buy first
                            net_credit = entry_trade2['price'] - entry_trade1['price']
                            net_debit = exit_trade2['price'] - exit_trade1['price']
                        
                        final_pnl = (net_credit - net_debit) * self.lot_size
                        
                        strategy = {
                            'number': strategy_number,
                            'type': strategy_type,
                            'entry_date': str(entry_trade1['date']),
                            'exit_date': str(exit_trade1['date']),
                            'entry_spot': entry_trade1['spot_price'],
                            'exit_spot': exit_trade1['spot_price'],
                            'leg1': {
                                'symbol': entry_trade1['symbol'],
                                'strike': self.extract_strike_price(entry_trade1['symbol']),
                                'entry_action': 'SELL' if entry_trade1['action'] == -1 else 'BUY',
                                'entry_price': entry_trade1['price'],
                                'exit_action': 'SELL' if exit_trade1['action'] == -1 else 'BUY',
                                'exit_price': exit_trade1['price']
                            },
                            'leg2': {
                                'symbol': entry_trade2['symbol'],
                                'strike': self.extract_strike_price(entry_trade2['symbol']),
                                'entry_action': 'SELL' if entry_trade2['action'] == -1 else 'BUY',
                                'entry_price': entry_trade2['price'],
                                'exit_action': 'SELL' if exit_trade2['action'] == -1 else 'BUY',
                                'exit_price': exit_trade2['price']
                            },
                            'net_credit': round(net_credit * self.lot_size, 2),
                            'net_debit': round(net_debit * self.lot_size, 2),
                            'final_pnl': round(final_pnl, 2),
                            'status': 'CLOSED'
                        }
                    else:
                        # Open position
                        if entry_trade1['action'] == -1:  # Sell first
                            net_credit = entry_trade1['price'] - entry_trade2['price']
                        else:  # Buy first
                            net_credit = entry_trade2['price'] - entry_trade1['price']
                        
                        strategy = {
                            'number': strategy_number,
                            'type': strategy_type,
                            'entry_date': str(entry_trade1['date']),
                            'exit_date': 'OPEN',
                            'entry_spot': entry_trade1['spot_price'],
                            'exit_spot': 'OPEN',
                            'leg1': {
                                'symbol': entry_trade1['symbol'],
                                'strike': self.extract_strike_price(entry_trade1['symbol']),
                                'entry_action': 'SELL' if entry_trade1['action'] == -1 else 'BUY',
                                'entry_price': entry_trade1['price'],
                                'exit_action': 'OPEN',
                                'exit_price': 'OPEN'
                            },
                            'leg2': {
                                'symbol': entry_trade2['symbol'],
                                'strike': self.extract_strike_price(entry_trade2['symbol']),
                                'entry_action': 'SELL' if entry_trade2['action'] == -1 else 'BUY',
                                'entry_price': entry_trade2['price'],
                                'exit_action': 'OPEN',
                                'exit_price': 'OPEN'
                            },
                            'net_credit': round(net_credit * self.lot_size, 2),
                            'net_debit': 'OPEN',
                            'final_pnl': 'OPEN',
                            'status': 'OPEN'
                        }
                    
                    self.strategies.append(strategy)
                    strategy_number += 1
                    
                    # Skip the next trade as we've processed it
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    
    def calculate_summary(self):
        """Calculate summary statistics"""
        closed_strategies = [s for s in self.strategies if s['status'] == 'CLOSED']
        
        total_pnl = sum(s['final_pnl'] for s in closed_strategies)
        winning_trades = len([s for s in closed_strategies if s['final_pnl'] > 0])
        losing_trades = len([s for s in closed_strategies if s['final_pnl'] < 0])
        win_rate = (winning_trades / len(closed_strategies) * 100) if closed_strategies else 0
        open_positions = len([s for s in self.strategies if s['status'] == 'OPEN'])
        
        return {
            'total_pnl': total_pnl,
            'total_strategies': len(self.strategies),
            'closed_strategies': len(closed_strategies),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'open_positions': open_positions
        }
    
    def generate_html_report(self, output_file='nifty_options_report.html'):
        """Generate HTML report"""
        summary = self.calculate_summary()
        
        # Start building HTML
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIFTY Options Trading P&L Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 3rem;
            font-weight: 300;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .summary {{
            padding: 40px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-bottom: 3px solid #3498db;
        }}
        
        .summary h2 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2rem;
            font-weight: 300;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
        }}
        
        .summary-card {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .summary-card:hover {{
            transform: translateY(-5px);
        }}
        
        .summary-card h3 {{
            color: #7f8c8d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .summary-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #27ae60;
        }}
        
        .summary-card .value.negative {{
            color: #e74c3c;
        }}
        
        .summary-card .value.neutral {{
            color: #3498db;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section-title {{
            color: #2c3e50;
            font-size: 2rem;
            font-weight: 300;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .table-container {{
            overflow-x: auto;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        
        th {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 20px 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 20px 15px;
            border-bottom: 1px solid #e9ecef;
            vertical-align: top;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e3f2fd;
        }}
        
        .strategy-name {{
            font-weight: 700;
            color: #2c3e50;
            font-size: 1.1rem;
        }}
        
        .strategy-number {{
            background: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            margin-right: 10px;
        }}
        
        .pnl-positive {{
            color: #27ae60;
            font-weight: bold;
            font-size: 1.1rem;
        }}
        
        .pnl-negative {{
            color: #e74c3c;
            font-weight: bold;
            font-size: 1.1rem;
        }}
        
        .pnl-open {{
            color: #f39c12;
            font-weight: bold;
            font-size: 1.1rem;
        }}
        
        .spot-movement {{
            font-size: 0.95rem;
            line-height: 1.4;
        }}
        
        .spot-change {{
            font-size: 0.85rem;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .trade-details {{
            font-size: 0.9rem;
            line-height: 1.5;
        }}
        
        .trade-leg {{
            margin-bottom: 12px;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        
        .sell-leg {{
            background: #ffebee;
            border-left-color: #e74c3c;
        }}
        
        .buy-leg {{
            background: #e8f5e8;
            border-left-color: #27ae60;
        }}
        
        .open-leg {{
            background: #fff3e0;
            border-left-color: #f39c12;
        }}
        
        .leg-title {{
            font-weight: 600;
            margin-bottom: 5px;
        }}
        
        .leg-details {{
            font-size: 0.85rem;
            color: #666;
        }}
        
        .total-row {{
            background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
            font-weight: bold;
            font-size: 1.2rem;
        }}
        
        .total-row td {{
            border-top: 3px solid #3498db;
            padding: 25px 15px;
        }}
        
        .credit-debit {{
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        
        .credit {{
            color: #27ae60;
        }}
        
        .debit {{
            color: #e74c3c;
        }}
        
        .date-badge {{
            background: #ecf0f1;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85rem;
            color: #2c3e50;
            white-space: nowrap;
        }}
        
        .open-badge {{
            background: #f39c12;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-left: 8px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
                border-radius: 12px;
            }}
            
            .header {{
                padding: 25px;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .summary, .content {{
                padding: 25px;
            }}
            
            .summary-grid {{
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            
            .summary-card {{
                padding: 20px;
            }}
            
            .summary-card .value {{
                font-size: 2rem;
            }}
            
            table {{
                font-size: 0.85rem;
            }}
            
            th, td {{
                padding: 12px 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NIFTY Options Trading Report</h1>
            <div class="subtitle">Supertrend Strategy Analysis | Generated on {datetime.now().strftime("%B %d, %Y")}</div>
        </div>
        
        <div class="summary">
            <h2>Performance Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total P&L (Closed)</h3>
                    <div class="value{'negative' if summary['total_pnl'] < 0 else ''}">₹{summary['total_pnl']:,.2f}</div>
                </div>
                <div class="summary-card">
                    <h3>Total Strategies</h3>
                    <div class="value neutral">{summary['total_strategies']}</div>
                </div>
                <div class="summary-card">
                    <h3>Closed Positions</h3>
                    <div class="value neutral">{summary['closed_strategies']}</div>
                </div>
                <div class="summary-card">
                    <h3>Winning Trades</h3>
                    <div class="value">{summary['winning_trades']}</div>
                </div>
                <div class="summary-card">
                    <h3>Win Rate</h3>
                    <div class="value">{summary['win_rate']}%</div>
                </div>
                <div class="summary-card">
                    <h3>Open Positions</h3>
                    <div class="value neutral">{summary['open_positions']}</div>
                </div>
            </div>
        </div>
        
        <div class="content">
            <h2 class="section-title">Detailed Trade Analysis</h2>
            
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Period</th>
                            <th>Spot Movement</th>
                            <th>Trade Details</th>
                            <th>Credit/Debit</th>
                            <th>P&L (₹)</th>
                        </tr>
                    </thead>
                    <tbody>'''
        
        # Add strategy rows
        for strategy in self.strategies:
            spot_change = ""
            if strategy['exit_spot'] != 'OPEN':
                change = strategy['exit_spot'] - strategy['entry_spot']
                spot_change = f"({'+' if change >= 0 else ''}{change:.0f} points)"
            else:
                spot_change = "(Active)"
            
            leg1_class = "sell-leg" if strategy['leg1']['entry_action'] == 'SELL' else "buy-leg"
            leg2_class = "sell-leg" if strategy['leg2']['entry_action'] == 'SELL' else "buy-leg"
            
            if strategy['status'] == 'OPEN':
                leg1_class = leg2_class = "open-leg"
            
            option_type1 = "CE" if "CE" in strategy['leg1']['symbol'] else "PE"
            option_type2 = "CE" if "CE" in strategy['leg2']['symbol'] else "PE"
            
            pnl_class = ""
            pnl_value = ""
            if strategy['final_pnl'] == 'OPEN':
                pnl_class = "pnl-open"
                pnl_value = "OPEN"
            else:
                pnl_class = "pnl-positive" if strategy['final_pnl'] > 0 else "pnl-negative"
                pnl_value = f"{'+ ' if strategy['final_pnl'] > 0 else ''}₹{strategy['final_pnl']:,.0f}"
            
            html_content += f'''
                        <tr>
                            <td class="strategy-name">
                                <span class="strategy-number">#{strategy['number']}</span>{strategy['type']}
                                {' <span class="open-badge">OPEN</span>' if strategy['status'] == 'OPEN' else ''}
                            </td>
                            <td>
                                <div class="date-badge">{strategy['entry_date']}</div>
                                <div style="text-align: center; margin: 5px 0;">to</div>
                                <div class="date-badge">{strategy['exit_date']}</div>
                            </td>
                            <td class="spot-movement">
                                {strategy['entry_spot']:,.0f} → {strategy['exit_spot'] if strategy['exit_spot'] != 'OPEN' else 'Current'}<br>
                                <span class="spot-change">{spot_change}</span>
                            </td>
                            <td class="trade-details">
                                <div class="trade-leg {leg1_class}">
                                    <div class="leg-title">{strategy['leg1']['entry_action']} {strategy['leg1']['strike']} {option_type1}</div>
                                    <div class="leg-details">Entry: ₹{strategy['leg1']['entry_price']} → Exit: ₹{strategy['leg1']['exit_price']}</div>
                                </div>
                                <div class="trade-leg {leg2_class}">
                                    <div class="leg-title">{strategy['leg2']['entry_action']} {strategy['leg2']['strike']} {option_type2}</div>
                                    <div class="leg-details">Entry: ₹{strategy['leg2']['entry_price']} → Exit: ₹{strategy['leg2']['exit_price']}</div>
                                </div>
                            </td>
                            <td>
                                <div class="credit-debit">
                                    <div class="credit">Credit: ₹{strategy['net_credit']:,.0f}</div>
                                    <div class="{'debit' if strategy['net_debit'] != 'OPEN' else ''}" style="{'color: #f39c12;' if strategy['net_debit'] == 'OPEN' else ''}">
                                        {'Debit: ₹' + str(int(strategy['net_debit'])) if strategy['net_debit'] != 'OPEN' else 'Status: OPEN'}
                                    </div>
                                </div>
                            </td>
                            <td class="{pnl_class}">{pnl_value}</td>
                        </tr>'''
        
        # Add total row
        html_content += f'''
                        <tr class="total-row">
                            <td colspan="5" style="text-align: right;"><strong>TOTAL P&L (Closed Positions)</strong></td>
                            <td class="pnl-positive"><strong>{'+ ' if summary['total_pnl'] > 0 else ''}₹{summary['total_pnl']:,.0f}</strong></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div style="margin-top: 40px; padding: 30px; background: #f8f9fa; border-radius: 12px; border-left: 5px solid #3498db;">
                <h3 style="color: #2c3e50; margin-bottom: 20px;">Key Insights</h3>
                <ul style="line-height: 1.8; color: #2c3e50;">'''
        
        # Find best and worst performing strategies
        closed_strategies = [s for s in self.strategies if s['status'] == 'CLOSED']
        if closed_strategies:
            best_strategy = max(closed_strategies, key=lambda x: x['final_pnl'])
            html_content += f'<li><strong>Best Performance:</strong> Strategy #{best_strategy["number"]} ({best_strategy["type"]}) generated ₹{best_strategy["final_pnl"]:,.0f} profit</li>'
        
        html_content += f'''
                    <li><strong>Strategy Mix:</strong> {len([s for s in self.strategies if 'Short Call' in s['type']])} Short Call Spreads, {len([s for s in self.strategies if 'Long Put' in s['type']])} Long Put Spreads showing diversified approach</li>
                    <li><strong>Risk Management:</strong> Consistent use of Supertrend signals for entries and exits</li>
                    <li><strong>Lot Size:</strong> All calculations based on NIFTY lot size of {self.lot_size} contracts</li>'''
        
        if summary['open_positions'] > 0:
            open_credit = sum(s['net_credit'] for s in self.strategies if s['status'] == 'OPEN')
            html_content += f'<li><strong>Open Positions:</strong> {summary["open_positions"]} active position(s) with ₹{open_credit:,.0f} total credit collected</li>'
        
        html_content += '''
                </ul>
            </div>
        </div>
    </div>
</body>
</html>'''
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_file}")
        return output_file
    
    def process_file(self, csv_file_path, output_file='nifty_options_report.html'):
        """Main function to process CSV and generate report"""
        print(f"Reading CSV file: {csv_file_path}")
        df = self.read_csv(csv_file_path)
        
        if df is None:
            return None
        
        print(f"Analyzing {len(df)} trades...")
        self.analyze_trades(df)
        
        print(f"Found {len(self.strategies)} strategies")
        summary = self.calculate_summary()
        print(f"Total P&L: ₹{summary['total_pnl']:,.2f}")
        print(f"Win Rate: {summary['win_rate']}%")
        
        print("Generating HTML report...")
        return self.generate_html_report(output_file)

def main():
    """Main function to run the analyzer"""
    import sys
    
    # Get CSV file path from command line argument or ask user
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("Enter the path to your CSV file: ")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return
    
    # Get lot size from user (default 75)
    try:
        lot_size = int(input("Enter NIFTY lot size (default 75): ") or "75")
    except ValueError:
        lot_size = 75
        print("Using default lot size: 75")
    
    # Create analyzer and process file
    analyzer = NiftyOptionsAnalyzer(lot_size=lot_size)
    output_file = analyzer.process_file(csv_file)
    
    if output_file:
        # Ask if user wants to open the report
        open_report = input("Do you want to open the HTML report? (y/n): ").lower().strip()
        if open_report in ['y', 'yes']:
            webbrowser.open(f'file://{os.path.abspath(output_file)}')

if __name__ == "__main__":
    main()