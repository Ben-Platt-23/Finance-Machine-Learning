"""
Portfolio Setup Script

This script helps you set up your investment advisor by importing your 
Robinhood portfolio data and configuring the system for daily use.

Usage:
    python setup_portfolio.py

This interactive script will guide you through:
1. Importing your Robinhood portfolio data
2. Setting your risk tolerance
3. Testing the system with a sample analysis
4. Scheduling daily runs (optional)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.robinhood_integration import RobinhoodPortfolioManager
from daily_investment_advisor import DailyInvestmentAdvisor


def print_header():
    """Print welcome header."""
    print("\n" + "="*80)
    print("INVESTMENT ADVISOR SETUP")
    print("="*80)
    print("Welcome! This script will help you set up your investment advisor")
    print("by importing your Robinhood portfolio and configuring the system.")
    print("="*80)


def get_robinhood_data_methods():
    """Show user different ways to import Robinhood data."""
    print("\n📊 ROBINHOOD DATA IMPORT OPTIONS")
    print("-" * 50)
    print("There are several ways to get your portfolio data into the system:")
    print()
    print("1. 📁 CSV EXPORT (Recommended)")
    print("   - Go to robinhood.com → Account → Settings")
    print("   - Scroll to 'Account Information'")
    print("   - Click 'Export Account Data'")
    print("   - Download CSV when ready")
    print()
    print("2. ✋ MANUAL ENTRY")
    print("   - Enter your positions manually")
    print("   - Good for small portfolios")
    print("   - Can be updated anytime")
    print()
    print("3. 📱 SCREENSHOT/STATEMENT")
    print("   - Take screenshot of your positions")
    print("   - Manual entry based on what you see")
    print("   - Less automated but works")


def setup_portfolio():
    """Interactive portfolio setup."""
    portfolio_manager = RobinhoodPortfolioManager()
    
    # Try to load existing data
    has_existing_data = portfolio_manager.load_saved_portfolio()
    if has_existing_data:
        summary = portfolio_manager.get_current_portfolio_summary()
        if summary['total_positions'] > 0:
            print(f"\n✅ Found existing portfolio data:")
            print(f"   Positions: {summary['total_positions']}")
            print(f"   Total Value: ${summary['total_market_value']:,.2f}")
            print(f"   Return: {summary['total_return_percent']:.2f}%")
            
            use_existing = input("\nUse existing portfolio data? (y/n): ").strip().lower()
            if use_existing == 'y':
                return portfolio_manager
    
    print("\n🔧 PORTFOLIO SETUP")
    print("-" * 30)
    
    while True:
        print("\nHow would you like to import your portfolio?")
        print("1. Import from Robinhood CSV export")
        print("2. Enter positions manually")
        print("3. Skip for now (use default watchlist)")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            csv_path = input("Enter path to your CSV file: ").strip()
            if os.path.exists(csv_path):
                print("Importing data...")
                result = portfolio_manager.import_from_csv_export(csv_path)
                if 'error' not in result:
                    print("✅ CSV import successful!")
                    break
                else:
                    print(f"❌ Import failed: {result['error']}")
                    print("Try option 2 for manual entry.")
            else:
                print("File not found. Please check the path.")
        
        elif choice == '2':
            print("\n📝 Manual position entry:")
            portfolio_manager.manual_position_entry()
            break
            
        elif choice == '3':
            print("Skipping portfolio import. You can add positions later.")
            break
            
        else:
            print("Invalid option. Please try again.")
    
    return portfolio_manager


def configure_risk_tolerance():
    """Configure risk tolerance settings."""
    print("\n⚖️  RISK TOLERANCE CONFIGURATION")
    print("-" * 40)
    print("Your risk tolerance affects position sizing and recommendations:")
    print()
    print("🟢 CONSERVATIVE")
    print("   - Max 5% per position")
    print("   - Lower volatility tolerance")
    print("   - Focus on stable, dividend-paying stocks")
    print()
    print("🟡 MODERATE (Recommended)")
    print("   - Max 10% per position")
    print("   - Balanced risk/return approach")
    print("   - Mix of growth and value stocks")
    print()
    print("🔴 AGGRESSIVE")
    print("   - Max 20% per position")
    print("   - Higher volatility tolerance")
    print("   - Focus on growth and momentum stocks")
    
    while True:
        risk_choice = input("\nSelect risk tolerance (conservative/moderate/aggressive): ").strip().lower()
        if risk_choice in ['conservative', 'moderate', 'aggressive']:
            return risk_choice
        print("Please enter: conservative, moderate, or aggressive")


def test_system(portfolio_manager, risk_tolerance):
    """Run a test analysis to make sure everything works."""
    print("\n🧪 SYSTEM TEST")
    print("-" * 20)
    print("Running a test analysis to make sure everything is working...")
    
    try:
        # Get portfolio data
        portfolio_data = portfolio_manager.export_for_investment_advisor()
        
        if 'error' not in portfolio_data:
            portfolio_value = portfolio_data['portfolio_value']
            symbols = portfolio_data['symbols_to_analyze']
        else:
            portfolio_value = 100000  # Default
            symbols = ['AAPL', 'GOOGL', 'MSFT']  # Default test symbols
        
        # Initialize advisor
        advisor = DailyInvestmentAdvisor(
            risk_tolerance=risk_tolerance,
            portfolio_value=portfolio_value
        )
        
        print(f"Testing with {len(symbols)} symbols...")
        
        # Run quick analysis
        results = advisor.analyze_investments(symbols[:3])  # Test with first 3 symbols
        
        if 'error' not in results:
            print("✅ System test successful!")
            
            # Show sample results
            summary = results.get('portfolio_summary', {})
            action_items = summary.get('action_items', [])
            
            if action_items:
                print(f"\nSample recommendations ({len(action_items)} total):")
                for item in action_items[:2]:  # Show first 2
                    action = item['action']
                    symbol = item['symbol']
                    confidence = item['confidence']
                    print(f"  {action} {symbol} (Confidence: {confidence:.1%})")
            
            return True
        else:
            print(f"❌ System test failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def create_config_file(portfolio_manager, risk_tolerance):
    """Create configuration file for easy daily use."""
    config = {
        'risk_tolerance': risk_tolerance,
        'created_date': datetime.now().isoformat(),
        'portfolio_data_location': str(portfolio_manager.data_directory),
        'setup_completed': True
    }
    
    config_file = Path('config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_file}")


def show_usage_instructions(risk_tolerance):
    """Show instructions for daily usage."""
    print("\n🎯 DAILY USAGE INSTRUCTIONS")
    print("-" * 40)
    print("Your investment advisor is now set up! Here's how to use it:")
    print()
    print("📅 DAILY ANALYSIS:")
    print("   python daily_investment_advisor.py")
    print()
    print("🔧 PORTFOLIO MANAGEMENT:")
    print("   python daily_investment_advisor.py --show-portfolio")
    print("   python daily_investment_advisor.py --manual-entry")
    print("   python daily_investment_advisor.py --import-portfolio data.csv")
    print()
    print("📊 BACKTESTING:")
    print("   python backtesting/strategy_backtester.py --start-date 2023-01-01")
    print()
    print("⚙️  CONFIGURATION:")
    print(f"   Risk Tolerance: {risk_tolerance}")
    print("   Portfolio Data: Saved locally in portfolio_data/")
    print()
    print("💡 TIPS:")
    print("   - Run daily analysis each morning before market open")
    print("   - Update portfolio data weekly or after major trades")
    print("   - Review backtesting results monthly")
    print("   - Adjust risk tolerance based on market conditions")


def main():
    """Main setup flow."""
    try:
        print_header()
        
        # Step 1: Show data import options
        get_robinhood_data_methods()
        
        input("\nPress Enter to continue with setup...")
        
        # Step 2: Setup portfolio
        portfolio_manager = setup_portfolio()
        
        # Step 3: Configure risk tolerance
        risk_tolerance = configure_risk_tolerance()
        
        # Step 4: Test the system
        print("\n" + "="*50)
        test_success = test_system(portfolio_manager, risk_tolerance)
        
        if test_success:
            # Step 5: Create config file
            create_config_file(portfolio_manager, risk_tolerance)
            
            # Step 6: Show usage instructions
            show_usage_instructions(risk_tolerance)
            
            print("\n🎉 SETUP COMPLETE!")
            print("Your investment advisor is ready to use.")
            print("Run 'python daily_investment_advisor.py' to get started!")
        
        else:
            print("\n❌ Setup incomplete due to system test failure.")
            print("Please check the error messages and try again.")
    
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
