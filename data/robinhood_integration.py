"""
Robinhood Portfolio Integration

This module provides multiple ways to integrate your Robinhood portfolio data
with the investment advisor system. Since Robinhood doesn't have an official API
for retail users, this module offers several practical approaches:

1. CSV Export Integration - Import from Robinhood CSV exports
2. Manual Portfolio Entry - Simple interface to input positions
3. Screenshot/PDF Parser - Parse account statements (future enhancement)
4. Portfolio Sync Utilities - Keep your data updated

IMPORTANT SECURITY NOTE:
This module does NOT store any login credentials or sensitive information.
All portfolio data is kept locally and encrypted if desired.
"""

import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobinhoodPortfolioManager:
    """
    Manages Robinhood portfolio data integration and synchronization.
    
    This class provides secure, local methods to import and manage your
    Robinhood portfolio data without requiring API access or credentials.
    """
    
    def __init__(self, data_directory: str = "portfolio_data"):
        """
        Initialize the portfolio manager.
        
        Args:
            data_directory: Directory to store portfolio data files
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        
        # Portfolio data storage
        self.current_positions = {}
        self.account_info = {}
        self.transaction_history = []
        self.portfolio_history = []
        
        # File paths
        self.positions_file = self.data_directory / "current_positions.json"
        self.account_file = self.data_directory / "account_info.json"
        self.transactions_file = self.data_directory / "transactions.csv"
        self.history_file = self.data_directory / "portfolio_history.json"
        
        logger.info(f"Portfolio manager initialized with data directory: {self.data_directory}")
    
    def import_from_csv_export(self, csv_file_path: str) -> Dict[str, Any]:
        """
        Import portfolio data from Robinhood CSV export.
        
        Robinhood allows you to export your data as CSV files. This function
        parses those files and extracts your current positions and transaction history.
        
        How to get CSV from Robinhood:
        1. Go to Robinhood Web (robinhood.com)
        2. Click Account → Settings
        3. Scroll to "Account Information"
        4. Click "Export Account Data"
        5. Download the CSV files when ready
        
        Args:
            csv_file_path: Path to the Robinhood CSV export file
            
        Returns:
            Dictionary with imported portfolio data
        """
        logger.info(f"Importing portfolio data from CSV: {csv_file_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Robinhood CSV formats can vary, so we'll try to detect the format
            columns = [col.lower().strip() for col in df.columns]
            
            imported_data = {
                'positions': {},
                'transactions': [],
                'account_summary': {},
                'import_timestamp': datetime.now().isoformat()
            }
            
            # 1. DETECT FILE TYPE AND PARSE ACCORDINGLY
            
            if 'instrument' in columns or 'symbol' in columns:
                # This looks like a positions/holdings file
                imported_data['positions'] = self._parse_positions_csv(df)
                logger.info(f"Imported {len(imported_data['positions'])} positions")
                
            elif 'side' in columns or 'type' in columns:
                # This looks like a transactions file
                imported_data['transactions'] = self._parse_transactions_csv(df)
                logger.info(f"Imported {len(imported_data['transactions'])} transactions")
                
            else:
                # Try to parse as general portfolio data
                logger.warning("Unknown CSV format, attempting general parsing...")
                imported_data = self._parse_general_csv(df)
            
            # 2. SAVE IMPORTED DATA
            self._save_imported_data(imported_data)
            
            # 3. UPDATE CURRENT PORTFOLIO STATE
            if imported_data['positions']:
                self.current_positions = imported_data['positions']
                self._save_current_positions()
            
            if imported_data['transactions']:
                self.transaction_history.extend(imported_data['transactions'])
                self._save_transaction_history()
            
            logger.info("CSV import completed successfully")
            return imported_data
            
        except Exception as e:
            logger.error(f"Error importing CSV file: {e}")
            return {'error': str(e)}
    
    def _parse_positions_csv(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Parse positions/holdings CSV format."""
        positions = {}
        
        # Common column mappings for Robinhood exports
        column_mappings = {
            'symbol': ['symbol', 'instrument', 'ticker'],
            'quantity': ['quantity', 'shares', 'amount'],
            'average_cost': ['average_cost', 'avg_cost', 'cost_basis', 'average_price'],
            'current_price': ['current_price', 'market_price', 'last_price'],
            'market_value': ['market_value', 'current_value', 'value'],
            'total_return': ['total_return', 'unrealized_pl', 'gain_loss'],
            'total_return_percent': ['total_return_percent', 'return_percent', 'gain_loss_percent']
        }
        
        # Find actual column names
        actual_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for col in df.columns:
                if col.lower().strip() in [name.lower() for name in possible_names]:
                    actual_columns[standard_name] = col
                    break
        
        # Parse each row
        for _, row in df.iterrows():
            try:
                symbol = None
                
                # Extract symbol
                if 'symbol' in actual_columns:
                    symbol = str(row[actual_columns['symbol']]).strip().upper()
                
                if symbol and symbol != 'NAN':
                    position_data = {'symbol': symbol}
                    
                    # Extract other fields with error handling
                    for field, col_name in actual_columns.items():
                        if field != 'symbol' and col_name in row.index:
                            try:
                                value = row[col_name]
                                
                                # Clean and convert numeric values
                                if pd.notna(value):
                                    if isinstance(value, str):
                                        # Remove currency symbols and commas
                                        cleaned_value = value.replace('$', '').replace(',', '').strip()
                                        if cleaned_value.replace('.', '').replace('-', '').isdigit():
                                            position_data[field] = float(cleaned_value)
                                        else:
                                            position_data[field] = value
                                    else:
                                        position_data[field] = float(value) if pd.notna(value) else 0
                                
                            except (ValueError, TypeError):
                                logger.warning(f"Could not parse {field} for {symbol}: {row[col_name]}")
                                continue
                    
                    # Calculate missing fields if possible
                    if 'quantity' in position_data and 'current_price' in position_data:
                        if 'market_value' not in position_data:
                            position_data['market_value'] = position_data['quantity'] * position_data['current_price']
                    
                    if 'quantity' in position_data and 'average_cost' in position_data:
                        if 'cost_basis' not in position_data:
                            position_data['cost_basis'] = position_data['quantity'] * position_data['average_cost']
                    
                    positions[symbol] = position_data
                    
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue
        
        return positions
    
    def _parse_transactions_csv(self, df: pd.DataFrame) -> List[Dict]:
        """Parse transactions CSV format."""
        transactions = []
        
        # Common transaction column mappings
        column_mappings = {
            'date': ['date', 'created_at', 'transaction_date', 'executed_at'],
            'symbol': ['symbol', 'instrument', 'ticker'],
            'side': ['side', 'type', 'transaction_type', 'action'],
            'quantity': ['quantity', 'shares', 'amount'],
            'price': ['price', 'execution_price', 'fill_price'],
            'total_amount': ['total_amount', 'net_amount', 'amount'],
            'fees': ['fees', 'commission', 'regulatory_fees']
        }
        
        # Find actual column names
        actual_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for col in df.columns:
                if col.lower().strip() in [name.lower() for name in possible_names]:
                    actual_columns[standard_name] = col
                    break
        
        # Parse each transaction
        for _, row in df.iterrows():
            try:
                transaction = {}
                
                # Extract fields
                for field, col_name in actual_columns.items():
                    if col_name in row.index and pd.notna(row[col_name]):
                        value = row[col_name]
                        
                        if field == 'date':
                            # Parse date
                            try:
                                transaction[field] = pd.to_datetime(value).isoformat()
                            except:
                                transaction[field] = str(value)
                        
                        elif field in ['quantity', 'price', 'total_amount', 'fees']:
                            # Parse numeric values
                            try:
                                if isinstance(value, str):
                                    cleaned = value.replace('$', '').replace(',', '').strip()
                                    transaction[field] = float(cleaned)
                                else:
                                    transaction[field] = float(value)
                            except:
                                transaction[field] = 0
                        
                        else:
                            # String values
                            transaction[field] = str(value).strip()
                
                if 'symbol' in transaction and transaction['symbol']:
                    transaction['symbol'] = transaction['symbol'].upper()
                    transactions.append(transaction)
                
            except Exception as e:
                logger.warning(f"Error parsing transaction row: {e}")
                continue
        
        return transactions
    
    def _parse_general_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Attempt to parse unknown CSV format."""
        # This is a fallback parser for unknown formats
        data = {
            'positions': {},
            'transactions': [],
            'raw_data': df.to_dict('records'),
            'columns_found': list(df.columns)
        }
        
        logger.info(f"Found columns: {list(df.columns)}")
        logger.info("Please check the raw_data field and contact support for format-specific parsing")
        
        return data
    
    def manual_position_entry(self) -> Dict[str, Dict]:
        """
        Interactive command-line interface for manually entering positions.
        
        This provides a simple way to input your current positions if you
        don't have CSV export or prefer manual entry.
        
        Returns:
            Dictionary with manually entered positions
        """
        print("\n" + "="*60)
        print("MANUAL PORTFOLIO ENTRY")
        print("="*60)
        print("Enter your current stock positions. Press Enter with empty symbol to finish.")
        print("Example: AAPL, 10, 150.50")
        
        positions = {}
        
        while True:
            try:
                print(f"\nPosition #{len(positions) + 1}:")
                
                # Get symbol
                symbol = input("Stock Symbol (or press Enter to finish): ").strip().upper()
                if not symbol:
                    break
                
                # Get quantity
                quantity_input = input(f"Number of shares for {symbol}: ").strip()
                if not quantity_input:
                    continue
                
                try:
                    quantity = float(quantity_input)
                except ValueError:
                    print("Invalid quantity. Please enter a number.")
                    continue
                
                # Get average cost (optional)
                avg_cost_input = input(f"Average cost per share for {symbol} (optional): ").strip()
                avg_cost = None
                if avg_cost_input:
                    try:
                        avg_cost = float(avg_cost_input.replace('$', ''))
                    except ValueError:
                        print("Invalid price format. Skipping average cost.")
                
                # Create position entry
                position = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_method': 'manual',
                    'entry_timestamp': datetime.now().isoformat()
                }
                
                if avg_cost is not None:
                    position['average_cost'] = avg_cost
                    position['cost_basis'] = quantity * avg_cost
                
                positions[symbol] = position
                print(f"✓ Added {quantity} shares of {symbol}")
                
            except KeyboardInterrupt:
                print("\nEntry cancelled.")
                break
            except Exception as e:
                print(f"Error: {e}. Please try again.")
                continue
        
        if positions:
            # Save positions
            self.current_positions.update(positions)
            self._save_current_positions()
            
            print(f"\n✓ Successfully entered {len(positions)} positions:")
            for symbol, pos in positions.items():
                shares = pos['quantity']
                cost = pos.get('average_cost', 'N/A')
                print(f"  {symbol}: {shares} shares @ ${cost}")
        
        return positions
    
    def update_account_info(self, account_data: Dict[str, Any]):
        """
        Update account information.
        
        Args:
            account_data: Dictionary with account information
                - total_value: Current total portfolio value
                - buying_power: Available buying power
                - day_change: Today's change in portfolio value
                - day_change_percent: Today's percentage change
        """
        self.account_info.update({
            **account_data,
            'last_updated': datetime.now().isoformat()
        })
        
        self._save_account_info()
        logger.info("Account information updated")
    
    def get_current_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current portfolio state.
        
        Returns:
            Dictionary with portfolio summary including positions,
            total value, allocation, and key metrics
        """
        if not self.current_positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'message': 'No positions found. Use import_from_csv_export() or manual_position_entry() to add positions.'
            }
        
        # Calculate portfolio metrics
        total_value = 0
        total_cost_basis = 0
        positions_summary = {}
        
        for symbol, position in self.current_positions.items():
            quantity = position.get('quantity', 0)
            avg_cost = position.get('average_cost', 0)
            market_value = position.get('market_value', 0)
            
            # If we don't have market value, we'll need to fetch current prices
            if market_value == 0 and quantity > 0:
                # This would require a price lookup - we'll estimate or mark as unknown
                market_value = quantity * avg_cost if avg_cost > 0 else 0
                position['estimated_market_value'] = True
            
            cost_basis = position.get('cost_basis', quantity * avg_cost if avg_cost > 0 else 0)
            
            positions_summary[symbol] = {
                'quantity': quantity,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'weight': 0  # Will calculate after total
            }
            
            total_value += market_value
            total_cost_basis += cost_basis
        
        # Calculate weights
        for symbol in positions_summary:
            if total_value > 0:
                positions_summary[symbol]['weight'] = positions_summary[symbol]['market_value'] / total_value
        
        # Calculate overall metrics
        total_return = total_value - total_cost_basis if total_cost_basis > 0 else 0
        total_return_percent = (total_return / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        summary = {
            'total_positions': len(self.current_positions),
            'total_market_value': total_value,
            'total_cost_basis': total_cost_basis,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'positions': positions_summary,
            'account_info': self.account_info,
            'last_updated': datetime.now().isoformat()
        }
        
        return summary
    
    def export_for_investment_advisor(self) -> Dict[str, Any]:
        """
        Export portfolio data in the format expected by the investment advisor.
        
        Returns:
            Dictionary formatted for use with the daily investment advisor
        """
        portfolio_summary = self.get_current_portfolio_summary()
        
        if portfolio_summary['total_positions'] == 0:
            return {
                'error': 'No positions to export',
                'message': 'Import your portfolio data first'
            }
        
        # Format for investment advisor
        advisor_format = {
            'current_positions': {},  # Symbol -> dollar value
            'portfolio_value': portfolio_summary['total_market_value'],
            'symbols_to_analyze': list(self.current_positions.keys()),
            'account_metadata': {
                'total_positions': portfolio_summary['total_positions'],
                'total_return_percent': portfolio_summary['total_return_percent'],
                'last_updated': portfolio_summary['last_updated']
            }
        }
        
        # Convert positions to dollar values
        for symbol, position_data in portfolio_summary['positions'].items():
            advisor_format['current_positions'][symbol] = position_data['market_value']
        
        return advisor_format
    
    def _save_current_positions(self):
        """Save current positions to file."""
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(self.current_positions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def _save_account_info(self):
        """Save account info to file."""
        try:
            with open(self.account_file, 'w') as f:
                json.dump(self.account_info, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving account info: {e}")
    
    def _save_transaction_history(self):
        """Save transaction history to CSV."""
        try:
            if self.transaction_history:
                df = pd.DataFrame(self.transaction_history)
                df.to_csv(self.transactions_file, index=False)
        except Exception as e:
            logger.error(f"Error saving transaction history: {e}")
    
    def _save_imported_data(self, data: Dict[str, Any]):
        """Save imported data for backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.data_directory / f"import_backup_{timestamp}.json"
            
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.info(f"Import data backed up to {backup_file}")
        except Exception as e:
            logger.error(f"Error saving backup: {e}")
    
    def load_saved_portfolio(self) -> bool:
        """
        Load previously saved portfolio data.
        
        Returns:
            True if data was loaded successfully, False otherwise
        """
        try:
            # Load positions
            if self.positions_file.exists():
                with open(self.positions_file, 'r') as f:
                    self.current_positions = json.load(f)
                logger.info(f"Loaded {len(self.current_positions)} positions")
            
            # Load account info
            if self.account_file.exists():
                with open(self.account_file, 'r') as f:
                    self.account_info = json.load(f)
                logger.info("Loaded account information")
            
            # Load transaction history
            if self.transactions_file.exists():
                df = pd.read_csv(self.transactions_file)
                self.transaction_history = df.to_dict('records')
                logger.info(f"Loaded {len(self.transaction_history)} transactions")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading saved portfolio: {e}")
            return False


def create_sample_csv():
    """Create a sample CSV file to demonstrate the expected format."""
    sample_data = [
        {
            'symbol': 'AAPL',
            'quantity': 10,
            'average_cost': 150.00,
            'current_price': 155.50,
            'market_value': 1555.00,
            'total_return': 55.00,
            'total_return_percent': 3.67
        },
        {
            'symbol': 'GOOGL',
            'quantity': 5,
            'average_cost': 2500.00,
            'current_price': 2600.00,
            'market_value': 13000.00,
            'total_return': 500.00,
            'total_return_percent': 4.00
        },
        {
            'symbol': 'MSFT',
            'quantity': 15,
            'average_cost': 300.00,
            'current_price': 310.00,
            'market_value': 4650.00,
            'total_return': 150.00,
            'total_return_percent': 3.33
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_portfolio.csv', index=False)
    print("Created sample_portfolio.csv - you can use this as a template")


def main():
    """Interactive CLI for portfolio management."""
    print("Robinhood Portfolio Integration")
    print("=" * 40)
    
    manager = RobinhoodPortfolioManager()
    
    # Try to load existing data
    if manager.load_saved_portfolio():
        print("✓ Loaded existing portfolio data")
        summary = manager.get_current_portfolio_summary()
        print(f"Current portfolio: {summary['total_positions']} positions, "
              f"${summary['total_market_value']:,.2f} total value")
    
    while True:
        print("\nOptions:")
        print("1. Import from CSV file")
        print("2. Manual position entry")
        print("3. View portfolio summary")
        print("4. Export for investment advisor")
        print("5. Create sample CSV template")
        print("6. Exit")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                csv_path = input("Enter path to CSV file: ").strip()
                if os.path.exists(csv_path):
                    result = manager.import_from_csv_export(csv_path)
                    if 'error' not in result:
                        print("✓ CSV import successful!")
                    else:
                        print(f"✗ Import failed: {result['error']}")
                else:
                    print("File not found")
            
            elif choice == '2':
                manager.manual_position_entry()
            
            elif choice == '3':
                summary = manager.get_current_portfolio_summary()
                print(f"\nPortfolio Summary:")
                print(f"Total Positions: {summary['total_positions']}")
                print(f"Total Value: ${summary['total_market_value']:,.2f}")
                print(f"Total Return: {summary['total_return_percent']:.2f}%")
                
                if summary['positions']:
                    print("\nPositions:")
                    for symbol, pos in summary['positions'].items():
                        weight = pos['weight'] * 100
                        value = pos['market_value']
                        print(f"  {symbol}: ${value:,.2f} ({weight:.1f}%)")
            
            elif choice == '4':
                export_data = manager.export_for_investment_advisor()
                if 'error' not in export_data:
                    print("✓ Portfolio data ready for investment advisor:")
                    print(f"  Portfolio Value: ${export_data['portfolio_value']:,.2f}")
                    print(f"  Symbols: {', '.join(export_data['symbols_to_analyze'])}")
                    
                    # Save to file
                    with open('portfolio_for_advisor.json', 'w') as f:
                        json.dump(export_data, f, indent=2)
                    print("  Saved to: portfolio_for_advisor.json")
                else:
                    print(f"✗ Export failed: {export_data['error']}")
            
            elif choice == '5':
                create_sample_csv()
            
            elif choice == '6':
                print("Goodbye!")
                break
            
            else:
                print("Invalid option")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
