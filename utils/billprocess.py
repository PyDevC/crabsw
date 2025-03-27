import pandas as pd
from datetime import datetime

def parse_text_file(file_path):
    """
    Parses a plain text file containing bill details.
    Assumes that each bill is separated by one or more blank lines
    and that each field in a bill is on a separate line in the format: key: value

    :param file_path: Path to the bills text file.
    :return: A list of dictionaries, each representing one bill.
    """
    bills = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the file content by blank lines to separate individual bills.
    records = [record.strip() for record in content.split('\n\n') if record.strip()]
    
    for record in records:
        bill = {}
        for line in record.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                bill[key.strip()] = value.strip()
        if bill:  # Only add non-empty dictionaries
            bills.append(bill)
    
    return bills

def process_bill_text_file(file_path):
    """
    Processes a plain text file (with one or more bills) to perform:
      1. Recipient Name Validation: Checks if the recipient contains "truesales farma" (case-insensitive).
      2. Duplicate Bill Detection: Identifies duplicate bills by 'bill_number'.
      3. Date Comparison: Compares 'bill_date' with 'expense_entry_date' and flags mismatches.
      4. Expense Filtering: Segregates bills that are not official expenses.
         Official expenses in this example are limited to: 'hotel', 'transport', or 'food'.
      5. Multi-Bill PDF Check: Flags if the text file contains multiple bills.

    Expected keys for each bill:
      - bill_number
      - bill_date             (in a recognizable date format, e.g., YYYY-MM-DD)
      - expense_entry_date    (in a recognizable date format)
      - recipient_name
      - expense_type          (e.g., hotel, grocery, entertainment, etc.)

    :param file_path: Path to the text file containing bill details.
    :return: A dictionary with the following keys:
             - invalid_recipients: Bills whose recipient does NOT include "truesales farma".
             - duplicate_bills:   Bills flagged as duplicates (by bill_number).
             - mismatched_dates:  Bills where bill_date and expense_entry_date do not match.
             - non_official_bills: Bills whose expense_type is not in the official list.
             - multi_bill_pdf:    Boolean indicating if multiple bills were detected.
             - all_bills:         The full pandas DataFrame of the parsed bills.
    """
    # Parse the text file into a list of dictionaries
    bill_records = parse_text_file(file_path)
    
    # Create a DataFrame from the list of bills
    df = pd.DataFrame(bill_records)
    
    # Convert date fields to datetime objects for proper comparison
    for date_col in ['bill_date', 'expense_entry_date']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Task 1: Recipient Name Validation
    # Check if 'recipient_name' contains "truesales farma" (case-insensitive)
    df['valid_recipient'] = df['recipient_name'].str.contains("truesales farma", case=False, na=False)
    invalid_recipients = df[~df['valid_recipient']].copy()
    
    # Task 2: Duplicate Bill Detection based on 'bill_number'
    duplicate_bills = df[df.duplicated(subset=["bill_number"], keep=False)].copy()
    
    # Task 3: Date Comparison - flag bills where 'bill_date' does not match 'expense_entry_date'
    mismatched_dates = df[df['bill_date'] != df['expense_entry_date']].copy()
    
    # Task 4: Expense Filtering - identify bills not related to official expenses.
    # Define the official expense types (example: hotel, transport, food)
    official_expenses = ['hotel', 'transport', 'food']
    df['expense_type'] = df['expense_type'].fillna("").astype(str)
    df['official_expense'] = df['expense_type'].str.lower().apply(
        lambda x: any(exp in x for exp in official_expenses)
    )
    non_official_bills = df[~df['official_expense']].copy()
    
    # Task 5: Multi-Bill PDF Check - if more than one bill is present in the file,
    # it likely came from a multi-bill PDF.
    multi_bill_pdf = len(df) > 1

    results = {
        "invalid_recipients": invalid_recipients,
        "duplicate_bills": duplicate_bills,
        "mismatched_dates": mismatched_dates,
        "non_official_bills": non_official_bills,
        "multi_bill_pdf": multi_bill_pdf,
        "all_bills": df
    }
    
    return results

# Example usage:
if __name__ == '__main__':
    file_path = "extracted_bills.txt"  # Replace with the path to your text file
    results = process_bill_text_file(file_path)
    
    print("Bills with invalid recipient (does not include 'truesales farma'):")
    print(results["invalid_recipients"].to_string(index=False), "\n")
    
    print("Duplicate bills (by bill_number):")
    print(results["duplicate_bills"].to_string(index=False), "\n")
    
    print("Bills with date mismatches (bill_date vs. expense_entry_date):")
    print(results["mismatched_dates"].to_string(index=False), "\n")
    
    print("Bills not related to official expenses (not hotel, transport, or food):")
    print(results["non_official_bills"].to_string(index=False), "\n")
    
    if results["multi_bill_pdf"]:
        print("Note: The text file contains multiple bills (likely from a multi-bill PDF).")
    else:
        print("The text file contains a single bill.")
