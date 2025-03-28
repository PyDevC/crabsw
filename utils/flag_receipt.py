import pandas as pd

def flag(receipt_number:str, flag_description):
    """flags the receipt 
    parameters: 
        receipt_number
        flag_description: choose between ["personal", "duplicate", "date mismatch"]
    """
    return 0

def save_flags(receipt_df, flag):
    """saves the flags locally in as a database
    parameter:
        receipt_df
        flag
    """
    flags = pd.read_csv("flags.csv")
    flags["receipt_name"] = receipt_df.receipt_name
    flags["flag"] = flag
