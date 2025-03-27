import pandas as pd
import numpy as np
import datetime
from calendar import monthrange


def random_date_generator(start_date, size):
    """genrates random dates for invoice dataset
    """
    d0 = datetime.datetime.strptime(start_date, '%d/%m/%Y')
    max_day = monthrange(d0.year, d0.month)[1]

    random_dates = []
    for i in range(size):
        random_dates.append( d0 + datetime.timedelta(days=np.random.randint(int(max_day/2), max_day+1)) )
    random_dates = np.array(random_dates)
    return random_dates


def invoice(type):
    """Creates the dataframe

    Args:
        type: check what type of dataset is it business or personal expense

    DataFrame:
        InvoiceNumber: Receipt number
        Date: date of expense
        Amount: Spent amount
        Vendor: vendor name
        Services: list of the services used or the products bought
    """
    InvoiceNumber = np.random.randint(10000000,10000000000000, size=(1,10000))
    Date = random_date_generator('01/01/2020', size=10000)
    Amount = np.random.randint(1, 1000000, size=(1,10000))
    Vendor = [] # need to get this from some dataset
    Services = [] # need to get this from some dataset

    data = { "InvoiceNumber": InvoiceNumber,
            "Date": Date,
            "Amount": Amount,
            "Vendor": Vendor,
            "Services": Services
    }

    df = pd.DataFrame(data)
    df.to_parquet(f"invoice{type}.parquet")


if __name__ == '__main__':
    invoice('personal')
    invoice('business')
