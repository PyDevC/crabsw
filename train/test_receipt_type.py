import joblib
import importlib.util as util
import os

path = os.getcwd()

module = util.spec_from_file_location("flag_receipt", os.path.join(path, "utils", "flag_receipt.py"))
flag_receipt = util.module_from_spec(module)
module.loader.exec_module(flag_receipt)

def receipt_type(receipt, model_path='', vectorizer_path=''):
    """checks whether receipt is of business type or personal expense
    parameters:
        model_path: contains path of the model for prediction
        vectorizer_path: contains the path of vectorizer
        receipt: receipt number and services mentioned in the receipt
    returns:
        prediction whether the receipt is of personal class or business
        flags the receipt if its of personal type
    """ 
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    data = receipt[1]
    new_data_transformed = loaded_vectorizer.transform(data)
    predictions = loaded_model.predict(new_data_transformed)
    if "personal" in predictions:
        flag_receipt.flag_personal(receipt[1])
