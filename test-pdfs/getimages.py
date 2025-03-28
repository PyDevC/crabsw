from datasets import (
    load_dataset,
    load_dataset_builder,
    get_dataset_config_names
)

def get_hugging(dataset_key:str)->None:
    """gets dataset from hugginface.io

    Parameters: 
        dataset_key: It is the key that fetches dataset from huggingface
        
    Example:
        dataset_key="katanaml-org/invoices-donut-data-v1"
    """
    builder = load_dataset_builder(dataset_key)
    info_features = builder.info.features
    info_splites = builder.info.splits
    data = load_dataset(dataset_key, split="train")
    column_names = data.column_names
    shape = data.shape
    return data, column_names, shape

get_hugging("Ananthu01/7000_invoice_images_with_json")
