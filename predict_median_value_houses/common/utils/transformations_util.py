import pandas as pd


def data_to_dataframe(data):
    """
    converts structured dats to dataframe
    :param data: structured data
    :return: pandas dataframe
    """
    data_df = pd.DataFrame(data)
    return data_df


