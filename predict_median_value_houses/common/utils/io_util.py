from sklearn.datasets import load_boston


def load_data():
    """
    Loads boston data
    :return:
    """
    data = load_boston()
    return data
