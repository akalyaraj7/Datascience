from common.services.logger import get_logger
log = get_logger()

def get_data_statistics(data):
    """
    Calculates various statistics about data
    :return:
    """
    log.info(f'Data description : ')
    log.info(f'Keys: {data.keys()}')
    log.info(f'Shape :{data.data.shape}')
    log.info(f'Features are ; {data.feature_names}')
    log.info(f'Description of the data {data.DESCR}')
    return


def get_dataframe_statistics(data):
    """
    Obtain all the information about the features in the dataframe
    :param data:
    :return:
    """
    log.info(f'Information about the data : \n {data.info()}')
    log.info(f'Summary statistics of the columns : \n {data.describe()}')
    return