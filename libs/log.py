import logging


def get_logger(name):
    """
    log example:
        ERROR 2019-07-03 18:53:35,365 expense_aggregation_service.server: error message
    :param name:
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(levelname)s %(asctime)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger
