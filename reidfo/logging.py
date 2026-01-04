import logging


def set_logging_level(level: int) -> None:
    """
    Set the logging level globally across all reidfo modules.
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    for logger_name in logging.root.manager.loggerDict:
        if "reidfo" in logger_name:
            logging.getLogger(logger_name).setLevel(level)


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger for a module.
    :param name: Typically __name__ of the module.
    :return: Logger object.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%d-%m-%Y %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
