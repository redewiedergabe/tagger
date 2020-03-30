import logging

__version__ = '1.0.0'

logger = logging.getLogger('rwtagger')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())