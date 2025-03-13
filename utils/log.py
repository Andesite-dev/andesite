import os
os.environ['LOGURU_FORMAT'] = '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>][{module}.py]:<bold>{line}</bold> on {function}: <bold>{message}</bold>'

from loguru import logger

andesite_logger = logger