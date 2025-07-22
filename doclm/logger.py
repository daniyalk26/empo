import os
import logging
# import logging.config
# from logging.config import dictConfig

log_dir = os.getenv('LOGDIR', './log')
if log_dir:
    os.makedirs(log_dir, exist_ok=True)


class CustomFilter(logging.Filter):
    """A custom filter"""
    #TODO: see if this class can be removed
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        try:
            if args:
                record.msg % record.args
        except TypeError:
            record.args = (''.join([str(a) for a in args]),)
            return True
        except Exception:
            return True
        else:
            return True


# logger.addFilter(NoParsingFilter())

configuration = {
    'version': 1,
    # "filters": {"customfilter": {"()": lambda: CustomFilter()}},  # pylint: disable=W0108
    'formatters': {
        'fileFormatter': {
            'format': '[%(asctime)s]:%(levelname)s: %(process)d: %(thread)d: %(name)s: %(funcName)s :%(message)s',  #
        },
        # 'streamFormatter': {
        #     'format': '[%(asctime)s]:%(levelname)s: %(process)d: %(thread)d: %(name)s: %(funcName)s :%(message)s',
        # }
    },
    'handlers': {
        'appFileHandler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_dir, "doclm.log"),  # {os.getenv("LOGDIR")}
            'maxBytes': 2000000,
            'backupCount': 30,
            'formatter': 'fileFormatter',
            # "filters": ["customfilter"]
        },
        'sqlalchemyFileHandler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_dir, "sql.log"),  # {os.getenv("LOGDIR")}
            'maxBytes': 2000000,
            'backupCount': 30,
            'formatter': 'fileFormatter',
            # "filters": ["customfilter"]
        },
        "streamhandler": {
            # "filters": ["customfilter"],
            "formatter": "fileFormatter",
            "class": "logging.StreamHandler"
        },

        # 'consolehandel': {
        #     'class': 'logging.StreamHandler',
        #     'stream': 'ext://sys.stdout',  # {os.getenv("LOGDIR")}
        #     'formatter': 'streamFormatter',
        #     'level': 'INFO'
        # },
    },
    'loggers': {
        'doclogger': {
            'level': 'DEBUG',
            'handlers': ['appFileHandler', 'streamhandler'],
            'qualname': 'doclogger',
            'propagate': 0,
        },
        'sqlalchemy.engine': {
            'level': 'DEBUG',
            'handlers': ['sqlalchemyFileHandler', 'streamhandler'],
            'qualname': 'sqlalchemy.engine',
            'propagate': 0,
        }
    },
    # 'root': {
    #     'level': 'DEBUG',
    #     'handlers': ['file']
    # },
    'disable_existing_loggers': False

}

# logging.config.fileConfig(fname='log.conf', disable_existing_loggers=False)

# log = logging.getLogger(__name__)
# log.debug('hello')
# from logging.handlers import RotatingFileHandler

# name_logger = 'doclm'
# log_dir = os.getenv('LOGDIR', None)
#
# if log_dir:
#     os.makedirs(log_dir, exist_ok=True)
#
# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
#
# log_formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")
# log_handel.setFormatter(log_formatter)
# log.addHandler(log_handel)
#
