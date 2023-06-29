<<<<<<< HEAD
import logging
import logging.config
import yaml

try:
    with open('/configs/logging_config.yaml', 'r') as f:
        conf = yaml.safe_load(f.read())
        logging.config.dictConfig(conf)
        logging.captureWarnings(True)
except:
    with open('../Trainer/configs/logging_config.yaml', 'r') as f:
        conf = yaml.safe_load(f.read())
        logging.config.dictConfig(conf)
        logging.captureWarnings(True)


def get_logger(name: str):
    return logging.getLogger(name=name)
=======
import logging
import logging.config
import yaml

try:
    with open('/configs/logging_config.yaml', 'r') as f:
        conf = yaml.safe_load(f.read())
        logging.config.dictConfig(conf)
        logging.captureWarnings(True)
except:
    with open('../Trainer/configs/logging_config.yaml', 'r') as f:
        conf = yaml.safe_load(f.read())
        logging.config.dictConfig(conf)
        logging.captureWarnings(True)


def get_logger(name: str):
    return logging.getLogger(name=name)
>>>>>>> d18bb69c323308ecd641f0ef77695d31e1fe144f
