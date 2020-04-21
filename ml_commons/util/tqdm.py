import logging
from tqdm import tqdm


def configure_logging(debug=False):
    class Stream(object):
        @classmethod
        def write(cls, msg):
            if msg == '\n':
                return
            tqdm.write(msg, end='\n')

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        stream=Stream,
        format="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S"
    )
