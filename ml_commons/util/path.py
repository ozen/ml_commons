import os
from datetime import datetime


def uniquify_path(path):
    root, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(path):
        path = f'{root}_{i}{ext}'
        i += 1
    return path


def timestamp_path(path):
    root, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    return f'{root}_{timestamp}{ext}'
