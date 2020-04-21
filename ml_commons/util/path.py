import os


def uniquify_path(path):
    root, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(path):
        path = f'{root}_{i}{ext}'
        i += 1
    return path
