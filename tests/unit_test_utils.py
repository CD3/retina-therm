import contextlib
import os

@contextlib.contextmanager
def working_directory(path):
    last_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(last_dir)
