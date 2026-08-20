import functools
import inspect
import os


def log_call(func):
    """Print the file, class, and method whenever a function is called."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        file_name = os.path.basename(inspect.getfile(func))
        method_name = func.__qualname__

        print(f"[CALL] {file_name} :: {method_name}")

        return func(*args, **kwargs)

    return wrapper