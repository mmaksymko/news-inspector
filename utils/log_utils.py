import logging
from functools import wraps

def log_output(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logging.log(level, f"{func.__name__} returned {result!r}")
            return result
        return wrapper
    return decorator

def log_input(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.log(level, f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_io(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.log(level, f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            logging.log(level, f"{func.__name__} returned {result!r}")
            return result
        return wrapper
    return decorator

def log_runtime(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.log(level, f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    return decorator

def log_exceptions(level=logging.ERROR):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.log(level, f"Exception in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def log_execution(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.log(level, f"Executing {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator