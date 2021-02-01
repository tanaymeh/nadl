import functools

class nadl():
    """
    All the decorators will be store inside this class for ease
    """
    def method(cls):
        """
        Used to define a method that will be understandable by Tensor class
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(*args, **kwargs)
            setattr(cls, func.__name__, wrapper)
            return func
        return decorator
    
    def register():
        """
        Registerar plugin that will book-keep specific functions.
        """
        PLUGINS = dict()
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                value = func(*args, **kwargs)
                PLUGINS[func.__name__] = func
                return value
            return wrapper
        return decorator