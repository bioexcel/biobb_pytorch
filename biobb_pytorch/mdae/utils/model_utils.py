import inspect


def assert_valid_kwargs(target_cls, kwargs, context=""):
    """
    Assert that the keys in kwargs are valid parameters for target_cls.__init__.
    Raises AssertionError if invalid keys are found.

    Args:
        target_cls: class whose __init__ signature to inspect
        kwargs (dict): keyword arguments to validate
        context (str): optional context name for error messages
    """
    sig = inspect.signature(target_cls.__init__)
    params = sig.parameters
    # if **kwargs is accepted, skip strict validation
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    valid_keys = set(params.keys()) - {'self'}
    invalid = set(kwargs.keys()) - valid_keys
    assert not invalid, (
        f"Invalid {context} arguments for {target_cls.__name__}: {invalid}. "
        f"Valid parameters are: {valid_keys}"
    )
