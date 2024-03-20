"""Some utility functions to test for python types."""


def is_strlist(obj: object) -> bool:
    """Test if the object is a list of strings."""
    return (bool(obj) and not isinstance(obj, str)
            and all(isinstance(elem, str) for elem in obj))


def is_str_or_strlist(obj: object) -> bool:
    return isinstance(obj, str) or is_strlist(obj)
