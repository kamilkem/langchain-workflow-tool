from random import getrandbits

from langchain_core.tools import tool


@tool
def delegate_to_human() -> bool:
    """Delegate conversation to real human."""
    return bool(getrandbits(1))
