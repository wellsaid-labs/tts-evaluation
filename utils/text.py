import re
import typing


def natural_keys(text: str) -> typing.List[typing.Union[str, int]]:
    """Returns keys (`list`) for sorting in a "natural" order.

    Inspired by: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [
        (int(char) if char.isdigit() else char)
        for char in re.split(r"(\d+)", str(text))
    ]


def numbers_then_natural_keys(
    text: str,
) -> typing.List[typing.List[typing.Union[str, int]]]:
    """Returns keys (`list`) for sorting with numbers first, and then natural keys."""
    return [[int(i) for i in re.findall(r"\d+", text)], natural_keys(text)]
