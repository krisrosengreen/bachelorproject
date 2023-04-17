import re
FORMATTING_DECIMALS = 4


def inputfile_row_format(row, row_num) -> str:
    """
    Format row list to a string, the same way Quantum Espresso does it.

    Parameters
    ----------
    row : list
        point to be formatted

    row_num : int
        The row-number this row corresponds to
    """
    FD = FORMATTING_DECIMALS
    return f"   {row[0]:.{FD}f} {row[1]:.{FD}f} {row[2]:.{FD}f} {row_num: >3}\n"


def get_values(text) -> list:
    """
    Given a text return a list of values contained within

    Parameters
    ----------
    text : str
        Str fromwhich values are found
    """
    text_values = re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', text)
    values = list(map(float, text_values))
    return values
