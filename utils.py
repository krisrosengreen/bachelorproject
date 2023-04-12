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


