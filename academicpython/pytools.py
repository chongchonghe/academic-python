"""
General purpose tools
"""
import sys

# def to_latex(f, n=2):
#     assert type(n) == int
#     fmt = "{{:.{}g}}".format(n)
#     float_str = fmt.format(f)
#     if "e" in float_str:
#         base, exponent = float_str.split("e")
#         return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
#     else:
#         return float_str

def to_latex(f, n=2):
    assert type(n) == int
    fmt = "{{:.{}g}}".format(n)
    float_str = fmt.format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base == '1':
            return r"10^{{{}}}".format(int(exponent))
        else:
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def getarg(arg, value=None):
    """Parse command line inputs and return True or False.

    Examples:

        if getarg("-d"):
            print("debug mode")
        >>> ./main.py -d # would run in debug mode
        >>> ./main.py # would run in non-debug mode

        fmt = "PNG"
        if getarg("-o", "PDF"):
            fmt = "PDF"
        >>> ./main.py -o PDF # fmpt = "PDF"
        >>> ./main.py or ./main.py -o PNG # fmpt = "PNG"
    """

    if len(sys.argv) <= 2:
        return False
    if arg in sys.argv[1:]:
        if value is None:
            return True
        else:
            idx = sys.argv.index(arg)
            try:
                return sys.argv[idx + 1] == value
            except IndexError:
                return False
    return False

