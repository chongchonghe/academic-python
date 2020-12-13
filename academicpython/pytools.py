"""
General purpose tools
"""

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
