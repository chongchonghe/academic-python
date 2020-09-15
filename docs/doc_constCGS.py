from astropy import constants as C

consts_all = [
    "G",
    "N_A",
    "R",
    "Ryd",
    "a0",
    "alpha",
    "atm",
    "b_wien",
    "c",
    "g0",
    "h",
    "hbar",
    "k_B",
    "m_e",
    "m_n",
    "m_p",
    "sigma_T",
    "sigma_sb",
    "u",
    "GM_earth",
    "GM_jup",
    "GM_sun",
    "L_bol0",
    "L_sun",
    "M_earth",
    "M_jup",
    "M_sun",
    "R_earth",
    "R_jup",
    "R_sun",
    "au",
    "kpc",
    "pc",
]

### print __doc__ in CGS unit
print(
"""
========== ============== ================ =========================
   Name        Value            Unit       Description
========== ============== ================ =========================""")
for _cons in consts_all:
    _c = eval("C." + _cons)
    print("{:^10s} {:^14e} {:^16s} {:s}".format(_c._abbrev, _c.cgs.value, _c.cgs.unit, _c.name))
print("========== ============== ================ =========================")
