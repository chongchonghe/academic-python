# -*- coding: utf-8 -*-
"""const.py

Importing physical/astrophysical constants from astropy.constants
@author: Chong-Chong He, che1234@umd.edu
"""

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
consts_only_SI = [
    "e",
    "eps0",
    "mu0",
    "muB",
]
consts = consts_all
for _c in consts:
    locals()[_c] = eval("C.{}.cgs.value".format(_c))
print("Imported the following physical constants from ChongChong He's"
      " python package constCGS.py:")
print(consts)
