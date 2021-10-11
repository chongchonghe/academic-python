# constSI.py or constCGS.py

## Usage 

```python
from constSI import *
```
or
```python
from constCGS import *
```
then use the constants, in SI or CGS units, respectively, as local variables, e.g. `G`. 

Alternatively, 
```python
import constSI
```
or 
```python
import constCGS
```
then use the constants as `constSI.G` or `constCGS.G`.

## Constants in constSI.py

The following constants are available as float point numbers, identical to the values in SI unit as listed in the 'Value' column:

```
========== ============== ================ =========================
   Name        Value            Unit       Description
========== ============== ================ =========================
    G        6.6743e-11     m3 / (kg s2)   Gravitational constant
   N_A     6.02214076e+23    1 / (mol)     Avogadro's number
    R        8.31446262     J / (K mol)    Gas constant
   Ryd       10973731.6       1 / (m)      Rydberg constant
    a0     5.29177211e-11        m         Bohr radius
  alpha    0.00729735257                   Fine-structure constant
   atm         101325            Pa        Standard atmosphere
  b_wien   0.00289777196        m K        Wien wavelength displacement law constant
    c        299792458        m / (s)      Speed of light in vacuum
    e      1.60217663e-19        C         Electron charge
   eps0    8.85418781e-12       F/m        Vacuum electric permittivity
    g0        9.80665          m / s2      Standard acceleration of gravity
    h      6.62607015e-34       J s        Planck constant
   hbar    1.05457182e-34       J s        Reduced Planck constant
   k_B      1.380649e-23      J / (K)      Boltzmann constant
   m_e     9.1093837e-31         kg        Electron mass
   m_n     1.6749275e-27         kg        Neutron mass
   m_p     1.67262192e-27        kg        Proton mass
   mu0     1.25663706e-06       N/A2       Vacuum magnetic permeability
   muB     9.27401008e-24       J/T        Bohr magneton
 sigma_T   6.65245873e-29        m2        Thomson scattering cross-section
 sigma_sb  5.67037442e-08   W / (K4 m2)    Stefan-Boltzmann constant
    u      1.66053907e-27        kg        Atomic mass
 GM_earth   3.986004e+14     m3 / (s2)     Nominal Earth mass parameter
  GM_jup   1.2668653e+17     m3 / (s2)     Nominal Jupiter mass parameter
  GM_sun   1.3271244e+20     m3 / (s2)     Nominal solar mass parameter
  L_bol0     3.0128e+28          W         Luminosity for absolute bolometric magnitude 0
  L_sun      3.828e+26           W         Nominal solar luminosity
 M_earth   5.97216787e+24        kg        Earth mass
  M_jup    1.8981246e+27         kg        Jupiter mass
  M_sun    1.98840987e+30        kg        Solar mass
 R_earth      6378100            m         Nominal Earth equatorial radius
  R_jup       71492000           m         Nominal Jupiter equatorial radius
  R_sun      695700000           m         Nominal solar radius
    au     1.49597871e+11        m         Astronomical Unit
   kpc     3.08567758e+19        m         Kiloparsec
    pc     3.08567758e+16        m         Parsec
========== ============== ================ =========================
```

## Constants in constCGS.py

The following constants are available as float point numbers, identical to the values in CGS unit as listed in the 'Value' column:

```
========== ============== ================ =========================
   Name        Value            Unit       Description
========== ============== ================ =========================
    G       6.674300e-08    cm3 / (g s2)   Gravitational constant
   N_A      6.022141e+23      1 / mol      Avogadro's number
    R       8.314463e+07   erg / (K mol)   Gas constant
   Ryd      1.097373e+05       1 / cm      Rydberg constant
    a0      5.291772e-09         cm        Bohr radius
  alpha     7.297353e-03                   Fine-structure constant
   atm      1.013250e+06       P / s       Standard atmosphere
  b_wien    2.897772e-01        cm K       Wien wavelength displacement law constant
    c       2.997925e+10       cm / s      Speed of light in vacuum
    g0      9.806650e+02      cm / s2      Standard acceleration of gravity
    h       6.626070e-27       erg s       Planck constant
   hbar     1.054572e-27       erg s       Reduced Planck constant
   k_B      1.380649e-16      erg / K      Boltzmann constant
   m_e      9.109384e-28         g         Electron mass
   m_n      1.674927e-24         g         Neutron mass
   m_p      1.672622e-24         g         Proton mass
 sigma_T    6.652459e-25        cm2        Thomson scattering cross-section
 sigma_sb   5.670374e-05    g / (K4 s3)    Stefan-Boltzmann constant
    u       1.660539e-24         g         Atomic mass
 GM_earth   3.986004e+20      cm3 / s2     Nominal Earth mass parameter
  GM_jup    1.266865e+23      cm3 / s2     Nominal Jupiter mass parameter
  GM_sun    1.327124e+26      cm3 / s2     Nominal solar mass parameter
  L_bol0    3.012800e+35      erg / s      Luminosity for absolute bolometric magnitude 0
  L_sun     3.828000e+33      erg / s      Nominal solar luminosity
 M_earth    5.972168e+27         g         Earth mass
  M_jup     1.898125e+30         g         Jupiter mass
  M_sun     1.988410e+33         g         Solar mass
 R_earth    6.378100e+08         cm        Nominal Earth equatorial radius
  R_jup     7.149200e+09         cm        Nominal Jupiter equatorial radius
  R_sun     6.957000e+10         cm        Nominal solar radius
    au      1.495979e+13         cm        Astronomical Unit
   kpc      3.085678e+21         cm        Kiloparsec
    pc      3.085678e+18         cm        Parsec
========== ============== ================ =========================
```
