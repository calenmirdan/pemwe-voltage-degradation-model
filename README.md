# pemwe-voltage-degradation-model

Author: Felix Dittmar (Fraunhofer Institute for Chemical Technology ICT)

This repository provides the code used for the voltage degradation model and
for the generation of synthetic data employed in the publication

"Development and validation of a layer-unspecific physical model for voltage
degradation to predict the remaining useful life of proton exchange membrane
water electrolyzers"

published in the Journal of Power Sources:
https://doi.org/10.1016/j.jpowsour.2025.239122

For a detailed description of the underlying assumptions, model formulation,
and parameter sources, please refer to the Journal of Power Sources article
and the corresponding Supplementary Material.


Repository contents
-------------------

synthetic_data_generation/
- __init__.py
- electrolyzer_model.py  
  Script for generating a synthetic dataset representing current density
  and voltage profiles of a PEM water electrolyzer plant.

- turbine_model.py  
  Script for generating a wind turbine power profile from measured wind data.

- data/
  - produkt_zehn_min_ff_20000101_20081031_02522.txt  
    Wind data obtained from the German Climate Data Center (CDC),
    “Historische 10-minütige Stationsmessungen der Windgeschwindigkeit
    in Deutschland”, Version v24.03.  
    Available at:
    https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/historical/  
    Accessed: 11 July 2025  
    License: Creative Commons Attribution 4.0 (CC BY 4.0)

  - synthetic_data_pemwe.csv  
    Synthetic dataset generated using electrolyzer_model.py, representing
    a four-stack PEM electrolyzer plant with a total capacity of 2.8 MW.

  - wind_data.csv  
    Wind turbine power profile generated using turbine_model.py.

  - wind_turbine_power_curve.csv  
    Power curve of a 3 MW wind turbine. The power curve differs from the one used
	in the Journal of Power Sources article in order to avoid copyright
	restrictions.

- functions/
  - __init__.py
  - electrolyzer_model_functions.py  
    Electrochemical model functions for computing PEM electrolyzer
    cell and stack voltages based on current density or power input.


voltage_degradation_model/
- __init__.py

- example_script_parameters.py  
  Example script demonstrating parameter identification and voltage
  model accuracy using the synthetic dataset.

- example_script_voltage_prediction.py  
  Example script demonstrating the full prognostics workflow, including
  degradation modeling and remaining useful life (RUL) prediction.

- functions/
  - __init__.py
  - voltage_degradation_model_functions.py  
    Core functions for data sectioning, polarization curve fitting,
    parameter regression, RUL estimation, and computation of PHM metrics.


config.py
Central configuration file controlling operating conditions, degradation
rates, noise levels, fitting models, end-of-life (EOL) definition, and PHM
metric settings.


Running the examples
--------------------
Install the required Python packages:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- joblib

Then run, for example:
- example_script_parameters.py
- example_script_voltage_prediction.py

The scripts print key results to the console and generate figures illustrating
the prognostic horizon and relative accuracy metrics.


Configuration
-------------
All relevant parameters are defined in config.py, including:
- operating conditions (temperature, pressures)
- degradation and noise parameters
- data sectioning settings
- regression model choices for R and i0
- end-of-life definition and PHM metric parameters

No code modifications are required to change model assumptions; adjusting
config.py is sufficient.


Citation
--------
This repository is archived on Zenodo and assigned a DOI.
Please cite both the Zenodo record and the associated Journal of Power Sources
article when using this code.


License
-------
The code is released under the BSD 3-Clause License as specified in the LICENSE file.


Contact
-------
For questions regarding the model or its application, please contact the
corresponding author Felix Dittmar or the Fraunhofer Institute for Chemical
Technology ICT.


Acknowledgements
----------------
The author acknowledges gratefully the financial support from the
German Federal Ministry of Education and Research (BMBF) through
the project ‘‘hyBit’’ under the grant 03SF0687E.
