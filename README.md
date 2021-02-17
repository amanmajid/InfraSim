## Multi-Infrastructure Simulation Model (InfraSim)

<!---
<img align="right" width="500" src="https://github.com/amanmajid/InfraSim/blob/main/demo/schematic.png">
-->

This repository contains the InfraSim, a generalised arc-node simulation model for modelling water-wastewater-energy systems. It dynamically analyses flows of water, wastewater, and electricity across the network using the multi-commodity flow formulation. 

### Contributors
**Project Lead:** Aman Majid (aman.majid@new.ox.ac.uk) <br>
**Project Supervisor:** [Professor Jim Hall](https://www.eci.ox.ac.uk/people/jhall.html) <br>
**Contributors and Collaborators**: <br>
[Tony Downward](https://unidirectory.auckland.ac.nz/profile/a-downward), University of Auckland <br>

### What's Here
The repository contains the InfraSim source code. The model has been applied to a case-study from the Thames catchment, England, and will be expanded to other cases in future. These cases serve as a guide for other users to apply the InfraSim model to their areas of interest. An overview of each directory within the repository is shown below.

_demo/_
- A Jupyter Notebook explaining the model theory, as well as a small demo model of a water-wastewater-energy network in London, UK.
- Updated December 2020

_data/demo/_
- **spatial**: Shapefiles of node and edge data that can be opened in QGis.
- **csv**: Time series demo nodal flow data.

_infrasim/_
- InfraSim source code related to the Thames system can be found in thames.py
- There are a series of other Python files that contain code for data pre-processing and post-processing.
- Model metadata, parameters, and assumptions can also be found here.

_qgis/_
- A QGis project file to explore the network spatial data.

_outputs/_
- All model outputs such as figures, data, and statistics are saved here.


### Requirements
The model requires [Gurobi](https://www.gurobi.com) and the associated [GurobiPy](https://www.gurobi.com) library for the optimisation. In addition, standard scientific libraries in Python are needed such as [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/) etc. Requirements for spatial network analysis include [QGis](https://www.qgis.org/en/site/), [geopandas](https://geopandas.org/install.html), and [snkit](https://github.com/tomalrussell/snkit).

<i>Note</i>: The Gurobi package requires a license for usage but this can be obtained freely for academic use. An open-source alternative version of the model is currently being developed in the [PuLP](https://github.com/coin-or/pulp) library and the [Julia](https://julialang.org) programming language.  

### Getting started
Download and clone this repository.

Get a [Gurobi license](https://www.gurobi.com/downloads/)

Create project enviroment using the config file in this directory (only tested on macOS Big Sur):

    conda env create --prefix ./env --file config.yml
    conda activate ./env

See the [demo notebook](https://github.com/amanmajid/InfraSim/blob/main/demo/demo.ipynb) for a small demonstration.

### To Do
- Implement the InfraSim model using Julia code to allow users to choose their solver
- Apply InfraSim to a case-study of Israel, Palestine, and Jordan for a regional energy-water nexus analysis. The associated code will be uploaded here in future. 
- Apply InfraSim to a case-study of Jamaica's water-energy network. The associated code will be uploaded here in future. 

### Citing Research
Coming soon...


### Support
This work was partially supported by funding from the [NERC Doctoral Training Programme](https://www.environmental-research.ox.ac.uk/) at the University of Oxford, as well as internal funding from the Oxford Martin School Programme on [Transboundary Resource Management](https://www.oxfordmartin.ox.ac.uk/transboundary-resource-management/) at the University of Oxford.


### License
Copyright (C) 2020 Aman Majid. All versions released under the [MIT License](https://opensource.org/licenses/MIT).
