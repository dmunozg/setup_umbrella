# setupUmbrella 

Initially a tool to automate the generation of windows
for a Umbrella Sampling calculation employing GROMACS. Now a
collection of tools I've written over the years to ease the setup of
each calculation

I recommend the use of a virtual environment (e.g.: Conda) with the
dependencies of these tools.

# Tools

## mktopol

Requires [pandas](https://anaconda.org/conda-forge/pandas) and [numpy](https://anaconda.org/conda-forge/numpy) to be installed.

Generates the `[molecules]` data required in the topology file from a `.gro` file