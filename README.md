# SimpleLatticeMC
A bare bones monte carlo code for simple lattice models

Usage:

1. Fill/modify vasp_input in accordance with example
directory for bilayer Zn intercalated TaS2. The code
currently only allows for intercalants in quasi-Oh
holes in (MM) aligned 2H-TaS2. Modification to the
make_supercell.py and setup_dirs.sh can change this.

2. Go into scells directory and run ./setup_dirs.sh to
create directories for vasp submission at desired
occupations. Modify end of setup_dirs.sh to make
different supercells.

3. Run these jobs to get energies, dos, etc.

4. Parse and fit the energies to a lattice model with
TBDSCRIPT.

5. Run the lattice model, use the extracted parameters
in the mc.jl file in the montecarlo directory.
Modify MC parameters as needed.

