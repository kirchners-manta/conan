.. figure:: pictures/CONAN_logo.png
   :width: 90%
   :class: align-center

.. note::

   This project is under active development.


CONAN - User Guide
===================================

**CONAN** is a tool to generate carbon and boron nitride structures, set up MD simulation boxes, generate xyz structures from a provided trajectory and analyze MD trajectories composed of a liquid in confinement.
The program has the following features:

* Generate structures such as carbon walls, carbon nanotubes (CNT), pore structures, and their boron nitride analogues. All carbon structures can be doped with graphitic nitrogen.
* Set up simulation boxes with a bulk liquid input file and carbon structures of choice.
* Identify and characterize solid structures found in an MD trajectory.
* Calculate the radial density within a CNT/pore, either weighted by mass or by partial charge.
* Calculate the accessible volume of a CNT/pore.
* Calculate the axial density along a simulation box.
* Cut a pore/CNT from the trajectory, either filled with liquid or empty.

In addition, the program includes a molecular identification tool, which makes it possible to perform all analyses for the individual types of molecules present in the system or even individual atoms.

Check out :doc:`first_steps/Installation` for more information on how to install CONAN.

.. toctree::
   :caption: First Steps
   :maxdepth: 2

   first_steps/Installation
   first_steps/input_output

.. toctree::
   :caption: Simulation Setup
   :maxdepth: 2

   simulation_setup/builder
   simulation_setup/simulation_box

.. toctree::
   :caption: Trajectory analysis
   :maxdepth: 2

   analysis/analysis
   analysis/picture_mode
   analysis/mol_ident
   analysis/acc_volume
   analysis/dens_prof

.. toctree::
   :caption: Other
   :maxdepth: 2

   other/versions

.. toctree::
    :caption: API
    :maxdepth: 2

    api
