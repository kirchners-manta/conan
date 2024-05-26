General Information
===================

The trajectory analysis tool is automatically called, when a trajectory is loaded. Use the ``-f`` flag to define which trajectory to load.

.. code-block:: none

    $ CONAN -f trajectory.xyz

.. note::
    The trajectory has to either xyz, pdb or LAMMPS (.lammpstrj or .lmp) format.
    If the trajectory is in xyz format, the user is prompted to enter the simulation box dimensions, as they are needed for some analyses.
    In the case of the pdb and LAMMPS format, the box dimensions are read directly from the trajectory.

As a first step, the program identifies all rigid structures in the trajectory and characterizes them.
The identification of solid structures is achieved by comparing the fist two frames of a given trajectory and identify all frozen atoms.
The structures therefore have to stay frozen over the course of the simulation.
The trajectory analysis part is divided into two main sections, the picture mode and the analysis mode, which includes all the analysis functions implemented.
All modi will be discussed in further detail in the following sections.

For the analysis options implemented, the following parameters are potentially needed:

* element masses
* van der Waals [1]_ or covalent [2]_ radii of the elements
* number of increments (set by the user)

.. note::

        The user is prompted to choose between the van der Waals radii and covalent radii.
        For all analysis options, the listed atomic masses are used.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - element
     - mass [u]
     - vdW radius [Å]
     - covalent radius [Å]
   * - H
     - 1.008
     - 1.20
     - 0.31
   * - C
     - 12.011
     - 1.70
     - 0.76
   * - N
     - 14.007
     - 1.55
     - 0.71
   * - O
     - 15.999
     - 1.52
     - 0.66
   * - F
     - 18.998
     - 1.47
     - 0.57
   * - P
     - 30.974
     - 1.80
     - 1.07
   * - S
     - 32.065
     - 1.80
     - 1.05
   * - Li
     - 6.941
     - 1.81
     - 1.28
   * - Na
     - 22.990
     - 2.27
     - 1.66
   * - K
     - 39.098
     - 2.75
     - 2.03
   * - Mg
     - 24.305
     - 1.73
     - 1.41
   * - Ca
     - 40.078
     - 2.31
     - 1.76
   * - B
     - 10.811
     - 1.65
     - 0.84
   * - Ag
     - 107.868
     - 1.72
     - 1.45
   * - Au
     - 196.967
     - 1.66
     - 1.36
   * - D
     - 0.00
     - 0.00
     - 0.00
   * - X
     - 0.00
     - 0.00
     - 0.00

.. note::

        The masses and radii of Drude particles (D) and dummy atoms (X) are set to zero, to not interfere with the molecular recognition of the program.


.. [1] A. Bondi, van der Waals Volumes and Radii, J. Phys. Chem. 68 (3) (1964) 441-451.
       DOI: doi.org/10.1021/j100785a001
.. [2] B. Cordero, V. Gómez, A. Platero-Prats, M. Revés, J. Echeverría, E. Cremades, F. Barragán, S. Alvarez, Covalent radii revisited, Journal of the Chemical Society. Dalton Transactions (2008), 2832–2838
       DOI: doi.org/10.1039/b801115j
