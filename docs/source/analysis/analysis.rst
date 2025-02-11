General Information
===================

The trajectory analysis tool is automatically called when a trajectory is loaded. Use the ``-f`` flag to specify which trajectory to load.

.. code-block:: none

    $ CONAN -f <trajectoryfile> ...

.. note::
    The trajectory must be in either xyz, pdb or LAMMPS (.lammpstrj or .lmp) format.
    If the trajectory is in xyz format, the user will be prompted to enter the simulation box dimensions, as they are required for some analyses.
    In the case of the pdb and LAMMPS formats, the box dimensions are read directly from the trajectory.

As a first step, the program identifies and characterizes all rigid structures in the trajectory.
The identification of rigid structures is done by comparing the first two frames of a given trajectory and identifying all frozen atoms.
The structures must therefore remain frozen throughout the simulation.
Alternatively, the user can define a structure manually.
The trajectory is then analyses in awaytheuser can choose from a variety of analysis options.

The following parameters may be required for the implemented analysis options

* Element Masses
* Van der Waals [1]_ or covalent [2]_ radii of the elements
* Number of increments (user defined)
.. note::

        For all analysis options, the listed atomic masses are used. If the user wants to use different masses, they must be added to defdict.py.

.. note::

        The user is prompted to choose between the van der Waals radii and covalent radii.

.. note::

        The masses and radii of Drude particles (D) and dummy atoms (X) are set to zero.

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



.. [1] A. Bondi, van der Waals Volumes and Radii, J. Phys. Chem. 68 (3) (1964) 441-451.
       DOI: doi.org/10.1021/j100785a001
.. [2] B. Cordero, V. Gómez, A. Platero-Prats, M. Revés, J. Echeverría, E. Cremades, F. Barragán, S. Alvarez, Covalent radii revisited, Journal of the Chemical Society. Dalton Transactions (2008), 2832–2838
       DOI: doi.org/10.1039/b801115j
