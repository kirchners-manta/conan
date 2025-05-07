Specific analyses
=================

.. note::
    The analyes tools presented here are just valid for specific simulation setups.
    All analyses options assume a rigid CNT with a liquid inside.
    In all cases, bar the minimal/maximal distance analysis, the CNT must be oriented along the z-axis.
    In case of the axial density analysis, a bulk phase on each of the two sides of the CNT is also assumed.
    The user should be aware of the limitations of the tools and the requirements for the simulation setup.

Accessible volume
-----------------

The accessible volume is obtained by scanning the entire trajectory to identify the atom furthest from the centerline of the CNT,
adding either the van der Waals radius or the covalent radius of the given element to the calculated distance.
The accessible volume :math:`V_{acc}` is then calculated by simplifying the CNT to be a cylinder, using the following equation:

.. math::

    V_{acc} = \pi*r_{acc}^2*l_{CNT}

where :math:`r_{acc}` is the radius of the atom furthest from the centerline of the CNT. :math:`l_{CNT}` is the length of a given CNT.

.. note::

    The accessible volume therefore depends on the most displaced atom and is not directly derived from the diameter of the CNT.
    Results may therefore vary slightly for different simulations of the same CNT, and a large sample size is highly recommended.

The calculated volumes can be used for further analysis, e.g. axial density profile calculation.


Radial charge/mass density
--------------------------
Calculate the radial density profile of the liquid inside the pore structure.
The pore is automatically identified in the trajectory and subdivided into the wall and the CNT part.
Using the centerline of the CNT as the origin, the space around the center is sliced into increments of user choice up to the wall of the tube.
The increment volumes are then calculated.
While scanning over the entire trajectory, each atom inside the pore is identified, weighed by either its elemental mass or partial charge, and then sorted into the appropriate increment given by the distance criterion.
After the scan, the obtained mass/charge of each increment is divided by the volume of the respective increment and the number of time steps examined to obtain the density.
The obtained results can be plotted in several user-defined ways, with the output saved as ``Radial_density_function.pdf``.
A contour plot of the density profile can also be generated, with the output saved as ``Radial_density_function_polar.pdf``.
The raw data is saved as ``Radial_density_function.csv``.
The analysis can be performed individually for all unique molecules and atoms.

Axial density
-------------
Compute an axial density profile over the entire simulation box. The pore must be oriented along the z-axis to use this analysis.
Again, as with the radial density, the CNT and all carbon structures in the trajectory are automatically identified.
The volume of the CNT is calculated either with the accessible radius :math:`r_{acc}`, which is calculated on the fly (see Accessible Volume), or with the radius :math:`r_{CNT}` of the CNT.
The radius :math:`r_{CNT}` is defined as the distance between a carbon atom of the CNT and the center line of the CNT.
The number of increments and thus their volumes are defined by the user.
Since the volume of each increment is different in the bulk and in the pore, multiple regions are defined in the simulation box.
The number of increments (set by the user) subdivides the CNT and the bulk independently to ensure that the increment volume calculations are performed correctly.
The analysis scans the entire trajectory and sorts all atoms into a given increment, weighted by their element type.
All obtained total increment masses are then divided by the number of frames and the respective increment volume to obtain the density.
The analysis can be performed individually for all liquid species present.
If desired, the results can be plotted in multiple ways, with the output saved as ``Axial_density.pdf``.
The raw data is written to ``Axial_density.csv``.

Atomic radial velocity
----------------------
Calculate the radial velocity of a given liquid or specific atom type inside the CNT.
The analysis is performed in a similar way to the radial density profile.
As with the radial density, the results can be plotted in several user-defined ways, with the output saved as ``Radial_velocity_profile.pdf``.
The raw data is saved as ``Radial_velocity_profile.csv``.


.. note::

    The the radial velocity is calculated atom by atom, and not for the center of mass of a given molecule.
    It does also not distinguish between translational motion and rattling of the atoms.
    The analysis is therefore not suitable for calculating of the diffusion coefficient.


Maximal/minimum distance
------------------------
Calculate the maximal and minimum distance between a given atom in the system investigated and the structure within it.
The analysis constructs a kd-tree with all the atoms of the identified structure atoms in the first frame.
The analysis assumes that the structure atoms are frozen in position.
Then all atoms of liquid molecules in the system are scanned throughout the trajectory and the distances to the structure atoms is calculated.
The minimum and maximum displacements are stored and written to the output.
