Density profiles
================

Radial density
---------------
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
---------------------
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
