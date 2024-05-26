Density Profiles
================

Radial Denstity
---------------
Compute the radial density profile of the liquid inside the pore structure.
The pore is automatically identified in the trajectory and subdivided into the wall and the CNT part.
With the center line of the CNT set as the origin, the space around the center is cut into increments of the users' choice, up to the tubes' wall.
The increment volumes are subsequently calculated.
While scanning over the whole trajectory, every atom which is inside the pore is identified, weighed by either its element mass or partial charge and then sorted into the according increment given by the distance criterion.
After the scan, the obtained mass/charge of each increment is divided by the volume of the respective increment as well as the number of time steps investigated to obtain the density.
The obtained results can be plotted in multiple ways, set by the user, with the output saved as ``Radial_density_function.pdf``.
A contour plot of the density profile is also possible to generate, with the output saved as ``Radial_density_function_polar.pdf``.
The raw data is saved as ``Radial_density_function.csv``.
The analysis can be performed individually for all unique molecules and atoms.

Axial Density Profile
---------------------
Compute an axial density profile over the whole simulation box. The pore has to be oriented along the z axis to use this analysis.
Again, as for the radial density, the CNT and all carbon structures are identified automatically in the trajectory.
The volume of the CNT is either calculated with the accessible radius :math:`r_{acc}`, which is computed on the fly (see accessible volume) or with the radius :math:`r_{CNT}` of the CNT.
The radius :math:`r_{CNT}` is defined as the distance between a carbon atom of the CNT and the center line of the CNT.
The number of increments and therefore their volumes are set by the user.
As the volume of each increment in the bulk and in the pore differ, multiple regions are defined in the simulation box.
The number of increments (set by the user) subdivides the CNT and the bulk independently to assure that the increment volume calculations are done correctly.
The analysis is scanning the entire trajectory and sorting all atoms into a given increment, weighed by its element kind.
All obtained total increment masses are then divided by the number of frames and the respective increment volume to obtain the density.
The analysis can be performed individually for all liquid species present.
If needed, the results can be plotted in multiple ways with the output saved as ``Axial_density.pdf``.
The raw data is written to ``Axial_density.csv``.
