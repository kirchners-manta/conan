Molecule Identifier
===================
A molecule identifier is included in the program.
All bonds between atoms are identified through distance criteria.
Thereby the cutoff distances, below which two atoms are considered bonded, depend on the elements.
All element combinations and the according cutoff distances are listed in a library.
The program deduces all molecules from the first frame in the trajectory and therefore cannot handle bond breaking or formation in the trajectory.
Periodic boundary conditions are taken into account.
After molecule recognition, the program prints the number of unique molecules and how many of each there are in the system.
It also generates images of the identified molecules (if the number of atoms is less than 50) with atom labels.
