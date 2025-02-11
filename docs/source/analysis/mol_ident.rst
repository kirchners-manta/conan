Molecule identification
===================

The program includes a molecule identifier that detects all bonds between atoms based on distance criteria.
The cutoff distances, which determine whether two atoms are bonded, vary depending on the elements involved.
These element combinations and their corresponding cutoff distances are detailed in a library.
The program analyzes the first frame of the trajectory to identify all molecules, meaning it does not account for bond breaking or formation throughout the trajectory.
Periodic boundary conditions are taken into consideration during this process.
Once molecules are recognized, the program outputs the number of unique molecules and their respective quantities within the system.
Additionally, it generates images of the identified molecules (for those with fewer than 50 atoms) complete with atom labels.
