Picture mode
============
The picture mode is able to generate seperate xyz files, extracted from the trajectory.

Simulation Box Picture
----------------------
Saves the xyz structure of the whole simulation box.

Pore Picture
------------
Save the pore structure in the simulation box in a seperate xyz file.
Additionally, the center point of the CNT can be added to the xyz file, labeled as 'X'.

CNT Picture
-----------
Save the xyz structure of just the CNT (without the outer walls of the pore).
Additionally, it is possible to add all atoms to the xyz file, which are contained inside the CNT.
As some molecules might be partially inside the CNT and partially outside, some molecules might be fragmented.
Another option is the addition of complete molecules to the xyz file which are either completely or partially contained inside the CNT.
