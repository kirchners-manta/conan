Snapshots
============
Create separate xyz files extracted from the trajectory.

Simulation box image
----------------------
Saves the xyz structure of the entire simulation box.

Pore image
------------
Saves the pore structure of the simulation box in a separate xyz file.
Additionally, the center point of the CNT can be added to the xyz file, marked as 'X'.

CNT image
-----------
Save the xyz structure of only the CNT (without the outer walls of the pore).
It is also possible to add to the xyz file all the atoms that are inside the CNT.
Since some molecules may be partially inside and partially outside the CNT, some molecules may be fragmented.
Another option is to add complete molecules to the xyz file that are either completely or partially inside the CNT.
