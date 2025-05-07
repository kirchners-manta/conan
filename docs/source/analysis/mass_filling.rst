CNT filling
-----------

This analysis calculates the filling of a CNT over the course of the trajectory.
The CNT can be flexible and move freely in the simulation box.
Please not that the program does take the movement into account, but not the crossing of the PBC boundaries.
For each frame, the program identifies all atoms that are located inside the CNT using this formalism:

.. math::

    (\vec{q} - \vec{p}_1) \cdot (\vec{p}_2 - \vec{p}_1) \geq 0

.. math::

    (\vec{q} - \vec{p}_2) \cdot (\vec{p}_2 - \vec{p}_1) \leq 0

.. math::

    \frac{|(\vec{q} - \vec{p}_1) \times (\vec{p}_2 - \vec{p}_1)|}{|\vec{p}_2 - \vec{p}_1|} \leq r


where :math:`\vec{q}` is the position of the atom, :math:`\vec{p}_1` and :math:`\vec{p}_2` are the two endpoints of the CNT, and :math:`r` is the radius of the CNT.

The program then calculates the total mass confined in the CNT by summing the masses of all atoms located inside the CNT.
The average mass filling is calculated by dividing the total mass obtained for all frames by the number of frames analyzed.
The average confined mass, the average mass per Angstrom, the mean distance between the two endpoints of the CNT, and the average radius of the CNT are also calculated and printed to the output file.
The results for each frame are saved in a CSV file, named ``frame_masses.csv``, together with the 5, 10 and 50 frame averages.
Another CSV file, named ``mass_per_angstrom.csv``, contains the average mass filling of the CNT per Angstrom.
The program also provides the distance between the two endpoints of the CNT and the CNT radii for each frame in seperate CSV files, named ``ring_ring_distances.csv`` and ``ring_radii.csv``, respectively.


.. note::
    To ensure that the program delivers correct results, it must be ensured that all atoms are wrapped in the simulation box and that the CNT does not cross the PBC boundaries.
    For pre-processing, please use e.g. Travis.
