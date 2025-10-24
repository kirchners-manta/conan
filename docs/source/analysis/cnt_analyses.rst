CNT specific analyses
=====================

The analysis tolls presented here focus on the analysis of liquids confined inside carbon nanotubes (CNTs).
All analyses work for rigid and flexible CNTs.
Periodic CNTs, that cross the PBC in one direction can however be analyzed using these tools.
If there are restrictions regarding the system setup, they are mentioned in the respective sections.
All analyses also work if multiple CNTs are present in the simulation box and provide individual results for each CNT.

.. note::
    More information regarding these analysis tools is provided in the method section of this working paper on `ChemRxiv <https://doi.org/10.26434/chemrxiv-2025-3znm9>`_

.. _average-cnt-filling:

Average CNT filling
---------------------

This analysis calculates the filling of CNTs over the course of the trajectory.
The CNT can be flexible and move freely in the simulation box.
Please not that the program does take movement into account, but not the crossing of the PBC boundaries for CNTs of finite length.
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



.. _densdist-cnt:

Density distribution inside CNT
-------------------------------

Compute radial density distributions inside of CNTs over the course of the trajectory.
According to the scheme described in the previous section :ref:`average-cnt-filling`, the program identifies all atoms located inside of the CNT for each frame.
The center axis :math:`\vec{z}` of the CNT is calculated using the two endpoints of the CNT, :math:`\vec{p}_1` and :math:`\vec{p}_2`.
The radial space around the center axis is divided into user-defined bins.
For each atom located inside of the CNT, the distance to the CNT axis is calculated and assigned to the corresponding bin.
The density is calculated by summing the masses of all atoms located in a given bin over all frames and dividing by the volume of the bin and the number of frames analyzed.
The results are saved in CSV files named ``CNT_{NUMBER}_radial_density_function.csv``, where ``{NUMBER}`` is the index of the CNT in the simulation box.
Additional files containing the raw data are also generated, namely ``CNT_{NUMBER}_radial_density_raw.csv``and ``CNT_{NUMBER}_radial_mass_dist_raw.csv``.
Plots of the radial density distributions are generated, saved as ``CNT_{NUMBER}_radial_density_function.png``.

Angular distribution inside CNT
--------------------------------

Calculate the angular distributions of two pre-defined vectors over the course of the trajectory.
The user defines two vectors of choice, from which the angle is calculated.
The following vectors are available for selection:

1. center axis of the CNT :math:`\vec{z}` (calculated using the two endpoints of the CNT, :math:`\vec{p}_1` and :math:`\vec{p}_2`)

2. radial vector :math:`\vec{r}` from the center axis of the CNT to the atom of interest, or the center of mass position of a given molecule (:math:`\vec{r}` is defined as such, that it always is oriented perpendicular to :math:`\vec{z}`).

3. bond vector :math:`\vec{d}` between two atoms of choice within a given molecule.

4. bisector vector :math:`\vec{u}` (the normalized sum of two bond vectors, representing the angle bisector) defined by two bond vectors :math:`\vec{d_1}` and :math:`\vec{d_2}` within a given molecule.

5. vectors directly read from the trajectory file (e.g. dipole moment vectors).

The program calculates the angle between the two selected vectors for each frame in the trajectory and sorts the angles into user-defined bins, similar to :ref:`densdist-cnt`, based on the center of mass position of the molecule.
Results are stored in CSV files named ``CNT_{NUMBER}_angle_distribution.csv`` and ``CNT_{NUMBER}_raw_angles.csv``, where ``{NUMBER}`` is the index of the CNT in the simulation box.
Combined distribution plots of the angle and distance distributions are also generated, saved as ``CNT_{NUMBER}_angle_analysis.png``.


Layer residence times
-----------------------
Calculate the residence times of molecules in user-defined radial layers inside the CNT over the course of the trajectory.
The program first identifies all atoms located inside the CNT for each frame, according to the scheme described in :ref:`average-cnt-filling`.
The user defines radial layer through a distance criterium from the center axis :math:`\vec{z}` of the CNT.
Additionally, the user sets a transient tolerance time t*, up to which a molecule leaving a layer and re-entering is still considered to be continuosly residing in the layer.
The program then tracks each molecule located inside the CNT over the entire trajectory and records the time spent in each layer.
This is achieved through a survival correlation function.
For each time frame, each molecule is assigned to a given layer based on the center of mass position.
Over the simulation time, each molecule generates residence time events for each layer.
If a molecule leaves a layer and re-enters it within the transient tolerance time t*, the event is considered continuous.
The survival correlation function is then defined as:

.. math::

    C_{l,c}(t,t*) = N_{l,c}(t,t*) / D_{l,c}(t,t*)

where :math:`D_{l,c}(t,t*)` is the number of start times and :math:`N_{l,c}(t,t*)` is the number of events that survive until time t, accounting for the transient tolerance time t*.
This calculation is done for each CNT :math:`c` and each layer :math:`l`.
By definition, :math:`C_{l,c}(0,t*) = 1`.
The residence time :math:`\tau_{l,c}` is then obtained through a fit, assuming an exponential decay:

.. math::

    C_{l,c}(t,t*) = exp(-t / \tau_{l,c})

.. math::

    \tau_{l,c} = -t / ln(C_{l,c}(t,t*))

Results are saved in CSV files, ``CNT_{NUMBER}_combined_analysis.csv``, ``CNT_{NUMBER}_correlation_functions_raw.csv``, and ``CNT_{NUMBER}_layer_population.csv``, ``CNT_{NUMBER}_molecule_layer_trajectory.csv`` and ``CNT_{NUMBER}_residence_time_statistics.csv``.
Plots of the correlation function and layer population are generated, saved as ``CNT_{NUMBER}_correlation_functions.png`` and ``CNT_{NUMBER}_population_analysis.png``.
