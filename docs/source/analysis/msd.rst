Mean square displacement
--------------------------

Calculate the mean square displacements (MSD(:math:`\tau`)) of a given species in the system.
The user needs to define the timestep between consecutive frames and set the maximum correlation depth :math:`\tau_{max}` for the analysis.
Generally, it is recommended to use a correlation depth of not larger than 1/3 of the total simulation time to assure enough sampling.
During the analysis, every frame is set as the reference frame at :math:`\tau_{0}` and the displacement is calculated for all frames until :math:`\tau_{max}`.
The MSD is calculated normally, as well as independently for the x, y, and z components for each species studied.
The raw data is saved in a CSV file, named ``msd_<species>.csv``, and plotted in a separate figure ``msd_<species>.png``.
The MSD is calculated as:

.. math::
    \begin{aligned}
    MSD(\tau) &= \frac{1}{N} \sum_{i=1}^{N} \left( \mathbf{r}_i(\tau) - \mathbf{r}_i(0) \right)^2 \\
              &= \frac{1}{N} \sum_{i=1}^{N} \left( (x_i(\tau) - x_i(0))^2 + (y_i(\tau) - y_i(0))^2 + (z_i(\tau) - z_i(0))^2 \right) \\
              &= MSD_x(\tau) + MSD_y(\tau) + MSD_z(\tau)
    \end{aligned}

where :math:`N` is the number of atoms in the system, :math:`\mathbf{r}_i(\tau)` is the position of atom :math:`i` at time :math:`\tau`, and :math:`\mathbf{r}_i(0)` is the position of atom :math:`i` at time :math:`0`.
The MSD is calculated for all liquid species in the system.

.. note::
    Make sure the trajectory is not wrapped before the analysis.
    Any potential movement of the center of mass of the simulation box is removed/accounted for during the analysis.
