Mean square displacement
--------------------------

Calculate the mean square displacements (MSD(:math:`\tau`)) of a given species in the system.
The user needs to define the timestep between consecutive frames and set the maximum correlation depth :math:`\tau_{max}` for the analysis.
Generally it is recommended to use a correlation depth of not larger than 1/3 of the total simulation time to assure enough sampling.
During the analsis, every frame is set as the reference frame at :math:`\tau_{0}` and the displacement is calculated for all frames until :math:`\tau_{max}`.
