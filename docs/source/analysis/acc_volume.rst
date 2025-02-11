Accessible volume
=================
The accessible volume is obtained by scanning the entire trajectory to identify the atom furthest from the centerline of the CNT, adding either the van der Waals radius or the covalent radius of the given element to the calculated distance.
The accessible volume :math:`V_{acc}` is then calculated by simplifying the CNT to be a cylinder, using the following equation:

.. math::

    V_{acc} = \pi*r_{acc}^2*l_{CNT}

where :math:`r_{acc}` is the radius of the atom furthest from the centerline of the CNT. :math:`l_{CNT}` is the length of a given CNT.

.. note::

    The accessible volume therefore depends on the most displaced atom and is not directly derived from the diameter of the CNT.
    Results may therefore vary slightly for different simulations of the same CNT, and a large sample size is highly recommended.

The calculated volumes can be used for further analysis, e.g. axial density profile calculation.
