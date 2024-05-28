Accessible Volume
=================
The accessible volume is obtained by scanning the whole trajectory to identify the atom which is furthest displaced from the center line of the CNT, while adding either the van der Waals radius or the covalent radius of the given element to the calculated distance.
The accessible volume :math:`V_{acc}` is then calculated by simplifying the CNT to be a cylinder, using the following equation:

.. math::

    V_{acc} = \pi*r_{acc}^2*l_{CNT}

where :math:`r_{acc}` is the radius of the atom which is furthest displaced from the center line of the CNT. :math:`l_{CNT}` is the length of a given CNT.

.. note::

    The accessible volume is therefore depending on the most displaced atom and not directly derived by the CNTs diameter.
    Results can thus slightly differ for varying simulations with the same CNT, and a large sampling is recommended to yield the best results.

The calculated volumes can be used for further analysis e.g. the axial density profile calculation.
