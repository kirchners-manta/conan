Pore
====

Nanopores are built from two graphene sheets and a nanotube.

* ``bond_length``

the bond_length arg sets the length of the C-C bonds. The arg accepts a single floating point value.
example: ``bond_length=1.55``. The default is ``bond_length=1.42``.

* ``tube_length``

specifies the length of the tube in angstrom. The arg accepts a single floating point value.
example: ``tube_length=10.0`` for a 10 Angstrom long tube.

* ``tube_size``

specifies the width of the tube. The arg accepts a single integer ``m`` and creates a carbon nanotube with the
standard cnt nomenclature cnt(m,m).
example: ``tube_length=8``

* ``armchair/zigzag``

The conformation of the carbon nanotube can be controlled by either supplying "armchair" or "zigzag" as a keyword.

example build:

.. code-block:: none

   CONAN-build: build type=pore tube_size=8 tube_length=10.0 zigzag

will yield the following structure:

.. image:: ../../../pictures/basic_pore.png
   :width: 40%
   :alt: Carbon Nanotube
