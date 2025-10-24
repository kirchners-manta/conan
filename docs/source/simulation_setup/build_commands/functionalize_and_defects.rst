functionalize
=============

.. note::
   The functionalize command is only available if the program was built using the development version of CONAN.
   To do this, please reinstall CONAN with the following command. For more information please visit the GitHub page of CONAN.

.. code-block:: none

      pip install -e '.[dev]'

The functionalize command is used to place a specified number of functional groups on the structure.
Functional groups are taken from the .xyz files in ``/current_version/build_modules/structure_lib/`` and selected with the ``group`` argument.
``group=OH`` will search for a file named "OH.xyz" in the ``structure_lib/`` directory and add its contents to the sheet.
The number of groups can be set with the ``group_count`` argument. The groups will be placed randomly on the sheet.
If the added groups should have a certain minimum distance from each other, an exclusion radius can be set with the ``exclusion_radius`` argument.


Example build:

.. code-block:: none

   CONAN-build: build type=graphene sheet_size=20.0 20.0
   CONAN-build: functionalize group=OH group_count=20.0 exclusion_radius=2.0

will result in the following structure:

.. image:: ../../pictures/functionalized_graphene.png
   :width: 40%
   :alt: functionalized graphene

Defects
=======

The ``defects`` command currently only creates holes in graphene or boron nitride sheets.
For graphene the created pores are circular while for boron nitride the pores are triangular.
The ``pore_size`` argument specifies the distance between the center of the pore and the edge in angstroms.

Example builds:

.. code-block:: none

   CONAN-build: build type=graphene
   CONAN-build: defects pore_size=4.3

will result in the following structure:

.. image:: ../../pictures/porous_graphene.png
   :width: 40%
   :alt: functionalized graphene


.. code-block:: none

   CONAN-build: build type=boronnitride
   CONAN-build: defects pore_size=1.0

will result in the following structure:

.. image:: ../../pictures/porous_boronnitride.png
   :width: 40%
   :alt: functionalized boronnitride
