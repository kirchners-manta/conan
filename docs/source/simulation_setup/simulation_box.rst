Simulation Box Generator
========================

A simulation box can be set up using the ``-b`` flag in the command line.

.. code-block:: none

    $ CONAN -b

With the simulation box mode activated, the user will be prompted to enter a string of letters to describe how the box should be set up.

.. list-table:: Simulation Box Generator
   :widths: 25 25 50
   :header-rows: 1

   * - Letter
     - Description
     - File name
   * - B
     - Liquid bulk
     - bulk.xyz
   * - W
     - Wall structure
     - wall.xyz
   * - P
     - Pore structure
     - pore.xyz
   * - L
     - Left pore structure
     - pore_left.xyz
   * - R
     - Right pore structure
     - pore_right.xyz

The 'L' and 'R' keywords are needed, if two different pore structures shall be included in the simulation box.
The provided xyz structures have to be in the folder where the program is executed, or in the ``structures`` folder generated beforehand.
The final simulation box will be saved as simbox.xyz in the present folder.


.. figure:: ../pictures/simulation_box.png
   :width: 90%
   :class: align-center

   Simulation box of an ionic liquid between carbon walls and a pore, built with the simulation box generator.
