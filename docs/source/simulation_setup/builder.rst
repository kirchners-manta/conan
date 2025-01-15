Build module
=================

The mode is used to build carbon and boron nitride structures from scratch and print the xyz structures to a file.

The program can create the following structures:

* Carbon nanotubes (CNTs) and boron nitride nanotubes of arbitrary length and radius in zigzag and armchair conformations.
* Walls of arbitrary size in x and y directions. Multiple layers can be constructed in ABA stacking.
* The CNTs and carbon walls can be combined to form a 'pore' structure. Either as an open tube with walls on both sides, or with the tube closed off on one side, and a wall on the other side.
* All structures are designed to satisfy periodic boundary conditions wherever possible.
* Structures can be functionalized using user-defined groups

This mode is started with the ``-c`` flag.

.. code-block:: none

    $ CONAN -c

Once the mode is started, a console opens that takes in commands to build the structures:

.. code-block:: none

   ###########################################
   ##                                       ##
   ##   #####  #####  #   #  #####  #   #   ##
   ##   #      #   #  ##  #  #   #  ##  #   ##
   ##   #      #   #  # # #  #####  # # #   ##
   ##   #      #   #  #  ##  #   #  #  ##   ##
   ##   #####  #####  #   #  #   #  #   #   ##
   ##                                       ##
   ###########################################

   Find the documentation on the CONAN website: http://con-an.readthedocs.io
   If you use CONAN in your research, please cite the following paper:
   doi.org/10.1021/acs.jcim.3c01075

   CONAN-build:

It is recommended to use VMD(Visual Molecular Dynamics) in conjunction
with the the build-tool. If VMD is installed it can be started using the ``vmd`` command (in CONAN). Make sure that no instance of VMD is running
when CONAN is started.

.. code-block:: none

   CONAN-build: vmd

once VMD is started, any changes to the structures are shown in the VMD window. If VMD is unavailable, the structures are
saved in ``structures/structure.xyz``. All available commands and their usage are listed here:

.. toctree::
   :caption: commands
   :maxdepth: 2

   build_commands/build.rst
   build_commands/functionalize.rst
   build_commands/defects.rst
   build_commands/add.rst
   build_commands/undo.rst
   build_commands/stack.rst

.. image:: ../pictures/wall.png
   :width: 40%
   :alt: Wall

.. image:: ../pictures/wall_N.png
   :width: 40%
   :alt: Wall_N

.. image:: ../pictures/wall_side.png
   :width: 10%
   :alt: wall_side

.. image:: ../pictures/CNT.png
   :width: 30%
   :alt: CNT

.. image:: ../pictures/pore_front.png
   :width: 30%
   :alt: Pore

.. image:: ../pictures/pore_side.png
   :width: 30%
   :alt: Pore_closed

Structures are built by using the following default parameters:

* Carbon-carbon distance is set to 1.42 Angstroms.
* Interlayer distance between carbon layers is set to 3.35 Angstroms.
  The layers are shifted with respect to each other, resulting in an ABA stacking.
* Boron-nitride distance is 1.44 Angstroms.
* Interlayer distance between boron nitride layers is set to 3.33 Angstroms.
  The positions of all nitrogen and boron atoms are switched in successive layers, resulting in an ABA stacking.

All set values can be freely adjusted by the user.
All relevant values are printed to the terminal and to the ``conan.log`` file. The xyz structures are written to separate files in a ``structures`` folder.
