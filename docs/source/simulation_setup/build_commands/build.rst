build
=====


The build command is used to build the initial structure. The type of generated structure is controlled using the ``type`` argument.
Currently available structures:

* graphene  (creates a 2D-graphene sheet)
* boronnitride (creates a 2D-boronnitride sheet)
* cnt (creates a carbon nanotube)
* pore (creates a pore)

depending on which structure is created, different arguments and commands for further customization are available.

.. toctree::
   :caption: structures
   :maxdepth: 2

   structures/Graphene_sheet.rst
   structures/Boronnitride.rst
   structures/Carbon_nanotube.rst
   structures/Pore.rst
