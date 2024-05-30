add
===

the add command is used to place functional groups at a specific position. Functional groups are
taken from the .xyz files in /current_version/structures/ and selected with the ``group`` argument. ``group=OH`` will search for
a file called "OH.xyz" in the structures folder and add its contents to the sheet. The position can be specified using the ``position``,
which takes in the index of the atom the group should be placed on. When using VMD the indices can be shown using the ``vmd show_index`` command in conan.
The command tries to take into account the local topology of the structure, so placements on curved surfaces (like the inside of a Pore) are possible.
