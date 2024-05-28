Input & Output
==============

The program can also be executed using an input file by specifying the ``-i`` flag and the input file in the command line.

.. code-block:: none

    $ CONAN -i input.log

The input file is to be structured in such a way that each question question, which is prompted by the program, is written down in a line.
The program reads all entries which are the same line subsequent to the question as the given answer.
This makes it possible to use the conan.log file of an previously conducted analysis as input file for a further analysis.


All output structures and data files are either directly saved in the current directory where the program is run, or a new folder is created in the current working directory.
If there are files or folders in the working directory, which are identically named as the ouput files, the old files are either renamed or overwritten.
Additionally a ``conan.log`` file is written in the current directory, which contains all the output of the program written to the terminal. The output file can be used as an input file for a further analysis.
