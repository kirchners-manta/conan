Installation
============

To install CONAN, simply clone the public repository to your machine using GitHub.

.. code-block:: none

   git clone https://github.com/kirchners-manta/conan.git

The code supports Python 3.10 to 3.12. To run the code, several libraries need to be installed, which are listed in the ``pyproject.toml`` file.

Setting up an environment
-------------------------

To ensure a clean and isolated installation, it is recommended to use an environment. You can use either ``conda`` or ``venv``.

Using conda (preferred)
^^^^^^^^^^^^^^^^^^^^^^^

- Create a new conda environment:

   .. code-block:: none

      conda create -n conan python=3.10
      conda activate conan

Using ``venv`` (Python's built-in virtual environment)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create a new virtual environment:

   .. code-block:: none

      python3 -m venv conan-env
      source conan-env/bin/activate

Once the environment is set up, you can proceed with the installation:

.. code-block:: none

   pip install .

Usage
-----

Now the code is ready to use. To start the program, just run one of the the following commands:\
To analyse a trajectory file:

.. code-block:: none

   CONAN -f <trajectoryfile>

To construct structures:

.. code-block:: none

   CONAN -c

To get help:

.. code-block:: none

   CONAN -h

The program is structured in such a way that it works according to a question-answer scheme, where the user has to answer the respective questions.
A log file called ``conan.log`` is written, containing everything that is printed to the terminal.
For certain modules, additional files may be created (e.g. a ``csv`` file for the results of the analysis).

For further information
-----------------------

For more details on setting up virtual environments, see the official documentation:

- `Conda Documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_

- `Python venv documentation <https://docs.python.org/3/library/venv.html>`_


Input & Output
==============
It is possible to automate the running of the program by specifying the input on the command line using the ``-i`` flag.

.. code-block:: none

    CONAN -f <trajectoryfile> -i <input_file>


The automation allows using the ``conan.log`` output file from a previous analysis as the input for another.
The input file must list each program question on a new line, with answers on the same line.
All output files are saved in the current directory or a new folder within it.
Existing files with the same name are renamed or overwritten.
The ``conan.log`` file, containing all terminal output, can be used for further analysis.
