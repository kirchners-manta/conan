Installation
============

To install CONAN, simply clone the public repository to your machine using GitHub.

.. code-block:: none

   git clone https://github.com/kirchners-manta/conan.git

The code supports Python versions 3.10 to 3.12. Multiple libraries need to be installed to run the code, which are listed in the `pyproject.toml` file.

Setting up an Environment
-------------------------

To ensure a clean and isolated installation, it is recommended to use an environment. You can use either `conda` or `venv` for this purpose.

Using conda (preferred)
^^^^^^^^^^^^^^^^^^^^^^^

- **Create a new conda environment**:

   .. code-block:: none

      conda create -n conan python=3.10
      conda activate conan

Using venv (Python's built-in virtual environment)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Create a new virtual environment**:

   .. code-block:: none

      python3 -m venv conan-env
      source conan-env/bin/activate

Installation
------------

Once the environment is set up, you can proceed with the installation:

.. code-block:: none

   pip install .

Usage
-----

Now the code is ready to be used. To start the program, simply run the following command:

.. code-block:: none

   CONAN

The program is structured in such a way that it works according to a question-answer scheme, where the user has to answer the respective questions.
A log file will be written called ``conan.log``, which contains everything that is printed to the terminal.
For certain modules, additional files might be created (e.g., a csv file for the results of the analysis).

Further Information
-------------------

For more details on setting up virtual environments, refer to the official documentation:

- `Conda Documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_

- `Python venv Documentation <https://docs.python.org/3/library/venv.html>`_
