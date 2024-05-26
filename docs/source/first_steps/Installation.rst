Installation
============

To install CONAN, simply clone the public repository to your machine using using GitHub. The program is located in the folder ``program``.

.. code-block:: none

   $ git clone --recursive https://github.com/kirchners-manta/conan.git
   $ cd conan/program

The code is written in Python 3.10. Multiple libraries need to be installed to run the code, which are listed in the requirements.txt file.
For the installation using pip, run the following command:

.. code-block:: none

   $ python3.10 -m pip install -r requirements.txt


For the installation in a new conda environment, run the following commands:

.. code-block:: none

   $ conda create -n conan python=3.10
   $ conda config --env --add channels conda-forge
   $ conda install -n conan --file requirements.txt
   $ conda activate conan

Now the code is ready to be used. To start the program, simply run the following command:

.. code-block:: none

   $ CONAN

The program is structured in such a way that it works according to a question-answer scheme, where the user has to answer the respective questions.
A log file will be written called ``conan.log``, which contains everything that is printed to the terminal.
For certain modules, additional files might be created (e.g. a csv file for the results of the analysis).
