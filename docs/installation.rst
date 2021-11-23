


Installation
===========================

.. toctree::
    :maxdepth: 2

    install_python
    

    

Install EDWIN (Simple user)
----------------------------

* Install from PyPi.org (official releases)::
  
    pip install ed_win

* Install from gitlab  (includes any recent updates)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/EDWIN.git
        


Install EDWIN (Developer)
--------------------------

We highly recommend developers install EDWIN into its own environment. (See
instructions above.) The commands to clone and install EDWIN with developer
options including dependencies required to run the tests into the current active 
environment in an Anaconda Prommpt are as follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/EDWIN.git
   cd EDWIN
   pip install -e .[test]
   


