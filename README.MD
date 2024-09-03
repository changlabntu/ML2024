### Excerpted and modified from Python Machine Learning (Sebastian Raschka, 2019)

## https://www.anaconda.com/products/distribution

##  Exercise 0: Installing Python packages

---

Python is available for all three major operating systems — Microsoft Windows, macOS, and Linux — and the installer, as well as the documentation, can be downloaded from the official Python website: https://www.python.org.
Python version `>= 3.7.0` is recommend.

You can check your current default version of Python by executing

    $ python -V


### Pip

Python utilize different functionalities from additional packages. These packages can be installed via the `pip` installer program. It is recommended to find the exact command of installing a specific python package by search the name package name in https://pypi.org, For example, the command to install pandas can be found at https://pypi.org/project/pandas/:

    pip install pandas

You can usually find the right command by googling "pip" + "PACKAGE NAME"


### Anaconda

A HIGHLY recommended alternative Python distribution for scientific computing
is Anaconda by Continuum Analytics. Anaconda comes with many pre-installed packages for scientific computing. The Anaconda installer can be downloaded at https://docs.anaconda.com/anaconda/install/, and an Anaconda quick start-guide is available at https://docs.anaconda.com/anaconda/user-guide/getting-started/.

But most importantly, installing Anaconda allow you to manage your python environment by Conda, an environment management package for many languages. It is good to pick up if you are want to be serious about data science in the future.

After successfully installing Anaconda, you can usually find the right command by googling "conda" + "PACKAGE NAME". For example, the command to install matplotlib in conda is

    conda install -c conda-forge matplotlib


### Core packages

- [NumPy](http://www.numpy.org) >= 1.17.4, for many numerical operations
  - pip install numpy
- [SciPy](http://www.scipy.org) >= 1.3.1, for scientific computing 
  - pip install scipy
- [scikit-learn](http://scikit-learn.org/stable/) >= 0.22.0, for machine learning
  - pip install scikit-learn
- [matplotlib](http://matplotlib.org) >= 3.1.0, for plotting and visualization
  - pip install matplotlib
- [pandas](http://pandas.pydata.org) >= 0.25.3, for handling data
  - pip install pandas

### Check if packaged are installed correctly

If you install the packages directly on system environment (sometimes called "base" environment), you can directly find your python package by first type

    python

to enter python interface. I will see the command line prompt changes from %, like

    (base) USER@MACHINE ~ %

to python prompt

    >>> 

then you can type

    >>> import numpy
    >>> numpy
    <module 'numpy' from '/Users/garychang/opt/anaconda3/envs/p38pl16/lib/python3.8/site-packages/numpy/__init__.py'>

to confirm that the package has been installed

HOWEVER, it is highly recommended to use conda to create a virtual environment to handle your python packages (so you won't mess up your computer). Once you have conda installed (comes with anaconda), you can create an environment by something like:

    conda create python=3.8 --name CourseML

you can then activate your virtual environment by 

    conda activate CourseML

where "CourseML" is the name of the virtual environment that we just created. You will see the command line prompt changes from

    (BASE) USER@MACHINE ~ % to (CourseML) USER@MACHINE ~ %

Now all the changes you make to the environments will only happen to the virtual environment (So you can start over when if mess up)!


## Google Colab

Colaboratory, or "Colab" for short, is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud.

The most powerful feature of Google Colab is the ability to use cloud GPUs for free! Follow the instruction at:

https://colab.research.google.com/drive/1-n1w_oj7gIDyfBF7GQsmu-kHE4t_c-je?usp=sharing


## Python/Jupyter Notebook

If you want to run notebook in your local machine, the Jupyter notebook can be installed as usually via pip.

    $ pip install jupyter notebook

Alternatively, we can use the Conda installer if we have Anaconda or Miniconda installed:

    $ conda install jupyter notebook

To open a Jupyter notebook, we `cd` to the directory that contains your code examples, e.g,.

    $ cd ~/Exercise1/python_basics.ipynb

and launch `jupyter notebook` by executing

    $ jupyter notebook

Jupyter will start in our default browser (typically running at [http://localhost:8888/](http://localhost:8888/)). Now, we can simply select the notebook you wish to open from the Jupyter menu.

For more information about the Jupyter notebook, I recommend the [Jupyter Beginner Guide](http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/what_is_jupyter.html) and [Jupyter Notebook Basics](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html).


    
