
# Running the code:
1. Clone the repository `git clone https://github.com/h-mnzlr/acpl-particlesims.git`
2. Make sure you run the correct python version `python --version`. Output should be `Python 3.10.6`.
3. Create a virtual environment at e.g. `env/`: `python -m venv env`
4. Install all requirement: `pip install -r requirements.txt`
5. Dynamically link the local code packages into the environment `pip install -e .`
6. Spin up the Jupyter server using `jupyter notebook`
7. Run the code from the notebooks.

# Repo structure
##### `notebooks/`
Contains the notebooks that implement the exercises. Notebooks are called `.sync.ipynb` due to workflow reasons (`jupyter_ascending` Jupyter server plugin).

##### `code/`
Contains all the packages dynamically linked in the environment. Contains both the provided libraries and modules with self-implemented
helper functions to use in the notebooks.

`code/integrate.py`: Find here methods to create MC numbers and samples and functions to integrate a function using such samplers.

`code/constants.py`: Find here different constants and implementation of the scattering matrix $|M(s, \cos( \theta  ) , \phi)|^2$, extends `scipy.constants`.

##### `data/`
Contains the data for all the exercises named by exercise.

##### `report/`
Contains the report in addition to version send by email.

