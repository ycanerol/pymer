# pymer
Collection of **Py**thon scripts for analyzing extracellular **m**ulti**e**lectrode
array recordings from the **r**etina in the Gollisch lab, Göttingen.

# Installing

Install the latest version of [Anaconda](https://www.anaconda.com/products/individual#Downloads) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Creating conda environment

Open the terminal and navigate to the directory where you downloaded the repository.
```bash
conda env create --file environment.yml
```

This will create a new conda environment named pymer with the required packages. 

Each time you want to use this environment, you need to activate it:
```bash
conda activate pymer
```

## Adding path variables

A few folder paths need to be made available to python so that the scripts in those
folders are reachable. This can be done by a single import when the directory is set
to the pymer folder.

```python
import pathadder
```

You should edit the `pymer_path` variable in this script to match the location of the
folder where pymer is located.


The import needs to be done each time you launch the python interpreter.

If you'd like to make this more permanent, the following lines can be added to your `~/.bashrc` file

```bash
PYMERPATH='/home/user/repositories/pymer'
export PYTHONPATH="${PYTHONPATH}:$PYMERPATH"
export PYTHONPATH="${PYTHONPATH}:$PYMERPATH/modules"
export PYTHONPATH="${PYTHONPATH}:$PYMERPATH/external_libs"
export PYTHONPATH="${PYTHONPATH}:$PYMERPATH/classes"
export PYTHONPATH="${PYTHONPATH}:$PYMERPATH/generalizedmodels"
```

## Random Number Generator
To set up the random number generator, run the following in the terminal with the pymer environment activated:

```bash
sudo apt install gcc g++

cd external_libs/randpy

python setup.py build_ext --inplace
```


## User Configuration

A local setting *root_experiment_dir* is necessary for most functions.

This is a shortcut for all experiments, so that they are accessible from anywhere regardless of
your current working directory. Once you set the root_experiment_directory, you can access experiments
simply by their folder name instead of the full path.

To set your local configuration, create an empty file named ".pymer_config" at
your home directory (`/home/<username>/` for Linux and Mac and
`C:\Users\<username>\` for Windows)
and add the desired options. See the default configuration file for all options.

A minimum working example to get you started would be:
```json
{
        "root_experiment_dir" : "/home/<username>/mydata/"
}
```
