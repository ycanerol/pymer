# pymer
Collection of **Py**thon scripts for analyzing extracellular **m**ulti**e**lectrode
array recordings from the **r**etina in the Gollisch lab, Göttingen.

# Installing
## Adding path variables
Folder paths to `modules/` and `external_libs/` should be added
to your python path variable, so that the imports can work.

One way would be adding a link to them in the site-packages:
```bash
echo "<path_to_main_dir>/modules" >> "<python_install_loc>/lib/python<your_version>/site-packages/modules.pth"
echo "<path_to_main_dir>/external_libs" >> "<python_install_loc>/lib/python<your_version>/site-packages/external_libs.pth"
```

OR
Add the following to the ~/.pythonrc
```bash
import sys
sys.path.append('<path_to_main_dir>/modules')
sys.path.append('<path_to_main_dir>/external_libs')
```

## Random Number Generator
The random number generator should be setup by running the setup.py file in
external_libs/randpy as described in the readme file there.

## User Configuration
The file `defaultconfig.json` is meant to be read-only. Local configurations may be set with a file `.pymer` in your home directory. These will override the default configuration.

Copy desired settings from `defaultconfig.json` into `~/.pymer` and adjust them to your preferences. A local setting *root_experiment_dir* is necessary for most functions.
