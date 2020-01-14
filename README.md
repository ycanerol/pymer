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
```python
import sys
sys.path.append('<path_to_main_dir>/modules')
sys.path.append('<path_to_main_dir>/external_libs')
```

## Random Number Generator
The random number generator should be setup by running the setup.py file in
external_libs/randpy as described in the readme file there.

## User Configuration
The file `pymer_config_default.json` is meant to be read-only. Local configurations
may be set with a file `.pymer_config` in your home directory. These will
override the default configuration.

A local setting *root_experiment_dir* is necessary for most functions.

To set your local configuration, create an empty file named ".pymer_config" at
your home directory (`/home/<username>` for Linux and Mac and
`C:\Users\<username>` for Windows)
and add the desired options. See the default configuration file for all options.

A minimum working example to get you started would be:
```json
{
        "root_experiment_dir" : "/home/<username>/mydata/"
}
```
