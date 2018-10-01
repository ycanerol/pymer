# pymer
Collection of **Py**thon scripts for analyzing extracellular **m**ulti**e**lectrode
array recordings from the **r**etina in the Gollisch lab, Göttingen.

# Installing
## Adding path variables
Folder paths to `modules/` and `external_libs/` should be added
to your python path variable, so that the imports can work.

One way would be adding a link to them in the site-packages:
```
echo "<path_to_main_dir>/modules" > "<python_install_loc>/lib/python<your_version>/site-packages/modules.pth"
echo "<path_to_main_dir>/external_libs" > "<python_install_loc>/lib/python<your_version>/site-packages/external_libs.pth"
```

OR
Add the following to the ~/.pythonrc
```
import sys
sys.path.append('<path_to_main_dir>/modules')
sys.path.append('<path_to_main_dir>/external_libs')
```

## Random number generator
The random number generator should be setup by running the setup.py file in
external_libs/randpy as described in the readme file there.

