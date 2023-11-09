# Cell Cycle Mapping Package

## Step 1: Install Environment

From the root directory of this repository:

```
conda env create -f .\environments\cc_mapping.yml
```

## Step 2: Update Global Variables

Due to the fact this is not an actual package, whenever you want to use it, you will have to tell your computer where to look. You will need to update these two files:

* cc_mapping\GLOBAL_VARIABLES\GLOBAL_VARIABLES.py
* notebooks\GLOBAL_VARIABLES\GLOBAL_VARIABLES.py

Replace the variable 'cc_mapping_package_dir' with the path to the root directory for the cc_mapping repository.

This means that if you want to use the cc_mapping package in another folder, you should copy this GLOBAL VARIABLES folder into that directory and add this to the imports of your python scripts

```
import sys
sys.path.append(os.getcwd())

from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir
sys.path.append(cc_mapping_package_dir)
```

## Step 3: Use the cc_mapping package!

There is a test notebook in the notebooks directory of this repository that can be used as an example.