# How to set up Marabou

Several gotchas in this process which are covered below. Got it working on Ubuntu.

1) Set up conda/miniconda, and start a new environment (without this, the build fails - I assume this is because conda provides certain packages by default).
2) Set Python version in the conda env to 3.8.5: `conda install python=3.8.5`
3) Install other conda packages: `conda install tensorflow numpy`
4) Follow instructions here to build: https://neuralnetworkverification.github.io/Marabou/Setup/0_Installation.html. Note these are slightly different to the instructions on the main GitHub repo (includes `-DBUILD_PYTHON=ON` flag).
5) Navigate to 'Marabou/maraboupy' folder. You'll see a file with '.so' extension, for me this was 'MarabouCore.cpython-38-x86_64-linux-gnu.so'. Copy this file so it's just 'MarabouCore.so' : `cp MarabouCore.cpython-38-x86_64-linux-gnu.so MarabouCore.so` (this seems to allow Python to access the C++ code in 'MarabouCore.cpp')
6) Copy the 'fc1.pb' file from 'Marabou/resources/tf/frozen_graph'
7) Now, the code here should run in a Jupyter notebook https://neuralnetworkverification.github.io/Marabou/Examples/1_TensorflowExample.html, except changing the first block to:
```
import sys
import numpy as np
import tensorflow
sys.path.append('/home/pidge/Masters/Marabou')
from maraboupy import Marabou, MarabouNetworkTF
```

Note I also had an issue where error was `ValueError: numpy.ufunc has the wrong size, try recompiling. Expected 192, got 216` - I never got to the bottom of why this came up, but uninstalling and reinstalling tensorflow and numpy resolved it.

