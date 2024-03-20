# SymStair

Numerical experiments for the paper: ["Symmetric Stair Preconditioning of Linear Systems for Parallel Trajectory Optimization"](https://arxiv.org/abs/2309.06427)

**This package contains submodules make sure to run ```git submodule update --init --recursive```** after cloning!

## Usage and Hyperparameters:
To modify the experiments please see the file ```runTests.py``` which specifies cost functions and trajectory lengths. That file can be run to output results with ```python3 runTests.py```.

### Instalation Instructions
In order to support the experiments and submodules there are 4 required external packages ```beautifulsoup4, lxml, numpy, sympy``` which can be automatically installed by running:
```shell
pip3 install -r requirements.txt
```

### Citing
To cite this work in your research, please use the following bibtex:
```
@inproceedings{bu2024symmetric,
  title={Symmetric Stair Preconditioning of Linear Systems for Parallel Trajectory Optimization}, 
  author={Xueyi Bu and Brian Plancher},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  address = {Yokohama, Japan},
  month={May.},
  year = {2024}
}
```
