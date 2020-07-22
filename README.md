# Offline Contextual Bayesian Optimization

## Overview

In Bayesian Optimization (BO), many times there are several systems or "tasks"
to simultaneously optimize. This repository contains Multi-task Thompson
Sampling (MTS), a BO algorithm we developed to pick both tasks and actions
to evaluate. Because some tasks are usually more difficult than others, MTS
often significantly outperforms standard BO techniques. 

## Getting Set Up

The code is compatible with python 2.7. First, clone this repo and run
```
pip install -r requirements
```
By default the code leverages the [Dragonfly](https://github.com/dragonfly/dragonfly)
library. 

## Reproducing Synthetic Experiments

The plots in the paper can be reproduced by running [ocbo.py](src/ocbo.py)
and [cts_ocbo.py](src/cts_ocbo.py) with the appropriate options file.

```
cd src
mkdir data
python ocbo.py --options <path_to_option_file>
```
or if continuous
```
python cts_ocbo.py --options <path_to_option_file>
```
After the simulation has finished, the plots can be reproduced by
```
cd scripts
python discrete_plotter.py --write_dir ../data --run_id <options_name>
```
or
```
python cts_plotter.py --write_dir ../data --run_id <options_name>
```
For discrete experiments, use the flag `--risk_neutral 1` to show the risk
neutral performance instead and use `--plot_props 1` flag to show the
proportion of resources given to different tasks.

With the exception of the experiment in Section 4, the table below shows the
option file the corresponds to a given experiment.

| Experiment         | Option File                                           | 
| -------------      |:-------------:                                        |
| Figure 1(a,b)      | [set2d.txt](src/options/set2d.txt)                    |
| Figure 1(c)        | [rand4d.txt](src/options/rand4d.txt)                  |
| Figure 1(d)        | [rand6d.txt](src/options/rand6d.txt)                  |
| Figure 1(e)/4(a)   | [jointbran.txt](src/options/jointbran.txt)            |
| Figure 1(f)/4(b)   | [jointh22.txt](src/options/jointh22.txt)              |
| Figure 1(g)/4(c)   | [jointh31.txt](src/options/jointh31.txt)              |
| Figure 1(h)/4(d)   | [jointh42.txt](src/options/jointh42.txt)              |
| Figure 5(a)        | [contbran.txt](src/options/contbran.txt)              |
| Figure 5(b)        | [conth22.txt](src/options/conth22.txt)                |
| Figure 5(c)        | [conth31.txt](src/options/conth31.txt)                |
| Figure 5(d)        | [conth42.txt](src/options/conth42.txt)                |
| Figure 5(e)        | [contbran_sethps.txt](src/options/contbran_sethps.txt)|
| Figure 5(f)        | [conth22_sethps.txt](src/options/conth22_sethps.txt)  |
| Figure 5(g)        | [conth31_sethps.txt](src/options/conth31_sethps.txt)  |
| Figure 5(h)        | [conth42_sethps.txt](src/options/conth42_sethps.txt)  |

## Citing Work
If you use any code please cite the following:
```
@inproceedings{char2019offline,
  title={Offline contextual bayesian optimization},
  author={Char, Ian and Chung, Youngseog and Neiswanger, Willie and Kandasamy, Kirthevasan and Nelson, Andrew Oakleigh and Boyer, Mark and Kolemen, Egemen and Schneider, Jeff},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4627--4638},
  year={2019}
}
```
