# TorchSRH2

WSI representation learning framework

[**MLiNS Lab**](https://mlins.org) /
[**TorchSRH**](https://github.com/MLNeurosurg/torchsrh) /
[**OpenSRH**](https://github.com/MLNeurosurg/opensrh)

## Installation

1. Clone OpenSRH github repo
    ```console
    git clone git@github.com:MLNeurosurg/torchsrh2.git
    ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment:
    ```console
    conda create -n ts2 python=3.10
    ```
4. Activate conda environment:
    ```console
    conda activate ts2
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/repo/dir>
    pip install -e .
    ```
6. Install rapids ai / cuML: see instruction [here](https://rapids.ai/start.html).
    These modules are used for generating tsne plots using gpu
    ```console
    conda install -c rapidsai -c nvidia -c conda-forge \
        -c defaults rapids=0.14 python=3.7 cudatoolkit=10.2
    ```

# Version control and source code formatting
`master` is a long-running branch. All work should be done in branches derived
from `master`, and will be rebased onto `master` once they are completed.
Spaces are used for indentation, and snake\_case is used to name variables
and functions.
