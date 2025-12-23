# Benchmarking

## Benchmarks Description
Here we provide the benchmarks showcased in CHAMMI-75. We have 6 benchmarks:

#### [CHAMMI](./CHAMMI/README.md) (Chen et al., 2023)
CHAMMI is a channel adaptive benchmarking suite. It has 3 sub-datasets that contain 3, 4 and 5 channels. These are respectively tasks of cell cycle classificaion, protein localization, and compound matching. A weighted average of the out of distribution tasks generates the CHAMMI score. For example, if we call the score of HPA task 2 as HPA2, an equation to calculate the CHAMMI score is: $\frac{WTC2}{3}+\frac{(HPA2+HPA3)}{6}+\frac{(CP2 + CP3 + CP4)}{9}$.

#### [CellPHIE](./CellPHIE/README.md) (Kang et al., 2025)
CellPHIE is a pooled genetic perturbation screen containing 14-channel images of single cells perturbed with one
of 19 genes. In CellPHIE, models seek to identify these pertubations over control examples. 

#### [HPAv23](./HPAv23/README.md) (Ouyang et al., 2019; Le et al., 2022)
HPAv23 is a protein localization task. In this task, models must solve a 19 or 31 class classificaiton task to identify where the proteins are localizing. 

#### [JUMPCP](./JUMPCP/README.md) (Chandrasekaran et al., 2023b)
JUMPCP is a compound screening benchmark. It contains 5 channel images, with the tasks of identifying pertubed compounds and of grouping compounds of biologically similar pertubations are clustered together.

#### [IDR0017](./IDR0017/README.md) (Breinig et al., 2015) 
IDR0017 is a chemical-genetic interaction study which has the models trying to identify hits by ranking gene-compound combinations that are likely to have a large effect with respect to controls. Ground truth hits were obtained
from the original study, and performance is evaluated using recall at the top 50 and 100.

#### [RBC-MC](./RBC-MC/README.md) (Doan et al., 2020)
RBC-MC is bright-field study of red blood cells imaged with flow cytometry. In this task, models bucket cells into seven clinically relevant morphological categories
associated with blood quality. 

## Usage
In this directory, users can configure the benchmark_config.yaml in order to run the benchmarks. After configuration, running `python evaluate.py` will run the benchmarks. An environment is specified in the `pixi.toml` file in this directory, and can be downloaded and initialized with `pixi shell` after pixi is installed. See the yaml file for a description of all of the fields and for help setting up the benchmark.

Alternatively, with the given environment, changing into the given benchmark directory allows a user to run a given benchmark with the commands provided in each benchmark's README. 

## Implementing your own model

We support, through a common interface, the models provided in the original CHAMMI-75 benchmarking results. To add your own model, follow these steps:

1. Add a model class into `model.py`, following the `Model` interface's methods. These are wrappers of the actual torch models, and are in charge of handling the model's forward pass within a common interface. Model's are required to take in `Batch, Channel, Width, Height` shaped images, and output embeddings in the shape of `Batch, Embed`. 

2. Add a basic call to creating your `Model` class in `get_model`, at the bottom of that file. This is the function that is called to load models in the various benchmarks, based on the provided configuration file.

3. In benchmark_config.yaml, provide your `model_type` at the top of the file, following the name you gave it in `get_model`. 

