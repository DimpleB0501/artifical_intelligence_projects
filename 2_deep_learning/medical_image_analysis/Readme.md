# Installation
## Installing with yml file
### Installing yml file 
```
conda env create -f pytorch_course_env_ubuntu.yml
```

### Activating the enviornment
conda activate pytorchenv

### Deactivate enviornment
conda deactivate

## Installing without yml file
Instead of using the environment file, you can also manually create the environment and subsequently install the necessary libraries.

Before doing so, please download **requirements.txt** from the attached link

To do so you at first have to create a conda environment:

```
conda create -n pytorchenv python=3.8.0
```

Then proceed with y -> Enter
```
conda activate pytorchenv
```
```
pip install -r requirements.txt
```
