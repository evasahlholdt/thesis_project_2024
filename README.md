## <p align="center"><b>Generation of Synthetic Time Series:</b><br> **Exploring DoppelGANger for Motor Imagery EEG in the Temporal Domain** </p>

This repository contains the code used the thesis project on synthesizing raw EEG using the GAN architecture DoppelGANger. 

Data is available through Google Drive: https://drive.google.com/drive/folders/1VKWJavXaOQPW4w97UxfpbnePjY2TwuKK?usp=sharing 

### TLDR 

This thesis covers the subject of utilizing generative adversarial network (GANs) for the problem of generating synthetic time series data, specifically electroenchalography (EEG) in the temporal domain. 

Synthetic EEG is generated for data augmentation in a left- and right-hand motor imagery (MI) classification task. 

For data generation, a GAN model called DoppelGANger (DG) proposed as a generalizable framework for multivariate time series is used (Lin et al., 2020). 

The main interest of the current work is to explore whether high-fidelity synthetic EEG can be generated from raw EEG data in the temporal domain using a general-purpose time series GAN such as DG.

### Project structure

NB: Data folders are empty and must be populated with data from corresponding folders in Google Drive. 

```
├── data_csv                               # csv file prepared in 1_load_prep_data.ipynb
│   └── ...                               
├── data_real                              # Numpy arrays prepared in 2_create_datasets.ipynb
│   ├── ...
│   └── ...    
├── data_synthetic                         
│   ├── baseline                           # Numpy arrays created in 3_generate_baseline.ipynb
│   │   ├── ...
│   │   └── ...
│   ├── synthetic                          # Numpy arrays created in 5_synthesization.ipynb
│   │   ├── ...
│   │   └── ...
├── model_classification
│   ├── __init__.py
│   ├── loader.py
│   ├── model.py
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── run_classification.py
│   └── utils.py
├── model_doppelganger
│   ├── __init__.py
│   ├── config.py
│   ├── dgan.py
│   ├── errors.py
│   ├── evaluate.py
│   ├── requirements.txt
│   ├── run_synthesization.py
│   ├── structures.py
│   ├── torch_modules.py
│   └── transformations.py
├── pipeline
│   ├── 1_load_prep_data.ipynb
│   ├── 2_create_datasets.ipynb
│   ├── 3_generate_baseline.ipynb
│   ├── 4_optimization.ipynb
│   ├── 5_synthesization.ipynb
│   ├── 6_evaluation.ipynb
│   ├── 7_classification.ipynb
│   ├── 8_evaluate_classification.ipynb
│   └── 9_visualize.ipynb
├── results
│   ├── results_classification          # Pickle files created in 7_classification.ipynb
│   │   ├── ... 
│   │   └── ...
│   ├── results_evaluation              # JSON files created in 6_evaluation.ipynb
│   │   ├── ...
│   │   └── ...
│   ├── results_optimization            # JSON file created in 4_optimization.ipynb
│   │   └── ...                         
│   └── results_training_info           # JSON files created in 5_synthesization.ipynb
│       ├── ...
│       └── ... 
└── scripts
    ├── __init__.py
    ├── baseline_data_generation.py
    ├── evaluation.py
    ├── metrics.py
    ├── optimization.py
    ├── optimization_eval.py
    ├── plots.py
    ├── run_synthesization.py
    └── util.py
```


### Credits

#### Gretel.ai
This project uses software developed by Gretel.ai. The original software can be found at [Gretel.ai's GitHub repository](https://github.com/gretelai). Modifications were made for academic purposes as permitted under the terms of the Gretel.ai Source Available License.

#### EEGMotorImagery
This project uses software developed by Roots et al. (2020) from the GitHub repository EEGMotorImagery available under the Apache License 2.0. The original source can be found at [EEGMotorImagery's GitHub repository](https://github.com/rootskar/EEGMotorImagery). Modifications were made for academic purposes.


