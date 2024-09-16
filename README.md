# Generation of Synthetic Time Series: 
Exploring DoppelGANger for Motor Imagery EEG in the Temporal Domain 

This repository contains the code used the thesis project on synthesizing raw EEG using the GAN architecture DoppelGANger. 

Data is available through Google Drive: https://drive.google.com/drive/folders/1VKWJavXaOQPW4w97UxfpbnePjY2TwuKK?usp=sharing 

## TLDR 

This thesis covers the subject of utilizing generative adversarial network (GANs) for the problem of generating synthetic time series data, specifically electroenchalography (EEG) in the temporal domain. 

Synthetic EEG is generated for data augmentation in a left- and right-hand **motor imagery (MI)** classification task. 

For data generation, a GAN model called **DoppelGANger** (DG) proposed as a generalizable framework for multivariate time series is used (Lin et al., 2020). 

The **main interest** of the current work is to explore whether high-fidelity synthetic EEG can be generated from raw EEG data in the temporal domain using a general-purpose time series GAN such as DG.

## Credits

### Gretel.ai
This project uses software developed by Gretel.ai. The original software can be found at [Gretel.ai's GitHub repository](https://github.com/gretelai). Modifications were made for academic purposes as permitted under the terms of the Gretel.ai Source Available License.

### EEGMotorImagery
This project uses software developed by Roots et al. (2020) from the GitHub repository EEGMotorImagery available under the Apache License 2.0. The original source can be found at [EEGMotorImagery's GitHub repository](https://github.com/rootskar/EEGMotorImagery). Modifications were made for academic purposes.


