# Stable Parameter Estimation for GLM-HMMs Applied to Neural Population Activity

This repository contains the code and processed data required to reproduce the analyses and figures presented in the manuscript (under review by PLOS Computation Biology):

> **Uncovering internal states with a robust shared-state multi-neuron GLM-HMM framework**
---

## Overview

This repository contains MATLAB code for fitting generalized linear model hidden Markov models (GLM-HMMs) to three electrophysiological datasets.

## Datasets

Medial prefrontal cortex dataset: 150628_Behaviour.dat, 150628_CellType.dat, 150628_Pos.dat, 150628_SpikeData.dat

> Adrien Peyrache, Mehdi Khamassi, Karim Benchenane, Sidney I Wiener, Francesco Battaglia (2018); Activity of neurons in rat medial prefrontal cortex during learning and sleep. CRCNS.org
http://dx.doi.org/10.6080/K0KH0KH5

Premotor cortex dataset: MM_S1_processed.mat

> Matthew G. Perich, Patrick N. Lawlor, Konrad P. Kording, Lee E. Miller (2018); Extracellular neural recordings from macaque primary and dorsal premotor motor cortex during a sequential reaching task. CRCNS.org.
http://dx.doi.org/10.6080/K0FT8J72


Insula dataset: insula.mat

Inside the dataset, RAW.Einfo provides the event information and RAW.Erast contains the event rasters. RAW.Nrast contains neuronal rasters.
This dataset was collected in our laboratory (Dr. Patricia Janak) and is provided in processed form to reproduce the analyses presented in the manuscript.

---

## Repository Structure

```
.
├── Main/
│   ├── HMM_GLM_Insula
│   ├── HMM_GLM_mPFC
│   ├── HMM_GLM_premotor
│
├── Data/
│   ├── Insula/
│   ├── mPFC/
│   └── Premotor cortex/
│
├── GLM-HMM supporting functions/
│
├── README.md
├── LICENSE

The main script for each dataset calls the supporting GLM-HMM functions and includes the statistical analyses and plots specific to that dataset.
Each brain dataset contains the .mat files for the fitted models used to create plots in the manuscript.
```

---

## Software Requirements

### MATLAB

The MATLAB analyses were performed using:

- MATLAB R2024b 

Required toolboxes:

- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox
  
Additional packages:

- glmnet for MATLAB

---

## License

This repository is distributed under the MIT License.

See `LICENSE` for details.

---

## Contact

For questions regarding this repository, please contact:

Aamna Lawrence

Johns Hopkins University

Email: alawre22@jh.edu
