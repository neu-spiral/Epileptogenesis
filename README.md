# Unimodal and Multimodal Classification of Epileptic Seizures

## Project Summary
Raw Data Samples: 

<img src=https://user-images.githubusercontent.com/38365057/213297916-258904c4-6b5f-4fa4-bb6b-a790cd61447a.png width="400">

Classification Pipeline: 

<img src=https://user-images.githubusercontent.com/38365057/213297928-0540da3e-4a37-460b-aebe-04bc29553773.png width="400">

## Prerequisites
Please install all necessary library versions by typing in terminal:

```pip install -r requirements.txt```

## File Structure
```
|--<_data>
|--<code> [multimodal]
     |--main.py
     |--helper.py
     |--plot_csv.py
     |--models.py
     |--multimodal_RA.ipynb
     |--extra
```

## Usage
Clone this repo, and copy the _\_data_ folder from [here](https://www.dropbox.com/sh/icfard16qqjpqrm/AAA3oMcZpGe1C0HZfqpflVIOa?dl=0) to the root directory seen in file tree above.

The code runs from terminal using ```main.py```, with supporting functions automatically parsed from ```models.py```, ```helper.py```, and open-sourced functions from the folder ```extra```.

Plots for results can be generated using ```plot_csv.py```

Some residual code snippets and inline results+visualization can be found in ```multimodal_RA.ipynb```

The raw source files can be found in _/SDrive/CSL/\_Archive/2019/DT\_LONI\_Epileptogenesis\_2019_

Two execution samples for ```main.py```:

1) Run Naive Bayesian Fusion with AdaBoost:

```python main.py --model NBF --text _adb_fs```

2) Run IDSF with CCA (7 components) followed by RECC (rho=0.7) on SFS (vary features between 1~10) with SVM classifier and ROC plots:

```python main.py --model CCA+SFS --roc_flag True --fixed_feat 7 --options roc_data --rho 0.7 --text _f_d_svm_feats```

## Publications
Please take a look at our papers below for details:
1. [Multimodal (dMRI, EEG, fMRI: 2022)](https://www.medrxiv.org/content/10.1101/2022.10.22.22281402.abstract)

Cite: 
```
@article{akbar2022post,
  title={Post Traumatic Seizure Classification with Missing Data using Multimodal Machine Learning on dMRI, EEG, and fMRI},
  author={Akbar, Md Navid and Ruf, Sebastian F and Singh, Ashutosh and Faghihpirayesh, Razieh and Garner, Rachael and Bennett, Alexis and Alba, Celina and Imbiriba, Tales and La Rocca, Marianna and Erdogmus, Deniz and others},
  journal={medRxiv},
  year={2022}
}
```

2. [Unimodal (dMRI: 2021)](https://link.springer.com/chapter/10.1007/978-3-030-87615-9_12)

Cite: 
```
@inproceedings{akbar2021lesion,
  title={Lesion Normalization and Supervised Learning in Post-traumatic Seizure Classification with Diffusion MRI},
  author={Akbar, Md Navid and Ruf, Sebastian and Rocca, Marianna La and Garner, Rachael and Barisano, Giuseppe and Cua, Ruskin and Vespa, Paul and Erdo{\u{g}}mu{\c{s}}, Deniz and Duncan, Dominique},
  booktitle={International Workshop on Computational Diffusion MRI},
  pages={133--143},
  year={2021},
  organization={Springer}
}
```

3. [Unimodal (EEG: 2021)](https://ieeexplore.ieee.org/abstract/document/9630242/)

Cite: 
```
@inproceedings{faghihpirayesh2021automatic,
  title={Automatic Detection of EEG Epileptiform Abnormalities in Traumatic Brain Injury using Deep Learning},
  author={Faghihpirayesh, Razieh and Ruf, Sebastian and La Rocca, Marianna and Garner, Rachael and Vespa, Paul and Erdo{\u{g}}mu{\c{s}}, Deniz and Duncan, Dominique},
  booktitle={2021 43rd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={302--305},
  year={2021},
  organization={IEEE}
}
```
