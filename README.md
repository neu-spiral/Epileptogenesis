# Unimodal and Multimodal Classification of Epileptic Seizures

## Project Summary
Please take a look at our papers below:
1. [Multimodal (dMRI, EEG, fMRI: 2022)](https://www.medrxiv.org/content/10.1101/2022.10.22.22281402.abstract)
2. [Unimodal (dMRI: 2021)](https://link.springer.com/chapter/10.1007/978-3-030-87615-9_12)
3. [Unimodal (EEG: 2021)](https://ieeexplore.ieee.org/abstract/document/9630242/)

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
Clone this repo, and copy the _\_data_ folder from [here](https://www.dropbox.com/sh/icfard16qqjpqrm/AAA3oMcZpGe1C0HZfqpflVIOa?dl=0) to the root directory [as shown in the file tree above], for all codes to work.

The code runs from terminal using ```main.py```, with supporting functions automatically parsed from ```models.py```, ```helper.py```, and open-sourced functions from the folder ```extra```.

Plots for results can be generated using ```plot_csv.py```

Some residual code snippets and inline results+visualization can be found in ```multimodal_RA.ipynb```

The raw source files can be found in _/SDrive/CSL/\_Archive/2019/DT\_LONI\_Epileptogenesis\_2019_
