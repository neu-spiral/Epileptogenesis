# Epileptogenesis
Clone this repo, and copy the _\_data_ folder from [here](https://www.dropbox.com/sh/icfard16qqjpqrm/AAA3oMcZpGe1C0HZfqpflVIOa?dl=0) to the root directory, for all codes to work.
Find the necessary library versions to install in _requirements.in_.

# Next Actions
## Potential Pipeline improvements
Bagging/ augmentation/ under-over sampling to deal with imbalanced classes

# History
Simple Naive Bayes:
    Get best classifier from each modality, predict_proba, divide by p(l), arg max
Go naive bayesian based on 5/12 PGM

## Notes
Check on v3.0_0047.csv, need rest of subject ID. Can wait because there's no way it could be a seizure positive subject
3_17_0007 and 3_17_0012 both lack duration information. EEG v0 is therefore just totals of events. 

# [May 21]
Collect probability predictions and run fusion on these soft labels (feel free to modify and pass classifiers not predictions)
