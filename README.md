# Epileptogenesis
Clone this repo, and put the \_data folder (contact repo owners for access) in the root directory, for all codes to work.

# Next Actions
Simple Naive Bayes:
    Get best classifier from each modality, predict_proba, divide by p(l), arg max
## Potential Pipeline improvements

Bagging approach to deal with small number of subjects
Go naive bayesian based on 5/12 PGM

## Notes

Check on v3.0_0047.csv, need rest of subject ID. Can wait because there's no way it could be a seizure positive subject

3_17_0007 and 3_17_0012 both lack duration information. EEG v0 is therefore just totals of events. 

# Last Navid Commit [May 21]
Collect probability predictions and run fusion on these soft labels (feel free to modify and pass classifiers not predictions)

List all applicable DWI subjects
