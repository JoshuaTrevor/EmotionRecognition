# Outline
This project is intended to transform a dataset of facial landmarks into an equivalent representation which instead uses polynomials to represent various curves. In doing so we hope to study the effectiveness of compressing facial expression data for the purpose of optimising machine learning applications.

# Draft User Guide
## Prerequisites
As long as python3, numpy, pandas and matplotlib are all installed there should be no issues.

## Command line arguments
### python3 src/run.py (no parameters)
Draws 3 random images using polynomial data

### python3 src/run.py create
Creates a polynomial version of scaled_dataset.csv as scaled_poly.csv

### python3 src/run.py train \<number of iterations\>
Attempts to train a decision tree classifier on a random split n times, then dumps the aggregate results in results.txt and the individual results in log_file.txt
  
### python3 src/run.py dist
Shows information about the dataset's emotion distribution

