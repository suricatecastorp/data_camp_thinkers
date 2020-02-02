# RAMP starting kit on Humanity Thinkers Network

_Project Data Camp 2019 2020_

_Authors: Augustin ESPINOSA, Pierre FONTAINE, Clément LIU, Paul ROUSSEL, Runtian ZHANG_

## Introduction

In this project, we are interested in the thinkers who have influenced the history of mankind. We know that a thinker is inspired by some of his predecessors and will also influence his successors. The aim of this project is to determine the existence of an influence of thought between two people.

## Description of the data set

The dataset will consist of three databases:
1. A dataset that contains information about each thinker with the following fields (obtained via Wikipedia in English):
   * `id`: arbitrary but unique integer identifier 
   * `name`
   * `date_birth` - `date_death`: dates of birth and death (if known)
   * `place_birth` - `place_death`: places of birth and death (if known)
   * `summary`: Abstract extracted from [Wikipedia](https://en.wikipedia.org/wiki/Main_Page)
2. A training file `training_set.csv` that lists pairs of thinker identifiers and a boolean (0 or 1) value corresponding to the exercise of influence or not.
3. A test file `testing_set.csv` with only pairs of thinkers whose influence must be determined or not

## Set up

Open a terminal and

1. Install the `ramp-workflow` library (if not already done):

 ```$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git```

2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

## Local notebook

Get started on this RAMP with the [notebook](https://github.com/suricatecastorp/data_camp_thinkers/blob/master/thinkers_starting_kit.ipynb).

To test the starting-kit, run:

 ```$ ramp_test_submission --quick-test```
