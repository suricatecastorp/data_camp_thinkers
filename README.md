# RAMP starting kit on Humanity Thinkers Network

_Project Data Camp 2019 2020_

_Authors: Augustin ESPINOSA, Pierre FONTAINE, Clément LIU, Paul ROUSSEL, Runtian ZHANG_

## Introduction

Whether it is in the field of science, of math, of philosophy or of litterature, an important part of creation is drawing inspiration from others in our field. It is not only natural but essential to creation to be open to what is being made around us. It is thus without suprise that a great number of works rely on the pieces of our contemporaries and of our forefathers. 

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

## Goal

Your goal is to predict, given a pair of thinkers if an influence exists between the two.

## Set up

Open a terminal and

1. Install the `ramp-workflow` library (if not already done):

 ```$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git```

2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

## Local notebook

Get started on this RAMP with the [notebook](https://github.com/suricatecastorp/data_camp_thinkers/blob/master/thinkers_starting_kit.ipynb).

To test the starting-kit, run:

 ```$ ramp_test_submission --quick-test```
