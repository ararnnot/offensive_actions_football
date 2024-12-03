
Name of the PhD: Lipschitz extensions in Machine Learning models: Applications to data-driven predictive football

Author: Andres Roger Arnau Notari (ararnnot@posgrado.upv.es, rogeran98@gmail.com)
Date: November 2024

---

## Overview

This project is a Python implementation of an off-line Reinforcement Learning for evaluating football players decisions.
We divide the offensive actions into shots, passes and carrys and the possible situation into states.
Using the data, we find the transition probability function P(s' | s, a).
With this, we compute the quality of each action in a particular state Q(s,a) and the value of each state V(s).

---

## Requirements

The code requires Python version 3.8 and the following libraries:
List of Libraries: [numpy, pandas, scikit-learn, matplotlib, torch, tqdm]

---

## Data

All event csv files must be inside the folder data/all_matches

---

## Run python

Order of running:
    1. import_data.py
    2. compute_probs.py
    3. compute_Q.py
Later, visualizations:
    4. video_match.py
    -. simulate_match.py
    -. show_images.py

 

