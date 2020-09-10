# HyperPref

I have removed everything but the bare necessities for HyperPref to run, so you will not be able to reproduce all results from the paper. If you use this code in any publication, please cite:

Hagg, A., Asteroth, A., Bäck, T.: A Deep Dive Into Exploring the Preference Hypervolume. International Conference on Computational Creativity, ICCC 2020.

Author: Alexander Hagg
Bonn-Rhein-Sieg University of Applied Sciences (HBRS)
email: info@haggdesign.de
Nov 2019; Last revision: 10-Sep-2020

# Demo

Run the demo.m script from the root directory of the repository. You can step through a (somewhat user unfriendly) version of HyperPref:

1. Train a VAE on a random shape set and run QD based on the VAE's 2D latent space
2. Select shapes (sorry, you will have to manually change the selection IDs, line 30: selectionIDs = [20, 25];)
3. Retrain the VAE on a set with perturbed variations of the selected shapes and run QD again to get the final result.