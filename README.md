# dhc_maintenance
# Predictive Maintenance of Pre-Insulated Bounded Pipes in District Heating

This project aims to provide a predictive maintenance solution for the pre-insulated bounded pipes used in district heating systems. The solution is implemented using a Python model.

## Overview
Pre-insulated bounded pipes are an important component of district heating systems, as they are used to transport hot water from the central heating plant to individual buildings. Regular maintenance is necessary to ensure the optimal performance and longevity of these pipes. However, traditional maintenance methods are reactive and can result in unexpected downtime and repair costs.

To address this issue, this project provides a predictive maintenance solution based on machine learning algorithms. The model analyzes various parameters such as temperature, pressure, and flow rate, to predict the condition of the pipes and proactively schedule maintenance when necessary.

## Requirements
The following libraries and packages are required to run the model:
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage
1. Clone the repository to your local machine:

git clone https://github.com/<your-username>/predictive-maintenance-pipes.git

2. Navigate to the project directory:

cd predictive-maintenance-pipes

3. Run the `main.py` file:

python main.py

## Funding
This project was funded by The Federal Ministry for Economic Affairs and Climate Action (BMWK).

## Contact
For any questions or inquiries, please contact the project lead at <your-email-address>.




"""
Predictive Maintenance for District Heating Pipes

Walking through District Heating Pipe Predictive Maintenance

Research Project Instandhaltung-FW
Project Number 03ET1625B
Funded by: The Federal Ministry for Economic Affairs and Climate Action (BMWK)

Developed by:
__________________
Pakdad Pourbozorgi Langroudi, M.Sc.
wissenschaftlicher Mitarbeiter / Research Associate

HafenCity Universität Hamburg (HCU)
Henning-Voscherau-Platz 1, 20457 Hamburg, Raum 5.008

pakdad.langroudi@hcu-hamburg.de
www.hcu-hamburg.de
+49 (0)40 42827-5332
__________________
Univ.-Prof. Dr.-Ing. Ingo Weidlich
Technisches Infrastrukturmanagement

HafenCity Universität Hamburg (HCU)
Henning-Voscherau-Platz 1, 20457 Hamburg, Raum 5.007

ingo.weidlich@hcu-hamburg.de
www.hcu-hamburg.de
+49 (0)40 42827-5700
__________________
Will Hoffmann
Research Assistant

Northwestern University
633 Clark St, Evanston, IL 60208, USA

willhoffmann2024@u.northwestern.edu
www.northwestern.edu
+1 563-543-2813
"""
# Adopted from: Machine Learning for Equipment Failure Prediction and Predictive Maintenance (PM) by Shad Griffin
# For more information on the topic please see the following article.
# https://medium.com/swlh/machine-learning-for-equipment-failure-prediction-and-predictive-maintenance-pm-e72b1ce42da1
# Published July 2020, Greatly revised February 2021.