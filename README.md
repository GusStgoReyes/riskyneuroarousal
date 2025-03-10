# Understanding Loss Aversion in Context-Dependent Risky Decision-Making

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Repository Structure](#repositorystructure)

## Introduction
This repository contains code and data processing pipelines for studying loss aversion during risky decision making. Loss aversion is a well-documented phenomenon in decision-making, often explained by Prospect Theory, where losses are subjectively weighted more heavily than equivalent gains. However, alternative models, such as Drift Diffusion Models (DDMs), suggest that loss aversion may also arise from the decision process itself rather than solely from subjective valuation.

In this project, we aim to explore why the degree of loss aversion varies across contexts and investigate the neural mechanisms underlying these differences. A particular focus is on the arousal system, which has been linked to loss aversion—where arousal increases with perceived losses and may influence response biases. To examine this, we leverage pupil dilation as an arousal index to understand how physiological responses adapt in different risky decision-making scenarios.

If you wish to learn more, navigate through `notebooks/reproducible`! The jupyter notebooks take you through the data and the analysis. 

## Installation
If you don't already have uv installed, use `pip install uv` then you can clone the repository and generate the `.venv` using:
```bash
sh setup_env.sh
```
If running the notebooks in VSCode, just select the `.venv` from the root dirctory as the kernel.

Furthermore, add the path of the data folder to `config.json` and add your username. 

## Repository Structure
- /data:
    - Contains the behavioral data and an example eye tracking data
    - CSV files with parameters estimated with the models
- /notebooks:
    - /exploratory:
        - Exploratory notebooks that are messy and are being worked on to make more readable. 
    - /reproducible:
        - Notebooks that are well documented and run through experiments. 
        - The Notebooks are numbered in an order that introduces the data
    - /preprocessing:
        - Notebooks to test functionality of preprocessing pipeline
- /src:
    - /modeling:
        - Python scripts to run the DDM and Prospect Theory models
        - These files save the results to /data
        - /sbatch:
            - Sbatch files to submit the jobs in Sherlock
    - /preprocessing:
        - R scripts to preprocess the pupil data.
        - You can run `pupil_preprocessing_enhanced.R` to use the pipeline. 
    - /utils:
        - Helper functions to condense analysis in notebooks to functions.