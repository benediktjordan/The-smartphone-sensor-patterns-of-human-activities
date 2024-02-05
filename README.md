# Automatic Identification of User-Contexts Based on Smartphone Sensors in Naturalistic Settings

## Project Description

This project develops machine learning models to identify user contexts linked with smartphone overuse based on smartphone sensor data. Four user contexts have been in focus: human motion, transportation mode, important locations, and lying in bed before or after sleep. The aim was to identify these contexts based on data from 14 different smartphone sensors, namely motion-, location-, phone-related-, and environmental sensors.  The project's goal is to aid in understanding and reducing smartphone overuse. The data was collected in a naturalistic setting. 

The project was created during the master thesis in the master´s program "Computational Modeling and Simulation" with a major in "Computational Life Sciences" at the TU Dresden (Germany) by Benedikt Jordan. The thesis was supervised by Lennart Schaepermeier and Prof. Kerschke from the chair of Big Data Analytics in Transportation. 

## Repository Content

This repository contains the Python code developed for the project. The codebase includes all modules of the data science project. The key components are:

- data loading and exploration
- initial label and sensor transformations
- merging and imputations to cope with missing values and combine different sensor data 
- feature extraction: different for specific sensors
- modeling: creation, training and evaluation of machine learning models

## Project Results
The project aimed fat identifying in real time four different user contexts based on smartphone sensor data. Subsequently, the main results are presented: 

### Human Motion 
Six human motions were in focus in this analysis: lying, sitting at a table, standing, walking, running, and cycling. 
Two different datasets were used to perform two analyses: the "laboratory dataset", in which the participants were asked 
to perform the activities in a controlled environment, and the "naturalistic dataset", in which the participants 
performed the activities in their daily life (compare the section about "data structure").

#### Laboratory Dataset based results
The best performing machine learning model trained on the laboratory dataset reached a balanced accuracy of 80.8% and a
F1 score of 81.7 on average over all LOSOCV iterations. The resulting confusion matrix is shown in the following figure.

 ![alt text]()

#### Naturalistic Dataset based results
As the main aim of this project was to develop machine learning models which can be used in real life, a model was also 
trained on the data collected in naturalistic settings. In the process of choosing and hyperparameter-optimising machine learning models, 
768 models have been compared. The best performing model reached a balanced accuracy of 71.3%
over all LOSOCV iterations. The resulting confusion matrix is shown in the following figure.

 ![alt text]()

### Transportation Mode

### Important Locations


    Important Locations: Identified "home" and "office" locations with recalls of 85% and 80%, respectively, using only GPS data.
    Lying in Bed Before and After Sleep: Detected with recalls of 72% and 50%.
    The models show potential in real-life applications for mitigating smartphone overuse.

### Lying in Bed Before and After Sleep
    
## Data Structure 

Data collected included smartphone sensor data (like GPS, accelerometer) and self-reported information through experience sampling (ES). This data was used to analyze various contexts like human motion, important locations, and periods of lying in bed before and after sleep.
Installation

Dataset Availability 

## Data Preprocessing 

## Modeling 

## Installation 

## Usage 

## Contributing 

## Acknowledgments

  
## Methodology

The project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) model, involving:

    Data Collection: Data was gathered from 18 participants over an average of 12 days in naturalistic settings.
    Data Exploration: Involved creating generic maps from GPS data and exploring sensor data.
    Data Preparation: Techniques included ES answers transformation, feature creation, and data-driven feature selection.
    Modeling: Machine learning models and other necessary methods were applied for analysis.

Key Findings

    Human Motion: Detected four human motion activities with a balanced accuracy of 64%.
    Important Locations: Identified "home" and "office" locations with recalls of 85% and 80%, respectively, using only GPS data.
    Lying in Bed Before and After Sleep: Detected with recalls of 72% and 50%.
    The models show potential in real-life applications for mitigating smartphone overuse.


[Installation instructions for the project.]
Usage

[How to use this project, including necessary commands or scripts.]
Code Structure

[Details about the structure of the code and its main components.]
Contributing

[Guidelines for contributing to the project.]
License

[License details of the project.]
Acknowledgments

Acknowledgments to those who contributed to the thesis and project.
