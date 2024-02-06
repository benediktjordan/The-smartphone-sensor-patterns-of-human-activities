# Automatic Identification of User-Contexts Based on Smartphone Sensors in Naturalistic Settings

## Project Description

This project develops machine learning models to identify user contexts linked with smartphone overuse based on smartphone sensor data. Four user contexts have been in focus: human motion, transportation mode, important locations, and lying in bed before or after sleep. The aim was to identify these contexts based on data from 14 different smartphone sensors, namely motion-, location-, phone-related-, and environmental sensors.  The project's goal is to aid in understanding and reducing smartphone overuse. The data was collected mainly in a naturalistic setting and for some contexts additionally in a controlled laboratory setting. Unfortunately, for the context of transportation modes, not enough data to perform any meaningful analysis could be collected. Therefore, this context was excluded in the further analysis. 

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
F1 score of 81.7 on average over all LOSOCV iterations. This is way above the 16.7% baseline performance (the accuracy 
one would reach with randomly guessing classes; this can be calculated by dividing 100 by the number of classes), but 
still not perfect. The resulting confusion matrix is shown in the following figure.

 ![alt text](https://github.com/benediktjordan/context_detection/blob/7028bd944b49b79eac9c42d91aef4b49f576e41f/img/humanMotion_laboratory_confusionMatrix.png)

##### Important Features
In this subsection, the important features found for the different human motion classes in the laboratory dataset-based analysis,
when analysing the final model using the SHAP analysis, are presented: 
- **lying:** Looking at the feature importances of "lying" in the upper-left subplot of the subsequent figure, it is
interesting to notice that all of the ﬁve most important features are derived from the rotation
sensor. This is in contrast to the important features of the naturalistic data-based human mo-
tion model for which 3/5 most important features were derived from accelerometer or linear
accelerometer sensors. This difference could indicate that the "lying" patterns in the natu-
ralistic data are more differentiated from other classes by identifying the movement while for
identifying "lying" in the laboratory data the absolute rotation of the phone played a bigger role.
Interestingly, all ﬁve rotation features are derived from the y-axis of rotation which indicates
the tilt of the smartphone in the right-left direction. One way how the difference in the y-axis
of rotation between "lying" and other stationary classes could have been created is if people
are lying on the side holding their smartphone tilted to the side while in the other classes the
smartphone is usually not tilted to the side.
- **sitting at a table and standing:** Looking at the most important features, it is interesting to note that for "lying", "sitting at a
table", and "standing", 4/5 most important features are identical. They are all derived from the
rotation sensors y-axis and describe either absolute rotation values (ranges) or one frequency
of the rotation signal time series. These features point to the conclusion that the absolute ro-
tation, the rotation range, and the frequency in the rotation movement of the smartphone in
left-right direction play a role in differentiating between "lying", "sitting at a table", and "stand-
ing".
- **walking, running, and cycling:** Interestingly, 3/5 most relevant features are shared by all three dynamic human motion
classes. All these features are derived from the axes of the linear accelerometer and they all
describe changes in the time series. The ﬁnding that changes in motion sensors are impor-
tant to differentiate between different dynamic motions makes intuitively sense, conﬁrms the
insights gained in the data exploration and also conﬁrms the similar ﬁnding when analysing
the important features of the naturalistic data-based human motion model.

![alt text](https://github.com/benediktjordan/context_detection/blob/fa1d7c038ddb96811cc19f0af5d79c7cf7a16493/img/HumanMotion_Laboratory_Modeling_BestModel_FeatureImportances.png)


#### Naturalistic Dataset based results
As the main aim of this project was to develop machine learning models which can be used in real life, a model was also 
trained on the data collected in naturalistic settings. As not enough data was collected for the classes of running and 
cycling, these classes were excluded. Additionally, there was data collected for the case when the phone was used while 
sitting, lying, or standing while the phone was lying on a flat surface. As in this case the micro-movements of the phone 
are presumably very different compared to when being held in a hand, this case was created as a new class. Therefore,
five different human motion classes were included into this analysis: lying, sitting, standing, lying or sitting or standing
while the phone is on a flat surface, and walking. 
In the process of choosing and hyperparameter-optimising machine learning models, 
768 models have been compared. The best performing model reached a balanced accuracy of 71.3%
over all LOSOCV iterations. This is much above the baseline performance of 20% accuracy, but still not perfect. The resulting confusion matrix is shown in the following figure.

 ![alt text](https://github.com/benediktjordan/context_detection/blob/7028bd944b49b79eac9c42d91aef4b49f576e41f/img/humanMotion_naturalistic_confusionMatrix.png)


### Important Locations
#### Results 
The aim in this section was to identify the semantic locations of "home" and "office" of the participants solely based
on GPS data. To achieve this, a multi-step feature creation process was applied (compare the section about "data preprocessing")
and subsequently different machine learning models were trained and optimized. 
The best performing model classifies home location with a precision of 94% and a recall of 84%. This implies if 
the model classiﬁes a place as "home", this classiﬁcation can be very much trusted, while 16%
of all home locations are not classiﬁed as such. 
Office was initially classified by the best performing model with a precision of 36% and a recall of 67% (compare confusion 
matrix below). An interesting discovery was made when investigating the seven participants, for whom the
model detected an "office" but the participants didn´t report in any ES questionnaire to be at
the oﬃce (six participants) or the algorithm detected the "office" incorrectly (one participant;
these seven cases are the reason for the low precision of "oﬃce" detection): in 5/7 cases, it
was possible to ask the participants about the semantic meaning of the place, which the model
labeled for them as "oﬃce". Amazingly, in four of these ﬁve cases, the participants reported 
that the place actually represented a workplace. This is amazing as the algorithm
managed to correctly identify the workplace of these four participants although these places
were incorrectly labeled as "other" and therefore a potential source of incorrectness in model
training and although the models were trained on so little data regarding the "office" class. If
these four participants would have correctly labeled their "office" place, the precision of the
classiﬁcation model would have more than doubled from 36% to 89% and the recall would
have increased from 67% to 80%.

Therefore, in summary, the model identified "home" and "office" locations with recalls of 
84% and 80%, respectively, using only GPS data.

![alt text](https://github.com/benediktjordan/context_detection/blob/fa1d7c038ddb96811cc19f0af5d79c7cf7a16493/img/Location_Modeling_BestModel_ConfusionMatrix.png)


#### Important Features 
- **Home:** The three most inﬂuential features in classifying home, identified by the SHAP analysis, are the rank ascent of the time
per day spent at the place between 3:00 - 4:00 and between 6:00 - 7:00 as well as the fraction
of time spent at this place compared to the time spent at all places. These features could be
relevant because, on average, people spent most of their nights at home, and for many people
"home" is the place where they spent most time of all of their most frequent places they go.

- **Office:** The three most inﬂuential features for classifying "office", identified by the SHAP analysis, 
are the time spent 
at a place in the afternoon
and the fraction of time- as well as the trusted fraction of time spent at a place between 11:00-12:00. These features also are associated with an intuitive interpretation, since in most jobs
people are at the oﬃce, in case they are there, around the noon and afternoon time.


![alt text](https://github.com/benediktjordan/context_detection/blob/fa1d7c038ddb96811cc19f0af5d79c7cf7a16493/img/Location_Modeling_BestModel_FeatureImportance.png)

### Lying in Bed Before and After Sleep
The aim of this analysis was to identify whenever a person is lying in bed directly before or after sleep. 

#### Results 
The best model 
reached an average balanced accuracy of 57.7% over all LOSOCV iterations. This is much above the baseline performance of
16.7% accuracy (since the model was trained on classifying six classes; compare confusion matrix below). 
The two main classes of interest, "lying in bed before sleep" and "lying in bed after sleep", were classified with recalls of 72% and 50%, respectively.

![alt text](https://github.com/benediktjordan/context_detection/blob/fa1d7c038ddb96811cc19f0af5d79c7cf7a16493/img/Sleep_Modeling_HyperparameterTuning_ConfusionMatrix.png)

#### Detailled analysis and Important Features
- **lying in bed before sleep:** The class of "lying in bed before sleep" is classiﬁed with a precision
of 43% and a recall of 72%. This implies that less than half of the as "lying in bed before sleep"
classiﬁed samples really belong to that class, and that nearly three-quarters of all "lying in bed
before sleep" events are detected as such. A possible reason for this imbalance in precision
and recall can be found when comparing the important features in the middle-right subplot
of the subsequent figure: The most important feature for classifying an event as "lying in bed before sleep"
is, with much distance to the second most important, the "hour of the day" (hour_of_day). This
could imply that many events at a certain timeperiod of the day (i.e. between 21:00 - 0:00) are
classiﬁed as "lying in bed before sleep". This would capture most of the "lying in bed before
sleep" events accurately, but generate also many FP, which would result
in the low precision - high recall imbalance observed above. The FP for "lying in bed before
sleep" are mostly with the "not lying: stationary" events, which include, for example, "standing"
or "sitting" events. The reason could be that only these events are also present at late times,
while participants were "lying in bed at other times" or "lying on the couch" at other times of
the day.
- **lying in bed after sleep:** The class of "lying in bed after sleep" is classiﬁed with a precision
of 56% and a recall of 50%. This implies that around half of all samples which are classiﬁed as
"lying in bed after sleep" are actually belonging to this class and exactly one-half of all "lying
in bed after sleep" events are detected as such. The low level of precision is surprising since
the feature "static period end distance" (which is the time-distance to the end of the last long
static period) seemed to provide quite speciﬁc information about the appearance of a "lying
in bed after sleep" event. But also in this case looking at the important
features for predicting this class, which are visualized in the middle-left subplot in the subsequent
figure , provides some possible explanatory clues: the feature describing the time-distance to the
end of the last long static period is only the fourth most important feature, after the hour of the
day and two classiﬁcations of the human motion model. The reason for choosing this feature-
ranking, instead of using the "static_period_end_distance" as the most important feature, is
found somewhere within the gini impurity-optimization during the DF training process. Most
probably, other events, besides "lying in bed after sleep", were similarly close to the end of the
closest long static period, and therefore the impurity optimization needed to be done using
another feature like "hour of the day". 


For both the "lying in bed before sleep" and "lying in bed after sleep" classes, as well as
the other two classes associated with lying ("lying on the couch" and "lying in bed at other
times") the human motion model prediction feature of "lying (in hand/s)" played an important
role. Interestingly, for the two main classes, this feature was only ranked second or third most
important. This could be a factor for the high rate of misclassiﬁcation between the two main
classes and the "not lying: stationary" class.

One other interesting insights from the SHAP feature importance ranking is that the pre-
dictions of important locations are not listed among the most important features for any of
six classes, except for "lying on the couch" but there also at ﬁfth place. This indicates that
the information if a participant is at "home", another "home 2", the "oﬃce", or "another place"
doesn´t contain much differentiative value for differentiating between these classes.

![alt text](https://github.com/benediktjordan/context_detection/blob/fa1d7c038ddb96811cc19f0af5d79c7cf7a16493/img/Sleep_Modeling_HyperparameterTuning_SHAPFeatureImportances.png)

## Data Structure 
The data collection occurred 
in two phases: naturalistic and laboratory settings, each addressing specific needs for model development.

**Naturalistic Data Collection**

In the naturalistic setting, data were collected from participants in their everyday environments. 
Ultimately, 18 participants from nine cities across five 
countries contributed, with a diverse range of occupations and an average age of 29.5 years. Participants were 
instructed on how to answer Experience Sampling (ES) questionnaires and set up the technical system for data collection.

**Data Overview:** Data collection comprised ES and smartphone sensor data. The ES method was employed to unobtrusively 
gather contextual information from participants, while sensor data collection aimed at capturing information relevant 
to context detection. A total of 1,113 ES events were recorded. Sensor data from 14 smartphone sensors 
across four categories — motion, location, phone-related, and environmental — were collected, hypothesized to be predictive 
of various contexts such as human motion, transportation mode, and lying in bed before/after sleep. Despite the aim to 
utilize a comprehensive set of sensors, practical constraints led to the exclusion of certain sensors like gravity, 
Bluetooth, WiFi, battery, and environmental data from the analysis.

**Naturalistic Data Distribution:** The table below illustrates the distribution of collected data among participants, 
highlighting the variability in the number of ES events and the duration of sensor data collection. On average, 
each participant contributed to 285 hours (approximately 11.9 days) of sensor data, though this varied significantly 
among participants. The table also denotes the cities where participants resided, offering a glimpse into the geographic 
diversity of the data collection effort.

![alt text]()

**Laboratory Data Collection**

The laboratory phase was initiated to supplement the naturalistic data with information from specific contexts 
that were underrepresented or absent in the initial collection. This controlled environment allowed for precise data 
collection on a set of predetermined activities. Six participants were involved in this phase, performing activities 
like standing, sitting, lying, walking, running, cycling, and using the lavatory. The variability in activities 
performed by each participant was due to external factors such as weather conditions.

**Laboratory Data Overview:** Sensor data were collected for six sensors, although issues with accelerometer and GPS sensor 
configuration led to their exclusion from further analysis. The activities and the duration of data collection are 
summarized in the table below, indicating the total minutes of activity performed and sensor data collected per participant.

![alt text]()

## Dataset Availability 
Please note that the dataset utilized in this project is not publicly available. This is due to privacy considerations 
of the participants as well as due to agreements with Wellspent. The data includes sensitive and personal information 
that necessitates strict confidentiality. As such, in adherence to privacy laws and further guidelines, 
the dataset is retained exclusively for the purposes of this research project and is not accessible for external 
use or review.

## Methodology 
The project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) model. Subsequently, the methodology 
of data collection, data preprocessing, and modeling is presented.


### Data Collection
The data collection process for this study was conducted in two distinct phases: naturalistic and laboratory settings, 
each with its unique experimental setup aimed at gathering comprehensive sensor and Experience Sampling (ES) data 
from participants.

**Naturalistic Data Collection**

In the naturalistic phase, participants from the author's social circle
who owned and used an iPhone 8 or newer, were recruited to collect data during their daily activities. 
This phase utilized ES, a technique where participants report their experiences, thoughts, or behaviors at 
multiple times throughout the day via smartphone notifications. Participants received notifications every 
full hour from 06:00 to 23:00, prompting them to answer a questionnaire designed to capture their current 
context across five areas: location, body motion, activity, evaluation of phone use, and phone location.

Sensor data from the participants' smartphones were collected passively and continuously alongside the ES responses. 
The sensors were grouped into four categories: motion (including accelerometer, linear accelerometer, gravity sensor, gyroscope, 
magnetometer, rotation sensor), 
location (GPS, Bluetooth, WiFi, and timezone sensors), phone-related (screen and battery status), and environmental
(barometer and weather conditions). These sensors were selected based on their potential relevance to detecting various 
contexts such as human motion, transportation mode, and lying in bed before or after sleep. The technical implementation of the data collection was facilitated through the Aware Framework, a system comprising the Aware Micro Server, the Aware App, and a cloud-based database, ensuring the efficient collection and management of both sensor and ES data.

**Laboratory Data Collection**

The laboratory phase was designed to collect data in a controlled environment, focusing on specific activities 
that were either underrepresented or absent during the naturalistic data collection. Participants were asked to perform 
a series of predefined activities, including dynamic human motions (walking, running, cycling) and static motions 
(standing, sitting, lying), as well as using the lavatory, each for a minimum of five minutes. The aim was to collect 
sensor data from smartphones while participants engaged in these activities, using iPhones owned by the participants. 
The data collection covered various sensors, similar to the naturalistic phase, but with a specific focus on motion 
sensors such as the accelerometer and gyroscope, recorded at specified sampling rates to ensure the capture of relevant 
activity data.

**Technical and Methodological Considerations**

Both phases of data collection were meticulously planned to balance the need for detailed, context-rich data against 
the constraints of smartphone battery life, data storage, and participant convenience. The choice of sensors and their 
sampling rates was influenced by literature reviews and the practical considerations of real-world application feasibility. 
The laboratory phase, in particular, allowed for the precise labeling of sensor data according to specific activities,
enhancing the dataset's utility for developing accurate classification models.

Overall, the data collection methodology outlined combines the breadth of real-life context capture through naturalistic 
data collection with the depth of specific context identification afforded by laboratory settings. This dual approach 
enriches the dataset, providing a robust foundation for developing and testing classification models that can 
accurately interpret and predict contexts based on smartphone sensor data.

### Data Preprocessing
This section outlines the multifaceted process of data preparation, crucial for transforming both laboratory and 
naturalistic data into a format suitable for modeling. The complexity arises from differing preparation steps for 
various types of data  and context-specific 
techniques for data manipulation. A specific subset of the following techniques has been used for each context 
contexts. 

Initially, the ***ES answer transformation*** is applied to transform ES events into class 
labels for context, contingent on the relevancy and completeness of responses. ***Initial data selection*** focuses on managing 
the voluminous sensor data by narrowing down to segments around ES events, enhancing computational feasibility. 
***Active smartphone sessions*** are identified through unlocked screen status as a proxy for user engagement, ensuring 
relevance to the context under investigation. To maintain data reliability, the ***duration of answers*** to ES questionnaires
is limited, excluding data points where participants took excessively long to respond.

In the realm of sensor data, ***merging raw time series*** addresses the challenge of inconsistent sampling times across 
sensors, using a method that aligns data points based on temporal proximity. ***Feature creation*** is meticulously undertaken
across various domains: motion sensors undergo a detailed process to generate features from time, frequency, and
time-frequency domains, utilizing a comprehensive suite of algorithms to capture the dynamics and characteristics 
of human motion. Similarly, GPS data is enriched by deriving speed and acceleration features, subsequently augmented 
by higher-level features to capture movement patterns like "places". Place features are crafted to identify significant locations 
through GPS data, incorporating time-based categorizations and frequency of visits to enhance contextual understanding. 
Time features are derived to inform models about temporal patterns, while the identification of long static periods, 
particularly from accelerometer data, assists in inferring periods of minimal movement, potentially indicative of sleep.

The process also involves ***merging feature time series*** from different sensors to construct a cohesive dataset, ready 
for modeling. A ***data-driven feature selection*** strategy is employed to refine the feature set, reducing dimensionality
while preserving the most informative attributes for modeling. This approach meticulously eliminates features with
excessive missing values, non-variability, and weak correlation to the target variable, employing statistical tests 
to ensure the relevance of retained features. 

### Modeling 
This section provides an overview of the machine learning models and methodologies employed in the project, aimed at 
addressing specific classification challenges through data-driven approaches. Central to our modeling strategy are 
Decision Forests (DF), which serve as the primary machine learning model for classifying sensor data across various 
contexts due to their robustness and efficiency in handling large, high-dimensional datasets. DFs are complemented by 
a rule-based one-value class selection procedure, specifically designed to ensure that certain classes, such as "home" 
or "office," are uniquely assigned to locations, addressing the unique requirements of identifying important places.

Additionally, hierarchical clustering, utilizing agglomerative methods with complete linkage, is applied to GPS data 
for extracting frequently visited locations, ensuring meaningful spatial grouping. This approach is particularly 
effective in managing and interpreting geospatial data, where the calculation of distances between clusters plays a 
crucial role in identifying significant locations.

The project also incorporates several critical methods and concepts to enhance the modeling process:

- **Label Segment Selection:** A data-driven methodology to determine the optimal time frame around event-specific 
sensor data, ensuring the relevance and accuracy of the data used for model training and prediction.
- **Leave One Subject Out Cross-Validation (LOSOCV):** A validation technique that enhances the model's 
generalizability by training and testing on data from different individuals, ensuring the model's robust 
performance across new, unseen data.
- **Hyperparameter Tuning:** Employing nested cross-validation, this process fine-tunes the model's hyperparameters 
to optimize performance, addressing the challenge of selecting the most effective model configuration.
- **Evaluation Metrics:** The use of balanced accuracy, F1 score, and other metrics allows for a fair assessment of 
model performance, especially crucial in imbalanced datasets. These metrics provide a quantitative measure of 
the model's accuracy and effectiveness.
- **Machine Learning Model Interpretation:** Leveraging SHapley Additive exPlanations (SHAP) values to interpret the 
model's predictions, this approach offers insights into the importance of different features, enhancing the transparency 
and understandability of the model.


## Installation 
To run the AGENDER 2.0 project, you need to have Python 3.7 installed on your machine. Additionally, certain libraries and 
dependencies are required, which can be installed via the following command:

`pip install -r requirements.txt
`

## Usage 
To use this project, first clone the repository to your local machine:

`git clone [repository-link]`

Navigate to the project directory and execute the main script:

`cd context_detection
python main.py`


## Contributing 
Contributions to the AGENDER 2.0 project are welcome. Please ensure to follow the standard procedures for contributing 
to open-source projects, including using pull requests for proposing changes.

License: this project is licensed under the MIT License.

## Acknowledgments
This project wouldn´t have been possible without the stellar continuous supervision of Lennart Schaepermeier (main 
university supervisor) and Marius Rackwitz (company supervisor) - thank you for this! Furthermore, thanks to Prof. Kerschke 
and Prof. Kiebel for agreeing to take the lead supervision roles. 
Additionally, my gratitude goes to the entire team of Wellspent for
their continuous support and the participants for their time and effort. 
  

