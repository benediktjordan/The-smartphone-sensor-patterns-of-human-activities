## Note: I could use Tensorflow or Sklearn for Decision Forests. I will use Tensorflow for this example.
## Seems that in Sklearn there are more decision forests, customization methods & documentation.

#region import
import pickle
import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time

#Classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#nested CV
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Feature Importance
import shap

#Statistical Testing
from scipy.stats import binom_test
from sklearn.model_selection import permutation_test_score

#Keras Tuner
import keras
import keras_tuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
from kerastuner.tuners import Hyperband



try:
  from wurlitzer import sys_pipes
except:
  from colabtools.googlelog import CaptureLog as sys_pipes
#endregion

#region Sklearn Decision Forests
## The Sklearn DF implementation is based on this tutorial: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

## the following implementation is based on AGENDER 2.0 project
#Pipeline: using LOSOCV & permutation test
t0 = time.time()

dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"

sensors_included = "all"
timeperiod_segments = 2 #seconds
path_df_features = dir_sensorfiles + "data_preparation/features/highfrequencysensors-" + sensors_included + "_timeperiod-" + str(timeperiod_segments) + " s.pkl"
with open(path_df_features, "rb") as f:
    df = pickle.load(f)


#data_new = pd.DataFrame()

#select only probands which have 5 or more stress events
#probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
#						  "AGENDER09", "AGENDER14", "AGENDER15"]
#for i in probands_above_5events:
#	data_new = data_new.append(data[data["Proband"]==i])



#select only probands which have accuracy less than 50%
#probands_below_50= ["AGENDER05", "AGENDER09", "AGENDER14", "AGENDER29"]
#for i in probands_below_50:
#	data_new = data_new.append(data[data["Proband"]==i])

df = df.reset_index(drop=True)

#which columns to drop (either with ACC or without ACC)
#dropcols = []
#data = data.drop(columns=dropcols)

# Make list of all ID's in idcolumn
IDlist = set(df["device_id"])

#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

permutation_pvalue = list()
permutation_modelaccuracy = list()
pvalues_binomial = list()

#for loop to iterate through LOSOCV "rounds"

for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	df["device_id"] = df["device_id"].apply(str)
	data_filtered = df[df["device_id"] != LOOCV_O]
	data_cv = df[df["device_id"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.copy()
	X_test_df = data_test.drop(columns=[label_column_name, "device_id", "sensor_timestamp", "ESM_timestamp"])
	X_test = np.array(X_test_df)
	y_test_df = data_test[label_column_name]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.copy()
	X_train_df = data_train.copy()

	#define the model
	model = RandomForestClassifier(random_state=1)

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["device_id"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 2):
		try:
			IDlist_inner[l+1]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["device_id"] != IDlist_inner[l]) & (X_train_df["device_id"] != IDlist_inner[l+1]) ]
			test = X_train_df[(X_train_df["device_id"] == IDlist_inner[l]) | (X_train_df["device_id"] ==  IDlist_inner[l+1]) ]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["device_id"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=[label_column_name, "device_id", "sensor_timestamp", "ESM_timestamp"])
	X_train = np.array(X_train_df)
	y_train_df = data_train[label_column_name]  # This is the outcome variable
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

	space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				 min_samples_leaf=min_samples_leaf, max_features=max_features)

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

	# execute search
	result = search.fit(X_train, y_train)

	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_

	#apply permutation test
	## create dataframe which contains all data and delete some stuff
	data_permutation = data.copy()
	data_permutation = data_permutation.reset_index(drop=True)

	## create list which contains indices of train and test samples (differentiate by proband)
	split_permutation = []
	train_permutation = data_permutation[data_permutation["device_id"] != i]
	test_permutation = data_permutation[data_permutation["device_id"] == i]
	add_permutation = [train_permutation.index, test_permutation.index]
	split_permutation.append(add_permutation)

	##Drop some stuff
	#data_permutation = data_permutation.drop(columns=dropcols)

	##Create X and y dataset
	X_permutation = data_permutation.drop(columns=[label_column_name, "device_id", "sensor_timestamp", "ESM_timestamp"])
	y_permutation = data_permutation[label_column_name]

	##compute permutation test
	score_model, perm_scores_model, pvalue_model = permutation_test_score(best_model, X_permutation, y_permutation, scoring="accuracy", cv=split_permutation, n_permutations=1000)

	## visualize permutation test restuls
	fig, ax = plt.subplots()
	plt.title("Permutation Test results with Proband " + str(i) + " as Test-Data")
	ax.hist(perm_scores_model, bins=20, density=True)
	ax.axvline(score_model, ls='--', color='r')
	score_label = (f"Score on original\ndata: {score_model:.2f}\n"
				   f"(p-value: {pvalue_model:.3f})")
	ax.text(0.14, 125, score_label, fontsize=12)
	ax.set_xlabel("Accuracy score")
	ax.set_ylabel("Probability")
	plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_timeperiod_segments-" + timeperiod_segments + "_test_proband-" + str(i) + "_Permutation.png")
	plt.show()


	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)

	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")

	#Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data" )
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_timeperiod_segments-" + timeperiod_segments + "_test_proband-" + str(i) + '_ConfusionMatrix.png')
	plt.show()

	#apply binomial test
	pvalue_binom = binom_test(x=mat[0][0]+mat[1][1], n=len(y_test), p=0.5, alternative='greater')

	#feature importance: compute SHAP values
	explainer = shap.Explainer(best_model)
	shap_values = explainer.shap_values(X_test)
	plt.figure()
	plt.title("Feature Importance for iteration with proband " +str(i) + " as test set")
	shap.summary_plot(shap_values[1], X_test_df, plot_type="bar", show=False)
	#plt.savefig('02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_SHAPFeatureImportance_v2.png')
	fig = plt.gcf()
	fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_timeperiod_segments-" + timeperiod_segments + "_test_proband-" + str(i) + '_SHAPFeatureImportance.png')
	plt.show()

	# store statistical test results (p-value permutation test, accuracy of that permutation iteration, pvalue binomial test) in list
	permutation_pvalue.append(pvalue_model)
	permutation_modelaccuracy.append(score_model)
	pvalues_binomial.append(pvalue_binom)

	# store the result
	test_proband.append(i)
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)

	# Save the results:
	results = pd.DataFrame()
	results["Test-Proband"] = test_proband
	results["Accuracy"] = outer_results_acc
	results["Accuracy by PermutationTest"] = permutation_modelaccuracy
	results["F1"] = outer_results_f1
	results["Precision"] = outer_results_precision
	results["Recall"] = outer_results_recall
	results["P-Value Permutation Test"] = permutation_pvalue
	results["P-Value Binomial Test"] = pvalues_binomial
	results.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_timeperiod_segments-" + timeperiod_segments + "_test_proband-" + str(i) + "_results_cumulated.csv.")

	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("Permutation-Test p-value was " + str(pvalue_model) + " and Binomial Test p-values was " + str(pvalue_binom))
	print("The proband taken as test-data for this iteration was " + str(i))
	print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

t1 = time.time()
print("The whole process has taken so many hours: " + str(((t1 - t0)/60/60)))

# summarize the estimated performance of the model
print('Mean Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('Mean F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Mean Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Mean Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
print("Mean p-value of Permutation Test: %.3f (%.3f)" % (mean(permutation_pvalue), std(permutation_pvalue)))
print("Mean of p-value of Binomial Test: %.3f (%.3f)" % (mean(pvalues_binomial), std(pvalues_binomial)))



# os.system("shutdown /h") #hibernate
# endregion












#endregion


#region temporary: checking AGENDER 2.0 methods


#region Pipeline 1: "traditional" CV & traditional feature interpretation
# define model
forest = RandomForestClassifier(random_state=1)

# tuning hyperparameter for RF with GridSearch
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 10, verbose = 1,n_jobs = -1)
bestF = gridF.fit(X_train, y_train)
##visualizing results
gridF.best_params_

#tuning hyperparameter for RF with Validation Curve
#region n_estimators
param_range = [100, 300, 500, 800, 1200]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'n_estimators',
                                param_range = param_range, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Estimators")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#region max_depth
param_range1 = [5, 8, 15, 25, 30]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'max_depth',
                                param_range = param_range1, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range1, train_mean, label="Training score", color="black")
plt.plot(param_range1, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range1, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range1, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("maximum depth")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#region min_sample_split
param_range2=[2, 5, 10, 15, 100]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'min_samples_split',
                                param_range = param_range2, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range2, train_mean, label="Training score", color="black")
plt.plot(param_range2, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range2, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range2, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("minimum sample split")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#region min_samples_leaf
param_range3 = [1, 2, 5, 10]
train_scores, test_scores = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train,
                                param_name = 'min_samples_leaf',
                                param_range = param_range3, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range3, train_mean, label="Training score", color="black")
plt.plot(param_range3, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range3, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range3, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("minimum sample leaf")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#endregion

#running actual model
#from sklearn.ensemble import RandomForestClassifier
#X, y = df.iloc[:,:-1], clusters_tsne_scale['tsne_clusters']
#clf = RandomForestClassifier(n_estimators=100).fit(X_test, y_test)
#print(y)
#Best parameters for ACC
## from Grid search: {'max_depth': 15, 'min_samples_leaf': 10,'min_samples_split': 2, 'n_estimators': 100}
##

# with CV
# 10-Fold Cross validation
from sklearn.metrics import SCORERS
#list of possible scores: sorted(SCORERS.keys())
from sklearn.model_selection import cross_val_score
clf_cv = RandomForestClassifier(random_state = 1, max_depth = 15,
							 n_estimators = 100, min_samples_split = 5,
							 min_samples_leaf = 2)
accuracy = cross_val_score(clf_cv, X_train, y_train, scoring= "accuracy", cv=10)
accuracy.mean()
f1 = cross_val_score(clf_cv, X_train, y_train, scoring= "f1", cv=10)
f1.mean()
precision = cross_val_score(clf_cv, X_train, y_train, scoring= "precision", cv=10)
precision.mean()
recall = cross_val_score(clf_cv, X_train, y_train, scoring= "recall", cv=10)
recall.mean()

# alternative cross validation funciton
from sklearn.model_selection import cross_validate
cv_results = cross_validate(clf, X_train, y_train, cv=10)


#without CV
clf = RandomForestClassifier(random_state = 1, max_depth = 15, n_estimators = 300, min_samples_split = 2, min_samples_leaf = 2).fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualizing in Confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

#Feature Importance Analysis
#modelname.feature_importance_
data = np.array([clf.feature_importances_, X_train.columns]).T
data = pd.DataFrame(data, columns=['Importance', 'Feature']).sort_values("Importance", ascending=False).head(10)
columns = list(data.Feature.values)
#plot
fig, ax = plt.subplots()
width = 0.4 # the width of the bars
ind = np.arange(data.shape[0]) # the x locations for the groups
ax.barh(ind, data["Importance"], width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(columns, minor=False)
plt.title("Feature importance in RandomForest Classifier")
plt.xlabel("Relative importance")
plt.ylabel("feature")
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
plt.show()

#endregion

#region Pipeline 2: using nested crossvalidation & feature interpretation with SHAP!

#region nested CV
t0 = time.time()
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns

for train_ix, test_ix in cv_outer.split(X_all, y_all):
	t0_inner = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = RandomForestClassifier(random_state=1)

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

	space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				  min_samples_leaf=min_samples_leaf, max_features = max_features)
	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True, n_jobs=-1)
	# execute search
	result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_
	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)
	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	# store the result
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)
	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	print("This inner iteration has taken so many seconds: " + str(t1_inner-t0_inner))

	#feature importance: compute SHAP values
	#explainer = shap.Explainer(best_model)
	#shap_values = explainer.shap_values(X_test)
	# for each iteration we save the test_set index and the shap_values
	#list_shap_values.append(shap_values)
	#list_test_sets.append(test_ix)
t1 = time.time()
print("This process took so many seconds: " +str(t1-t0))
os.system("shutdown /h") #hibernate


#combining SHAP results from all iterations
#test_set = []
#shap_values = []
test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])

for i in range(1,len(list_test_sets)):
	test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
	shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
#bringing back variable names: X_test is the X_all data reordered due to the 10-test sets in outer CV
test_set = test_set.astype(int)
X_test = pd.DataFrame(X_all[test_set],columns = columns) #this should be used for SHAP plots!

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))

save_obj(shap_values, "normalstress_shap_values_DecisionForest_withoutACC_WITHOUTMEANMEDIANRR")
save_obj(X_test, "normalstress_X_test_DecisionForest_withoutACC_WITHOUTMEANMEDIANRR")
#shap_values = load_obj("shap_values_DecisionForest_withACC.pkl")
#X_test = load_obj("X_test_DecisionForest_withACC.pkl")

#endregion

#region SHAP Feature interpretation
# feature importance plot
shap.summary_plot(shap_values[0], X_test, plot_type="bar")

#SHAP feature importance plot
shap.summary_plot(shap_values[1], X_test)

#Force Plot
#shap.plots.force(explainer.expected_value[0], shap_values[0])

#partial dependence plot: analyze individual features
shap.dependence_plot("HRV_MeanNN", shap_values[0], X_test)

#endregion
#endregion

#region Pipeline 3: using LOSOCV
t0 = time.time()
from sklearn.metrics import confusion_matrix

df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")
data = df_class_allfeatures_del_NaN_binary_shuffled.copy()

# drop Proband 16 as it is a duplicate of Proband 15
data = data[data["Proband"]!= "AGENDER16"]

#select only probands which have 5 or more stress events

probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
						  "AGENDER09", "AGENDER14", "AGENDER15"]

data_new = pd.DataFrame()
for i in probands_above_5events:
	data_new = data_new.append(data[data["Proband"]==i])

data = data_new.reset_index(drop=True)


#which columns to drop (either with ACC or without ACC)
dropcols = []

# Make list of all ID's in idcolumn
IDlist = set(data["Proband"])

#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

#for loop to iterate through LOSOCV "rounds"


for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	data["Proband"] = data["Proband"].apply(str)
	data_filtered = data[data["Proband"] != LOOCV_O]
	data_cv = data[data["Proband"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.drop(columns=dropcols)
	X_test_df = data_test.drop(columns=["Label", "Proband"])
	X_test = np.array(X_test_df)
	y_test_df = data_test["Label"]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.drop(columns=dropcols)
	X_train_df = data_train.copy()

	#define the model
	model = RandomForestClassifier(random_state=1)

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["Proband"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 3):
		try:
			IDlist_inner[l+2]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["Proband"] != IDlist_inner[l]) & (X_train_df["Proband"] != IDlist_inner[l+1]) & (X_train_df["Proband"] != IDlist_inner[l+2])]
			test = X_train_df[(X_train_df["Proband"] == IDlist_inner[l]) | (X_train_df["Proband"] ==  IDlist_inner[l+1]) | (X_train_df["Proband"] ==  IDlist_inner[l+2])]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["Proband"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=["Label", "Proband"])
	X_train = np.array(X_train_df)
	y_train_df = data_train["Label"]
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

	space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				 min_samples_leaf=min_samples_leaf, max_features=max_features)

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

	# execute search
	result = search.fit(X_train, y_train)

	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_

	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)

	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")

	# store the result
	test_proband.append(i)
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)
	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("The proband taken as test-data for this iteration was " + str(i))

	#Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data" )
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()

	print("This inner iteration has taken so many seconds: " + str(t1_inner - t0_inner))
t1 = time.time()
print("The whole process has taken so many seconds: " + str(t1 - t0))

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
os.system("shutdown /h") #hibernate






# Add idcolumn to the dropped columns
drop = [idcolumn]
drop = drop + dropcols

# Initialize empty lists and dataframe
errors = []
rmse = []
mape = []
importances = pd.DataFrame(columns=['value', 'importances', 'id'])









# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

list_shap_values = list()
list_test_sets = list()
columns = X_all_df.columns

for train_ix, test_ix in cv_outer.split(X_all, y_all):
	t0_inner = time.time()
	# split data
	X_train, X_test = X_all[train_ix, :], X_all[test_ix, :]
	y_train, y_test = y_all[train_ix], y_all[test_ix]
	# configure the cross-validation procedure
	cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = RandomForestClassifier(random_state=1)

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

	space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				  min_samples_leaf=min_samples_leaf, max_features = max_features)
	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True, n_jobs=-1)
	# execute search
	result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_
	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)
	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	# store the result
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)
	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc,f1, precision, recall, result.best_score_, result.best_params_))
	print("This inner iteration has taken so many seconds: " + str(t1_inner-t0_inner))

	#feature importance: compute SHAP values
	#explainer = shap.Explainer(best_model)
	#shap_values = explainer.shap_values(X_test)
	# for each iteration we save the test_set index and the shap_values
	#list_shap_values.append(shap_values)
	#list_test_sets.append(test_ix)
t1 = time.time()
print("This process took so many seconds: " +str(t1-t0))
os.system("shutdown /h") #hibernate


#combining SHAP results from all iterations
#test_set = []
#shap_values = []
test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])

for i in range(1,len(list_test_sets)):
	test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
	shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
#bringing back variable names: X_test is the X_all data reordered due to the 10-test sets in outer CV
test_set = test_set.astype(int)
X_test = pd.DataFrame(X_all[test_set],columns = columns) #this should be used for SHAP plots!

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))

save_obj(shap_values, "normalstress_shap_values_DecisionForest_withoutACC_WITHOUTMEANMEDIANRR")
save_obj(X_test, "normalstress_X_test_DecisionForest_withoutACC_WITHOUTMEANMEDIANRR")
#shap_values = load_obj("shap_values_DecisionForest_withACC.pkl")
#X_test = load_obj("X_test_DecisionForest_withACC.pkl")


#endregion

#region Pipeline 4: using LOSOCV & permutation test
t0 = time.time()

df_class_allfeatures_del_NaN_binary_shuffled = load_obj("df_class_allfeatures_del_NaN_binary_shuffled_includingProband.pkl")
data = df_class_allfeatures_del_NaN_binary_shuffled.copy()

# drop Proband 16 as it is a duplicate of Proband 15
data = data[data["Proband"]!= "AGENDER16"]


#select only probands which have 5 or more stress events
#probands_above_5events = ["AGENDER01", "AGENDER02", "AGENDER04", "AGENDER05", "AGENDER06",
#						  "AGENDER09", "AGENDER14", "AGENDER15"]

data_new = pd.DataFrame()
#for i in probands_above_5events:
#	data_new = data_new.append(data[data["Proband"]==i])



#select only probands which have accuracy less than 50%
probands_below_50= ["AGENDER05", "AGENDER09", "AGENDER14", "AGENDER29"]
for i in probands_below_50:
	data_new = data_new.append(data[data["Proband"]==i])

data = data_new.reset_index(drop=True)
#data = data.reset_index(drop=True)

#which columns to drop (either with ACC or without ACC)
dropcols = []
data = data.drop(columns=dropcols)

# Make list of all ID's in idcolumn
IDlist = set(data["Proband"])

#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

permutation_pvalue = list()
permutation_modelaccuracy = list()
pvalues_binomial = list()

#for loop to iterate through LOSOCV "rounds"

for i in IDlist:
	t0_inner = time.time()
	LOOCV_O = str(i)
	data["Proband"] = data["Proband"].apply(str)
	data_filtered = data[data["Proband"] != LOOCV_O]
	data_cv = data[data["Proband"] == LOOCV_O]

	# define Test data - the person left out of training
	data_test = data_cv.copy()
	X_test_df = data_test.drop(columns=["Label", "Proband"])
	X_test = np.array(X_test_df)
	y_test_df = data_test["Label"]  # This is the outcome variable
	y_test = np.array(y_test_df)

	# define Train data - all other people in dataframe
	data_train = data_filtered.copy()
	X_train_df = data_train.copy()

	#define the model
	model = RandomForestClassifier(random_state=1)

	#define list of indices for inner CV (use again LOSOCV with the remaining subjects)
	IDlist_inner = list(set(X_train_df["Proband"]))
	inner_idxs = []

	X_train_df = X_train_df.reset_index(drop=True)
	for l in range(0, len(IDlist_inner), 2):
		try:
			IDlist_inner[l+1]
		except:
			continue
		else:
			train = X_train_df[(X_train_df["Proband"] != IDlist_inner[l]) & (X_train_df["Proband"] != IDlist_inner[l+1]) ]
			test = X_train_df[(X_train_df["Proband"] == IDlist_inner[l]) | (X_train_df["Proband"] ==  IDlist_inner[l+1]) ]
			add = [train.index, test.index]
			inner_idxs.append(add)

	data_train = data_train.drop(columns=["Proband"]) #drop Proband column
	X_train_df = X_train_df.drop(columns=["Label", "Proband"])
	X_train = np.array(X_train_df)
	y_train_df = data_train["Label"]
	y_train_df = y_train_df.reset_index(drop=True)
	y_train = np.array(y_train_df)  # Outcome variable here

	# define search space
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ["sqrt", "log2", 3]

	space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
				 min_samples_leaf=min_samples_leaf, max_features=max_features)

	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

	# execute search
	result = search.fit(X_train, y_train)

	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_

	#apply permutation test
	## create dataframe which contains all data and delete some stuff
	data_permutation = data.copy()
	data_permutation = data_permutation.reset_index(drop=True)

	## create list which contains indices of train and test samples (differentiate by proband)
	split_permutation = []
	train_permutation = data_permutation[data_permutation["Proband"] != i]
	test_permutation = data_permutation[data_permutation["Proband"] == i]
	add_permutation = [train_permutation.index, test_permutation.index]
	split_permutation.append(add_permutation)

	##Drop some stuff
	#data_permutation = data_permutation.drop(columns=dropcols)

	##Create X and y dataset
	X_permutation = data_permutation.drop(columns=["Label", "Proband"])
	y_permutation = data_permutation["Label"]

	##compute permutation test
	score_model, perm_scores_model, pvalue_model = permutation_test_score(best_model, X_permutation, y_permutation, scoring="accuracy", cv=split_permutation, n_permutations=1000)

	## visualize permutation test restuls
	fig, ax = plt.subplots()
	plt.title("Permutation Test results with Proband " + str(i) + " as Test-Data")
	ax.hist(perm_scores_model, bins=20, density=True)
	ax.axvline(score_model, ls='--', color='r')
	score_label = (f"Score on original\ndata: {score_model:.2f}\n"
				   f"(p-value: {pvalue_model:.3f})")
	ax.text(0.14, 125, score_label, fontsize=12)
	ax.set_xlabel("Accuracy score")
	ax.set_ylabel("Probability")
	plt.savefig('02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_Permutation.png')
	plt.show()

	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)

	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	f1 = f1_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	precision = precision_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")
	recall = recall_score(y_true=y_test, y_pred=yhat, pos_label=1, average="binary")

	#Visualize Confusion Matrix
	mat = confusion_matrix(y_test, yhat)
	sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data" )
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()

	#apply binomial test
	pvalue_binom = binom_test(x=mat[0][0]+mat[1][1], n=len(y_test), p=0.5, alternative='greater')

	#feature importance: compute SHAP values
	explainer = shap.Explainer(best_model)
	shap_values = explainer.shap_values(X_test)
	plt.figure()
	plt.title("Feature Importance for iteration with proband " +str(i) + " as test set")
	shap.summary_plot(shap_values[1], X_test_df, plot_type="bar", show=False)
	#plt.savefig('02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_SHAPFeatureImportance_v2.png')
	fig = plt.gcf()
	fig.savefig('02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_SHAPFeatureImportance.png')
	plt.show()

	# store statistical test results (p-value permutation test, accuracy of that permutation iteration, pvalue binomial test) in list
	permutation_pvalue.append(pvalue_model)
	permutation_modelaccuracy.append(score_model)
	pvalues_binomial.append(pvalue_binom)

	# store the result
	test_proband.append(i)
	outer_results_acc.append(acc)
	outer_results_f1.append(f1)
	outer_results_precision.append(precision)
	outer_results_recall.append(recall)

	# Save the results:
	results = pd.DataFrame()
	results["Test-Proband"] = test_proband
	results["Accuracy"] = outer_results_acc
	results["Accuracy by PermutationTest"] = permutation_modelaccuracy
	results["F1"] = outer_results_f1
	results["Precision"] = outer_results_precision
	results["Recall"] = outer_results_recall
	results["P-Value Permutation Test"] = permutation_pvalue
	results["P-Value Binomial Test"] = pvalues_binomial
	results.to_csv("02_Analysis/LOSOCV_statisticaltests/DF_withACC_probandsbelow50%_proband' + str(i) + 'astest_results_cumulated.csv.")

	# report progress
	t1_inner = time.time()
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
	print("Permutation-Test p-value was " + str(pvalue_model) + " and Binomial Test p-values was " + str(pvalue_binom))
	print("The proband taken as test-data for this iteration was " + str(i))
	print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

t1 = time.time()
print("The whole process has taken so many hours: " + str(((t1 - t0)/60/60)))

# summarize the estimated performance of the model
print('Mean Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('Mean F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Mean Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Mean Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
print("Mean p-value of Permutation Test: %.3f (%.3f)" % (mean(permutation_pvalue), std(permutation_pvalue)))
print("Mean of p-value of Binomial Test: %.3f (%.3f)" % (mean(pvalues_binomial), std(pvalues_binomial)))


os.system("shutdown /h") #hibernate












#endregion

#endregion

#region Tensorflow Decision Forests
## The Tensorflow DF implementation is based on this tutorial: https://towardsdatascience.com/tensorflow-decision-forests-train-your-favorite-tree-based-models-using-keras-875d05a441f

# load data
dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"
sensors_included = "all"
timeperiod = 5 #seconds
label_column_name = "label_human motion - general"
path_df_features = dir_sensorfiles + "data_preparation/features/activity-" + label_column_name + "_highfrequencysensors-" + sensors_included + "_timeperiod-" + str(timeperiod) + " s_featureselection.pkl"
with open(path_df_features, "rb") as f:
    df = pickle.load(f)

participant_column = "device_id"
ESM_event_column_name = "ESM_timestamp"

# decide label-segment: how much time before and after the ESM timestamp should be considered for the label?
label_segment = 30 # in seconds

# delete all data which is not in ESM_event +- label_segment
print("Shape before deleting data: ", df.shape)
df = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=label_segment)) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=label_segment))]
print("Number of records after deleting all data which is not in ESM_event +- label_segment: ", len(df))


# choose only records which are

# balance dataset based on the data exploration
# TODO: IMPROVE BALANCING
## only keep activities with at least 50.000 records
df[label_column_name].value_counts()
df = df[df[label_column_name].isin(df[label_column_name].value_counts()[df[label_column_name].value_counts() > 800].index)]
df[label_column_name].value_counts()

## only keep data from participants with at least 50.000 records
df[participant_column].value_counts()
df = df[df[participant_column].isin(df[participant_column].value_counts()[df[participant_column].value_counts() > 1500].index)]
df[participant_column].value_counts()


#todo: split data into training and testing (by participants)
# Make list of all ID's in idcolumn
IDlist = set(df[participant_column])

#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()


#for loop to iterate through LOSOCV "rounds"
t0 = time.time()
for i in IDlist:
    t0_inner = time.time()
    LOOCV_O = str(i)

	# select training and testing data for this LOSOCV round
	df[participant_column] = df[participant_column].apply(str)
	df_train = df[df[participant_column] != LOOCV_O].copy()
	df_test = df[df[participant_column] == LOOCV_O].copy()

	#delete the columns which are not needed in training
	df_train = df_train.drop(columns=[participant_column, ESM_event_column_name])
	df_test = df_test.drop(columns=[participant_column, ESM_event_column_name])

	#region create tensorflow dataset
	# have to rename label column to "label" for tensorflow
	df_train = df_train.rename(columns={label_column_name: "label"})
	df_test = df_test.rename(columns={label_column_name: "label"})

	# testsection: create dataset only for a few features
	df_train_small = df_train[["label", "accelerometer_x", "accelerometer_y", "accelerometer_z"]]
	#endtestsection

	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train_small, label="label", task=tfdf.keras.Task.CLASSIFICATION).unbatch()
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, label="label", task=tfdf.keras.Task.CLASSIFICATION).unbatch()

	# define the target column and create TensorFlow datasets
	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="label", task=tfdf.keras.Task.CLASSIFICATION).unbatch()
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, label="label", task=tfdf.keras.Task.CLASSIFICATION).unbatch()
	# This method creates a batched Dataset with batch_size=64.
	# We unbatch it to be able to split into smaller subsets and use other batch size

	# define the model
	# the model & tuner definition are based on: https://www.kaggle.com/code/ekaterinadranitsyna/kerastuner-tf-decision-forest/notebook

	# Small subsets of data to use in quick search for optimal hyperparameters.
	train_subset = 10_000  # Number of samples
	valid_subset = 1_000
	batch_size = 256
	train_small_ds = train_ds.take(train_subset).batch(batch_size)
	valid_small_ds = train_ds.skip(train_subset).take(valid_subset).batch(batch_size)

	def build_model(hp):
		"""Function initializes the model and defines search space.
        :param hp: Hyperparameters
        :return: Compiled GradientBoostedTreesModel model
        """
		model = tfdf.keras.GradientBoostedTreesModel(
			num_trees=hp.Int('num_trees', min_value=10, max_value=710, step=25),
			growing_strategy=hp.Choice('growing_strategy', values=['BEST_FIRST_GLOBAL', 'LOCAL']),
			max_depth=hp.Int('max_depth', min_value=3, max_value=16, step=1),
			subsample=hp.Float('subsample', min_value=0.1, max_value=0.95, step=0.05),
			num_threads=4,
			missing_value_policy='GLOBAL_IMPUTATION')  # Default parameter,
		# missing values are replaced by the mean or the most frequent value.

		model.compile(metrics=['accuracy', tf.keras.metrics.AUC()])
		return model

	# build tuner
	# Keras tuner
	tuner = BayesianOptimization(  # Or RandomSearch, or Hyperband
		build_model,
		objective=kt.Objective('val_auc', direction='max'),  # Or 'val_loss'
		max_trials=20,
		directory="/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name + "/models",
		project_name="tuner_" + label_column_name + "_" + str(i)
	)

	# Select the best parameters using a small subset of the train data.
	tuner.search(train_small_ds, epochs=1, validation_data=valid_small_ds)

	# Display the results
	tuner.results_summary()

	# Best model trained on a small subset of the thain data
	# (could be used for predictions as is).
	best_model = tuner.get_best_models(num_models=1)[0]

	# Instantiate untrained model with the best parameters
	# and train on the larger training set.
	best_hp = tuner.get_best_hyperparameters()[0]
	model = tuner.hypermodel.build(best_hp)

	history = model.fit(train_ds, validation_data=valid_ds,
						shuffle=False,
						workers=4, use_multiprocessing=True)

	# Train metrics
	inspect = model.make_inspector()
	inspect.evaluation()

	# Visualize training progress
	logs = inspect.training_logs()

	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.plot([log.num_trees for log in logs],
			 [log.evaluation.accuracy for log in logs])
	plt.xlabel('Number of trees')
	plt.ylabel('Accuracy (out-of-bag)')
	plt.subplot(1, 2, 2)
	plt.plot([log.num_trees for log in logs],
			 [log.evaluation.loss for log in logs])
	plt.xlabel('Number of trees')
	plt.ylabel('Logloss (out-of-bag)')
	plt.show()


	# Model accuracy on the validation set
	evaluation = model.evaluate(valid_ds, return_dict=True)
	for name, value in evaluation.items():
		print(f'{name}: {value:.4f}')

	# Prediction on the test set
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
		test_data[features])
	test_data['claim'] = model.predict(
		test_ds, workers=4, use_multiprocessing=True)

	# Save predicted values for the test set
	test_data[['id', 'claim']].to_csv('submission.csv', index=False)
	test_data[['id', 'claim']].head()

	#region define & train the model
	## it trains using all the features in the training dataset (except the target column)
	# instantiate the model
	model_rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)

# optional step - add evaluation metrics
model_rf.compile(metrics=["mse", "mape"])

# fit the model
# "sys_pipes" is optional and it enables the display of the training logs
with sys_pipes():
  model_rf.fit(x=train_ds)


def build_model(hp):
	model = Sequential()
	model.add(LSTM(hp.Int('input_unit', min_value=32, max_value=512, step=32), return_sequences=True,
				   input_shape=(X_train.shape[1], X_train.shape[2])))
	for i in range(hp.Int('n_layers', 1, 4)):
		model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32), return_sequences=True))
	model.add(LSTM(hp.Int('layer_2_neurons', min_value=32, max_value=512, step=32)))
	model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))
	model.add(Dense(y_train.shape[1],
					activation=hp.Choice('dense_activation', values=['relu', 'sigmoid'], default='relu')))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
	return model


tuner = RandomSearch(
	build_model,
	objective='mse',
	max_trials=2,
	executions_per_trial=1,
	overwrite=True
)

tuner.search(
	x=X_train,
	y=y_train,
	epochs=20,
	batch_size=128,
	validation_data=(X_test, y_test),
)

best_model = tuner.get_best_models(num_models=1)[0]

### evaluate model
yhat = best_model.predict(X_test)

### save model
save_model(best_model,
		   "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included + "_test_proband-" + str(
			   i) + "_LSTM.h5")

#endregion

#region evaluate the model
evaluation = model_rf.evaluate(test_ds, return_dict=True)

print(evaluation)
print(f"MSE: {evaluation['mse']:.2f}")
print(f"RMSE: {math.sqrt(evaluation['mse']):.2f}")
print(f"MAPE: {evaluation['mape']:.2f}")

# get the out-of-bag score
model_rf.make_inspector().evaluation()

# plot the RMSE during training
logs = model_rf.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.title("RMSE vs number of trees")

plt.show()

#endregion

#region model interpretation: feature importance
# plot the first tree, restricted to depth of 3
tfdf.model_plotter.plot_model_in_colab(model_rf, tree_idx=0, max_depth=3)

# plot the feature importance
# get the feature importance
feature_importance = model_rf.make_inspector().variable_importances()
# inspect the features used in the model
model_rf.make_inspector().features()
tfdf.model_plotter.plot_feature_importances(feature_importance)
#endregion
#endregion

