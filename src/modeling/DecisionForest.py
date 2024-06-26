
#region import
import pickle
#import tensorflow_decision_forests as tfdf

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt

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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# import classification report
from sklearn.metrics import classification_report

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

# visualization
import seaborn as sns

# for keeping track
from tqdm import tqdm



try:
  from wurlitzer import sys_pipes
except:
  from colabtools.googlelog import CaptureLog as sys_pipes
#endregion

# build DF class including the options to enable and disable permutation and binomial test as well as LOSOCV
class DecisionForest:

	#DF including LOSOCV, hyperparameter tuning, permutation test, etc
	#Note: since 20.02., the index of the df needs to be reset before using this function
	def DF_sklearn(df, label_segment, label_column_name, n_permutations, path_storage, parameter_tuning,
				   confusion_matrix_order, title_confusion_matrix, title_feature_importance_grid, grid_search_space = None, feature_importance = None,
				   parameter_set = "default", combine_participants = False):
		"""
		Builds a decision forest using the sklearn library
		:param df:
		:param label_segment: has to be in seconds
		:return:
		"""

		# initialize metrics list for LOSOCV metrics
		timeperiod_list = []
		label_list = []
		acc_final_list = []
		bal_acc_final_list = []
		f1_final_list = []
		precision_final_list = []
		recall_final_list = []

		#reset index
		#df = df.reset_index(drop = True)

		# convert timestamp and ESM_timestamp to datetime
		# check if ESM_timestamp is in the df (since it isnt in the "location" datasets)
		if "ESM_timestamp" in df.columns:
			df["timestamp"] = pd.to_datetime(df["timestamp"])
			df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

		# initialize
		test_proband = list()
		outer_results_acc = list()
		outer_results_acc_balanced = list()
		outer_results_f1 = list()
		outer_results_precision = list()
		outer_results_recall = list()
		best_parameters = list()
		outer_results_best_params = list()

		permutation_pvalue = list()
		permutation_modelaccuracy = list()
		pvalues_binomial = list()

		#test_proband_array = np.empty([0])
		#y_test_array = np.empty([0])
		#y_pred_array = np.empty([0])
		df_labels_predictions = pd.DataFrame()

		shap_values_dict = dict()

		# Make list of all participants
		IDlist = set(df["device_id"])
		# make different IDlist in case a simple train-test split should be used for outer iteration (for parameter tuning, LOSOCV will still be used)
		if combine_participants == True:
			IDlist = set(df["device_id_traintest"])


		# for loop to iterate through LOSOCV "rounds"
		counter = 1
		num_participants = len(IDlist)
		for i in tqdm(IDlist):
			print("Start with participant " + str(i) + " as test participant")
			t0_inner = time.time()

			#split data into train and test
			LOOCV_O = i
			df_train = df[df["device_id"] != LOOCV_O]
			df_test = df[df["device_id"] == LOOCV_O]
			if combine_participants == True:
				if i == 1:
					print("This iteration will be skipped since train-set is test-set")
					continue
				df_train = df[df["device_id_traintest"] != LOOCV_O]
				df_test = df[df["device_id_traintest"] == LOOCV_O]

			# define Test data - the person left out of training
			##  check if ESM_timestamp is in the df (since it isnt in the "location" datasets)
			if "ESM_timestamp" in df_test.columns:
				X_test_df = df_test.drop(columns=[label_column_name, "device_id", "ESM_timestamp",
													"timestamp"])  # add sensor_timestamp here as soon as available
			else:
				X_test_df = df_test.drop(columns=[label_column_name, "device_id"])

			if combine_participants == True:
				X_test_df = df_test.drop(columns=[label_column_name, "device_id", "ESM_timestamp", "device_id_traintest",
												  "timestamp"])  # add sensor_timestamp here as soon as available

			X_test = np.array(X_test_df)
			y_test_df = df_test[label_column_name]  # This is the outcome variable
			y_test = np.array(y_test_df)

			# jump over this iteration if y_test contains only one class
			if len(set(y_test)) == 1:
				print("y_test contains only one class")
				continue

			# define Train data - all other people in dataframe
			X_train_df = df_train.copy()

			# define the model
			if parameter_set == "default":
				model = RandomForestClassifier(random_state=11)
			else:
				model = RandomForestClassifier(**parameter_set)


			# define list of indices for inner CV for the GridSearch (use again LOSOCV with the remaining subjects)
			# here a "Leave Two Subejcts Out" CV is used!
			if parameter_tuning == "yes":
				IDlist_inner = list(set(X_train_df["device_id"]))
				inner_idxs = []
				X_train_df = X_train_df.reset_index(drop=True)
				for l in range(0, len(IDlist_inner), 2):
					try:
						IDlist_inner[l + 1]
					except:
						continue
					else:
						train = X_train_df[
							(X_train_df["device_id"] != IDlist_inner[l]) & (X_train_df["device_id"] != IDlist_inner[l + 1])]
						test = X_train_df[
							(X_train_df["device_id"] == IDlist_inner[l]) | (X_train_df["device_id"] == IDlist_inner[l + 1])]
						add = [train.index, test.index]
						inner_idxs.append(add)

			# drop participant column
			df_train = df_train.drop(columns=["device_id"])
			if combine_participants == True:
				df_train = df_train.drop(columns=["device_id_traintest"])

			# drop other columns
			## check if ESM_timestamp is in the df (since it isnt in the "location" datasets)
			if "ESM_timestamp" in X_train_df.columns:
				X_train_df = X_train_df.drop(columns=[label_column_name, "device_id", "ESM_timestamp", "timestamp"])  # add sensor_timestamp later on here
			else:
				X_train_df = X_train_df.drop(columns=[label_column_name, "device_id"])

			if combine_participants == True:
				X_train_df = X_train_df.drop(columns=[ "device_id_traintest"])  # add sensor_timestamp later on here

			#X_train = np.array(X_train_df)
			y_train_df = df_train[label_column_name]  # This is the outcome variable
			y_train_df = y_train_df.reset_index(drop=True)
			#y_train = np.array(y_train_df)  # Outcome variable here

			# parameter tuning: only do, if parameter_tuning is set to True
			if parameter_tuning == "yes":

				# define search
				print("Start parameter tuning")
				search = GridSearchCV(model, grid_search_space, scoring='balanced_accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

				# execute search
				print("Start fitting model with parameter tuning...")
				result = search.fit(X_train_df, y_train_df)
				print("Model fitted.")
				print("Best: %f using %s" % (result.best_score_, result.best_params_))

				# get the best performing model fit on the whole training set
				best_model = result.best_estimator_
				best_params = result.best_params_
				parameter_tuning_active = "yes"

			# if parameter tuning is set to False, use the default parameters
			else:
				print("Start fitting model without parameter tuning...")
				best_model = model.fit(X_train_df, y_train_df)
				parameter_tuning_active = "no"

			# save the model
			with open(path_storage + label_column_name + "_timeperiod_around_event-" + str(
					label_segment) + "_parameter_tuning-"+ parameter_tuning_active +"_test_proband-" + str(
					i) + "_model.sav", 'wb') as f:
				pickle.dump(best_model, f)
			print("Best model: ", best_model)

			# apply permutation test
			print("Start permutation test...")
			## create dataframe which contains all data and delete some stuff
			data_permutation = df.copy()
			data_permutation = data_permutation.reset_index(drop=True)

			## create list which contains indices of train and test samples (differentiate by proband)
			split_permutation = []
			train_permutation = data_permutation[data_permutation["device_id"] != i]
			test_permutation = data_permutation[data_permutation["device_id"] == i]
			if combine_participants == True:
				train_permutation = data_permutation[data_permutation["device_id_traintest"] != i]
				test_permutation = data_permutation[data_permutation["device_id_traintest"] == i]
			add_permutation = [train_permutation.index, test_permutation.index]
			split_permutation.append(add_permutation)

			##Drop some stuff
			# data_permutation = data_permutation.drop(columns=dropcols)

			##Create X and y dataset
			### check if ESM_timestamp is in the df (since it isnt in the "location" datasets)
			if "ESM_timestamp" in data_permutation.columns:
				X_permutation = data_permutation.drop(
					columns=[label_column_name, "device_id", "timestamp", "ESM_timestamp"])
			else:
				X_permutation = data_permutation.drop(columns=[label_column_name, "device_id"])

			if combine_participants == True:
				X_permutation = data_permutation.drop(
					columns=[label_column_name, "device_id", "timestamp", "ESM_timestamp", "device_id_traintest"])
			y_permutation = data_permutation[label_column_name]

			##compute permutation test
			score_model, perm_scores_model, pvalue_model = permutation_test_score(best_model, X_permutation,
																				  y_permutation, scoring="balanced_accuracy",
																				  cv=split_permutation,
																				  n_permutations=n_permutations,
																				  n_jobs=-1)
			print("Permutation test done.")

			## visualize permutation test results
			fig, ax = plt.subplots(figsize=(10, 5))
			plt.title("Results of Permutation Test (Participant " + str(i) + " as Test-Data)")
			# create histogram with sns histplot
			sns.histplot(perm_scores_model, bins=50, stat="density", ax=ax)

			#add vertical line for score on original data
			#ax.hist(perm_scores_model, bins=(n_permutations/2), density=True)
			ax.axvline(score_model, ls='--', color='r')
			#score_label = (f"Score on original\ndata: {score_model:.2f}\n"
			#			   f"(p-value: {pvalue_model:.3f})")
			# put text in upper right corner
			#ax.text(0.9, 0.8, score_label, transform=ax.transAxes)
			plt.tight_layout()
			ax.set_xlabel("Balanced Accuracy")
			ax.set_ylabel("Count")
			plt.show()

			plt.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning_active + "_test_proband-" + str(
					i) + "_Permutation.png", bbox_inches='tight', dpi = 300)
			# plt.show()


			# evaluate model on the hold out dataset
			## create predictions
			print("Start evaluating model...")
			yhat = best_model.predict(X_test_df)

			## create probabilities for each class
			probabilities = best_model.predict_proba(X_test_df)
			df_probabilities = pd.DataFrame(probabilities, columns=best_model.classes_)
			## set index of X_test_df to the index of df_probabilities
			df_probabilities = df_probabilities.set_index(X_test_df.index)

			# evaluate the model
			acc = accuracy_score(y_test_df, yhat)
			acc_balanced = balanced_accuracy_score(y_test_df, yhat)
			print('Balanced Accuracy: %.3f' % acc_balanced)
			f1 = f1_score(y_true=y_test_df, y_pred=yhat, average="weighted")
			precision = precision_score(y_true=y_test_df, y_pred=yhat, average="weighted")
			recall = recall_score(y_true=y_test_df, y_pred=yhat, average="weighted")
			# TODO document this: why I am using balanced accuracy (https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)
			# and weighted f1, precision and recall (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html=

			# convert categorical labels to numerical labels and save the mapping for later visualization
			#convert y_test to pandas series
			y_test_confusionmatrix = pd.Series(y_test_df)
			y_test_confusionmatrix = y_test_confusionmatrix.astype('category')
			label_mapping = dict(enumerate(y_test_confusionmatrix.cat.categories))
			#y_test_confusionmatrix = y_test_confusionmatrix.cat.codes
			#y_test_confusionmatrix = y_test_confusionmatrix.to_numpy()
			# also convert yhat to numerical labels using same mapping
			yhat_confusionmatrix = pd.Series(yhat)
			yhat_confusionmatrix = yhat_confusionmatrix.astype('category')
			label_mapping2 = dict(enumerate(yhat_confusionmatrix.cat.categories))
			#yhat_confusionmatrix = yhat_confusionmatrix.cat.codes
			#yhat_confusionmatrix = yhat_confusionmatrix.to_numpy()

			# create joint label mapping which is == confusion_matrix_order except the elements which are not in label_mapping or label_mapping2
			label_mapping_joint = list(set(label_mapping.values()) | set(label_mapping2.values()))
			label_mapping_confusion_matrix = confusion_matrix_order.copy()
			for key in label_mapping_confusion_matrix:
				if key not in label_mapping_joint:
					#delete from list
					label_mapping_confusion_matrix.remove(key)

			# Visualize Confusion Matrix with absolute values
			fig, ax = plt.subplots(figsize=(10, 5))
			#plt.gcf().subplots_adjust(bottom=0.15)
			mat = confusion_matrix(y_test_confusionmatrix, yhat_confusionmatrix, labels = label_mapping_confusion_matrix)
			sns.heatmap(mat, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, fmt='d', cbar=False, linewidths=0.2)
			plt.title('Confusion Matrix Absolute Values with Test-Proband ' + str(i) )
			plt.suptitle('Balanced Accuracy: {0:.3f}'.format(acc_balanced), fontsize=16)
			# add xticks and yticks from label_mapping (which is a dictionary)
			tick_marks = np.arange(len(label_mapping_confusion_matrix)) + 0.5
			plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
			plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')
			#plt.show()
			plt.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning_active + "_test_proband-" + str(
					i) + "_ConfusionMatrix_absolute.png", bbox_inches="tight")


			# visualize confusion matrix with percentages
			# Get and reshape confusion matrix data
			matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
			plt.figure(figsize=(16, 7))
			sns.set(font_scale=1.4)
			sns.heatmap(matrix, annot=True, annot_kws={'size': 10},cmap=plt.cm.Greens, linewidths=0.2)
			# Add labels to the plot
			tick_marks = np.arange(len(label_mapping_confusion_matrix)) + 0.5
			plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
			plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')
			plt.title('Confusion Matrix Relative Values with Test-Proband ' + str(i) )
			plt.suptitle('Balanced Accuracy: {0:.3f}'.format(acc_balanced), fontsize=16)
			plt.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning_active + "_test_proband-" + str(
					i) + "_ConfusionMatrix_percentages.png", bbox_inches="tight")
			#plt.show()

			# apply binomial test
			print("Start binomial test...")
			# calculate sum of diagonal axis of confusion matrix
			sum_diagonal = 0
			for l in range(len(mat)):
				sum_diagonal += mat[l][l]
			# calculate number of classes in y_test
			classes = len(np.unique(y_test))
			pvalue_binom = binom_test(x=sum_diagonal, n=len(y_test), p=(1/classes), alternative='greater')
			print("P-value binomial test: ", pvalue_binom)
			print("Binomial test done.")

			# feature importance: compute SHAP values
			#TODO include here rather DF explanatory variables than SHAP values
			if feature_importance == "shap":
				print("Start computing SHAP values...")
				explainer = shap.Explainer(best_model)
				shap_values = explainer.shap_values(X_test_df)

				# Compute the absolute averages of the SHAP values for each sample and feature across all classes
				## Explanation: in shap_values are three dimensions: classes x samples x features
				## In absolute_average_shape_values, the absolute average for each feature and sample over the classes
				## is computed. The result is a matrix with two dimensions: samples x features
				absolute_average_shap_values = np.mean(np.abs(shap_values), axis=0)

				fig, ax = plt.subplots(figsize=(10, 5))
				plt.title("Feature Importance for iteration with proband " + str(i) + " as test set")
				shap.summary_plot(absolute_average_shap_values, X_test_df.iloc[0:0], plot_type="bar", show=False,
								  plot_size=(20, 10))
				plt.show()
				fig.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(
					label_segment) + "_parameter_tuning-" + parameter_tuning_active + "_test_proband-" + str(
					i) + "_SHAPFeatureImportance.png")

				# store the SHAP values (in order to get combined SHAP values for the whole LOSOCV in the end)
				## create dictionary which contains as key the proband number and as values the SHAP values and the best_model.classes_
				shap_values_dict[i] = [shap_values, best_model.classes_]

			# store statistical test results (p-value permutation test, accuracy of that permutation iteration, pvalue binomial test) in list
			print("Start storing statistical test results...")
			permutation_pvalue.append(pvalue_model)
			permutation_modelaccuracy.append(score_model)
			pvalues_binomial.append(pvalue_binom)

			# store the resulted metrics
			test_proband.append(i)
			outer_results_acc.append(acc)
			outer_results_acc_balanced.append(acc_balanced)
			outer_results_f1.append(f1)
			outer_results_precision.append(precision)
			outer_results_recall.append(recall)
			if parameter_tuning == "yes":
				outer_results_best_params.append(best_params)

			# store the y_test and yhat and probabilities for the final accuracy computation using concatenation
			## transform y_test_df into dataframe and add column with y_pred and column with proband number

			# For each class, label the sample with the highest probability as that class, and set all others to NaN
			df_labels_predictions_intermediate = pd.DataFrame(y_test_df)
			# concatenate df_probabilities with df_labels_predictions_intermediate based on index
			df_labels_predictions_intermediate = pd.concat([df_labels_predictions_intermediate, df_probabilities], axis=1)
			df_labels_predictions_intermediate = df_labels_predictions_intermediate.rename(columns={label_column_name: "y_test"})
			df_labels_predictions_intermediate["y_pred"] = yhat
			df_labels_predictions_intermediate["test_proband"] = i
			## concatenate the dataframes
			df_labels_predictions = pd.concat([df_labels_predictions, df_labels_predictions_intermediate])


			# report progress
			t1_inner = time.time()
			print("Time for participant " + str(counter) + "/" + str(num_participants) + " has been " + str((t1_inner - t0_inner)/60) + " minutes.")
			counter += 1
		# print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
		# print("Permutation-Test p-value was " + str(pvalue_model) + " and Binomial Test p-values was " + str(pvalue_binom))
		# print("The proband taken as test-data for this iteration was " + str(i))
		# print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

		# Save the resulting metrics:
		results_LOSOCV = pd.DataFrame()
		results_LOSOCV["Test-Proband"] = test_proband
		results_LOSOCV["Accuracy"] = outer_results_acc
		results_LOSOCV["Balanced Accuracy"] = outer_results_acc_balanced
		results_LOSOCV["Accuracy by PermutationTest"] = permutation_modelaccuracy
		results_LOSOCV["F1"] = outer_results_f1
		results_LOSOCV["Precision"] = outer_results_precision
		results_LOSOCV["Recall"] = outer_results_recall
		results_LOSOCV["P-Value Permutation Test"] = permutation_pvalue
		results_LOSOCV["P-Value Binomial Test"] = pvalues_binomial
		# add best parameters if parameter tuning was active
		if parameter_tuning == "yes":
			results_LOSOCV["Best Parameters"] = outer_results_best_params
		# add label column name as column
		results_LOSOCV["Label Column Name"] = label_column_name
		# add seconds around event as column
		results_LOSOCV["Seconds around Event"] = label_segment
		# add timeperiod of features as column
		results_LOSOCV.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning + "_results_LOSOCV.csv")

		# save the y_test and yhat for the final accuracy computation
		df_labels_predictions.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning +  "_results_labelsRealAndPredicted.csv")

		# TODO: document this!
		# compute the final metrics for this LOSOCV: here the cumulated y_test and yhat are used in order to account
		# for the fact that some test-participants have more data than others AND that some participants more label-classes
		# were present then for other participants
		balanced_accuracy_overall = balanced_accuracy_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"])
		results_overall = pd.DataFrame()
		results_overall["Label"] = [label_column_name]
		results_overall["Seconds around Event"] = [label_segment]
		results_overall["Balanced Accuracy"] = [balanced_accuracy_overall]
		results_overall["Accuracy"] = [accuracy_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"])]
		results_overall["F1"] = [f1_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"], average="macro")]
		results_overall["Precision"] = [precision_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"], average="macro")]
		results_overall["Recall"] = [recall_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"], average="macro")]

		# visualize confusion matrix for all y_test and y_pred data
		# convert categorical labels to numerical labels and save the mapping for later visualization
		# convert y_test to pandas series
		y_test_confusionmatrix = df_labels_predictions["y_test"]
		y_test_confusionmatrix = y_test_confusionmatrix.astype('category')
		label_mapping = dict(enumerate(y_test_confusionmatrix.cat.categories))
		#y_test_confusionmatrix = y_test_confusionmatrix.cat.codes
		#y_test_confusionmatrix = y_test_confusionmatrix.to_numpy()

		# also convert yhat to numerical labels using same mapping
		yhat_confusionmatrix = df_labels_predictions["y_pred"]
		yhat_confusionmatrix = yhat_confusionmatrix.astype('category')
		label_mapping2 = dict(enumerate(yhat_confusionmatrix.cat.categories))
		#yhat_confusionmatrix = yhat_confusionmatrix.cat.codes
		#yhat_confusionmatrix = yhat_confusionmatrix.to_numpy()

		# create joint label mapping which is == confusion_matrix_order except the elements which are not in label_mapping or label_mapping2
		label_mapping_joint = list(set(label_mapping.values()) | set(label_mapping2.values()))
		label_mapping_confusion_matrix = confusion_matrix_order.copy()
		for key in label_mapping_confusion_matrix:
			if key not in label_mapping_joint:
				# delete from list
				label_mapping_confusion_matrix.remove(key)

		# Visualize Confusion Matrix with absolute values
		plt.figure(figsize=(16, 7))
		sns.set(font_scale=1.4)
		mat = confusion_matrix(y_test_confusionmatrix, yhat_confusionmatrix, labels = label_mapping_confusion_matrix)
		sns.heatmap(mat, square=True, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, fmt='d', cbar=False,
					linewidths=0.2)

		plt.title("Confusion Matrix Absolute Values for All Participants")
		plt.suptitle('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_overall), fontsize=16)
		# add xticks and yticks from label_mapping (which is a dictionary)
		tick_marks = np.arange(len(label_mapping_confusion_matrix)) + 0.5
		plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
		plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')
		#plt.show()
		plt.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(
			label_segment) + "_parameter_tuning-" + parameter_tuning_active + "_ConfusionMatrix_absolute.png",
					bbox_inches="tight")
		#

		# visualize confusion matrix with percentages
		# Get and reshape confusion matrix data
		matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
		plt.figure(figsize=(16, 7))
		sns.set(font_scale=1.4)
		sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
		# Add labels to the plot
		plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
		plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')
		plt.title("Confusion Matrix Relative Values for All Participants")
		plt.suptitle('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_overall), fontsize=16)
		plt.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(
			label_segment) + "_parameter_tuning-" + parameter_tuning_active +
					"_ConfusionMatrix_percentages.png", bbox_inches="tight")
		#plt.show()

		# visualize confusion matrix with percentages and absolute values combined
		matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
		matrix_abs = mat.astype('float')
		plt.figure(figsize=(16, 7))
		sns.set(font_scale=1.4)
		sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
		for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
			if matrix[i, j] > 0.5:
				text_color = 'white'
			else:
				text_color = 'black'
			plt.text(j + 0.5, i + 0.75, '({0})'.format(int(matrix_abs[i, j])), ha='center', va='center', fontsize=16,
					 color=text_color)
		plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
		plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')
		plt.title(title_confusion_matrix, fontsize=16)
		#plt.suptitle('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_overall), fontsize=10)
		#plt.show()
		plt.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(
			label_segment) + "_parameter_tuning-" + parameter_tuning_active +
					"_ConfusionMatrix_percentages_absolute.png", bbox_inches="tight", dpi=500)


		# visualize SHAP values for whole LOSOCV
		## save raw shap values "shap_values_dict" to pickle
		with open (path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-" + parameter_tuning_active + "_SHAPValues_AllLOSOCVIterations.pkl", "wb") as f:
			pickle.dump(shap_values_dict, f)

		## the a list of shap value lists: each sub-list are the joint shap values for one class; classes are ordered according to confusion_matrix_order
		shap_values_list_joint = []
		for i in range(len(confusion_matrix_order)):
			shap_values_list_joint.append(np.empty((0, len(X_test_df.columns)), float))

		for key in shap_values_dict:
			# iterate through shap_values_dict[key][1]
			classes = shap_values_dict[key][1]
			counter = 0
			for single_class in classes:
				# find out at which place this class is in confusion_matrix_order
				index = confusion_matrix_order.index(single_class)
				# append corresponding shap values to shap_values_list
				shap_values_list_joint[index] = np.append(shap_values_list_joint[index], shap_values_dict[key][0][counter], axis=0)
				counter += 1

		## visualize the shap values for whole LOSOCV and each individual class
		for single_class in confusion_matrix_order:
			index = confusion_matrix_order.index(single_class)

			fig, ax = plt.subplots(figsize=(10, 5))
			plt.title("Feature Importance for LOSOCV Combined For Class " + single_class)
			shap.summary_plot(shap_values_list_joint[index], X_test_df.iloc[0:0], plot_type="bar", show=False,
							  plot_size=(20, 10), max_display=5)
			plt.xlabel("Average of Absolute SHAP Values")
			#plt.show()
			# replace in "single_class" all "/" with "-" (otherwise the file cannot be saved)
			single_class = single_class.replace("/", "-")
			fig.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-" +
						parameter_tuning_active + "_SHAPValues_class-" + single_class + "_AllLOSOCVIterations.png", bbox_inches="tight")

		## visualize the shap values for whole LOSOCV and each individual class in a grid
		# determine the number of classes and calculate the number of rows and columns for the grid
		num_classes = len(confusion_matrix_order)
		num_cols = min(num_classes, 2)  # set the maximum number of columns to 3
		num_rows = (num_classes - 1) // num_cols + 1

		# create the grid of subplots
		fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
		plt.subplots_adjust(left=0.1, bottom=0.1, right=10000.9, top=0.9, wspace=0.5, hspace=0.5)

		# create title for whole figure
		fig.suptitle(title_feature_importance_grid, fontsize=16)

		# iterate over the classes and plot the SHAP summary plot on the appropriate subplot
		for i, single_class in enumerate(confusion_matrix_order):
			index = confusion_matrix_order.index(single_class)

			# determine the row and column indices for the current subplot
			row_idx = i // num_cols
			col_idx = i % num_cols

			# create a new subplot on the current grid location
			plt.subplot(num_rows, num_cols, i + 1)
			plt.title(single_class)
			shap.summary_plot(shap_values_list_joint[index], X_test_df.iloc[0:0], plot_type="bar", show=False,
							  plot_size=(20, 10), max_display=5)
			plt.xlabel("Average of Absolute SHAP Values")
			#plt.tick_params(axis='y', which='major', labelsize=8)
			#plt.tick_params(axis='y', which='major', labelrotation=45)

			# replace in "single_class" all "/" with "-" (otherwise the file cannot be saved)
			single_class = single_class.replace("/", "-")

		# adjust the spacing between the subplots and show the figure
		fig.tight_layout()
		plt.show()
		#save
		fig.savefig(path_storage + label_column_name + "_timeperiod_around_event-" + str(
			label_segment) + "_parameter_tuning-" +
					parameter_tuning_active + "_SHAPValues_AllClassesInGrid_AllLOSOCVIterations.png",
					 dpi=600)


		## visualize the shap values for whole LOSOCV and all classes combined
		### take the absolute averages of the shap values over all classes
		shap_values_averaged_samples = [np.mean(np.abs(shap_values_list_joint[i]), axis=0) for i in range(len(confusion_matrix_order))]
		shap_values_averaged_samples_classes = np.mean(np.abs(shap_values_averaged_samples), axis=0)

		### have to reshape in order to match need of summary_plot: (n_samples, n_features); create artificial samples by duplicating the shap values
		### NOTE: this wasnt possible before because sample values from different LOSOCV iteration can´t be averaged, as there are
		### sometimes different number of sample values for each class
		shap_values_averaged_samples_classes_includingartificialsamples = np.repeat(
			shap_values_averaged_samples_classes[np.newaxis, :], 2, axis=0)

		fig, ax = plt.subplots(figsize=(10, 5))
		plt.title("Feature Importance for whole LOSOCV combined (absolute average SHAP values)")
		shap.summary_plot(shap_values_averaged_samples_classes_includingartificialsamples, X_test_df.iloc[0:0], plot_type="bar", show=False,
						  plot_size=(20, 10))
		#plt.show()
		fig.savefig(
			path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-" +
			parameter_tuning_active + "_SHAPValues_all-classes-combined_AllLOSOCVIterations.png",
			bbox_inches="tight")


		# TODO include here also a binomial test for the final accuracy
		# TODO think how to apply permutation test here
		return results_overall, df_labels_predictions

	#DF only used for training the final deployment model: train on all data without testing
	## Note: this doesnt include LOSOCV, hyperparameter tuning, training-testing, permutation/binomial tests, etc
	def DF_sklearn_deployment(df, label_segment, label_column_name, parameter_set, path_storage):
		# Transform data needed for DF
		## reset index
		df = df.reset_index(drop=True)
		## convert timestamp and ESM_timestamp to datetime
		df["timestamp"] = pd.to_datetime(df["timestamp"])
		df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

		## drop columns
		X_train_df =  df.drop(columns=[label_column_name, "device_id", "ESM_timestamp", "timestamp"])  # add sensor_timestamp later on here
		y_train_df = df[label_column_name]  # This is the outcome variable
		y_train_df = y_train_df.reset_index(drop=True)

		#define model and train
		model = RandomForestClassifier(**parameter_set)
		print("Start fitting model...")
		best_model = model.fit(X_train_df, y_train_df)


		return best_model


#region OUTDATED: Tensorflow Decision Forests and other methods (not used anymore)

t0 = time.time()

timeperiods = [30, 10, 5, 2] #seconds
# creat list of labels
## the following implementation is based on AGENDER 2.0 project
#Pipeline: using LOSOCV & permutation test
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_all_transformed_labeled.csv")
label_columns = [col for col in df_esm.columns if "label" in col]

#set general variables
dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"
sensors_included = "all"


#region train DF for all timeperiods and activities. Apply LOSOCV, permutation test & binomial test

timeperiod_list_outside = []
label_list_outside = []
acc_final_list_outside = []
bal_acc_final_list_outside = []
f1_final_list_outside = []
precision_final_list_outside = []
recall_final_list_outside = []

for timeperiod in timeperiods:
	t0_timeperiod = time.time()

	#initialize metrics list for LOSOCV metrics
	timeperiod_list = []
	label_list = []
	acc_final_list = []
	bal_acc_final_list = []
	f1_final_list = []
	precision_final_list = []
	recall_final_list = []


	# iterate through all labels
	for label_column_name in label_columns:
		# start timer
		t0_label = time.time()

		print("Start of timeperiod: " + str(timeperiod) + "s and label: " + label_column_name)

		# jump over the ones which didn´t work

		# jump over the ones which exist already

		path_df_features = dir_sensorfiles + "data_preparation/features/activity-" + label_column_name + "_highfrequencysensors-" + sensors_included + "_timeperiod-" + str(timeperiod) + " s_featureselection.pkl"
		# check if file doesnt exist
		if not os.path.exists(path_df_features):
			print("File not found: " + path_df_features)
			continue

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

		# delete all data which is not in ESM_event +- label_segment
		if timeperiod == 30:
			label_segment = 31  # the segment (in seconds) around ESM_event that is used for classification (before and after)
		elif timeperiod == 10:
			label_segment = 11
		elif timeperiod == 5:
			label_segment = 11
		elif timeperiod == 2:
			label_segment = 11
		print("Shape before deleting data: ", df.shape)

		# convert sensor_timestamp and ESM_timestamp to datetime
		df["sensor_timestamp"] = pd.to_datetime(df["sensor_timestamp"])
		df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

		# select only data which are in the label_segment around ESM_event
		df = df[(df['sensor_timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=label_segment)) & (df['sensor_timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=label_segment))]
		print("Number of records after deleting all data which is not in ESM_event +- label_segment: ", len(df))


		# Make list of all ID's in idcolumn
		IDlist = set(df["device_id"])
		# shuffle IDlist
		#IDlist = list(IDlist)
		#random.shuffle(IDlist)

		#initialize
		test_proband = list()
		outer_results_acc = list()
		outer_results_acc_balanced = list()
		outer_results_f1 = list()
		outer_results_precision = list()
		outer_results_recall = list()
		best_parameters = list()

		permutation_pvalue = list()
		permutation_modelaccuracy = list()
		pvalues_binomial = list()

		test_proband_array = np.empty([0])
		y_test_array = np.empty([0])
		y_pred_array = np.empty([0])

		#for loop to iterate through LOSOCV "rounds"
		counter = 0
		for i in tqdm(IDlist):
			# only do LOSOCV for 5 probands (due to computational time)
			counter += 1
			if counter < 5 or counter > 13:
				print("First 5 probands are not used for LOSOCV, we are hopping above proband " + str(i))
				continue

			t0_inner = time.time()
			LOOCV_O = str(i)
			df["device_id"] = df["device_id"].apply(str)
			data_filtered = df[df["device_id"] != LOOCV_O]
			data_cv = df[df["device_id"] == LOOCV_O]

			# define Test data - the person left out of training
			data_test = data_cv.copy()
			X_test_df = data_test.drop(columns=[label_column_name, "device_id",  "ESM_timestamp", "sensor_timestamp"]) #add sensor_timestamp here as soon as available
			X_test = np.array(X_test_df)
			y_test_df = data_test[label_column_name]  # This is the outcome variable
			y_test = np.array(y_test_df)

			# jump over this iteration if y_test contains only one class
			if len(set(y_test)) == 1:
				print("y_test contains only one class")
				continue

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
			X_train_df = X_train_df.drop(columns=[label_column_name, "device_id", "ESM_timestamp", "sensor_timestamp"]) # add sensor_timestamp later on here
			X_train = np.array(X_train_df)
			y_train_df = data_train[label_column_name]  # This is the outcome variable
			y_train_df = y_train_df.reset_index(drop=True)
			y_train = np.array(y_train_df)  # Outcome variable here

			# define search space
			#n_estimators = [100, 300, 500, 800, 1200]
			#max_depth = [5, 8, 15, 25, 30]
			#min_samples_split = [2, 5, 10, 15, 100]
			#min_samples_leaf = [1, 2, 5, 10]
			#max_features = ["sqrt", "log2", 3]

			#test search space
			#TODO Fine-Tuning kann auch später passieren; default Werte reichen fürs erste aus
			#TODO check welche Parameter ausgewählt werden, adapt Search Space
			n_estimators = [50, 100, 500, 800 ] #default 500
			max_depth = [2, 5, 15]
			min_samples_split = [2, 10, 30]
			min_samples_leaf = [1, 5]
			max_features = ["sqrt", 3, "log2", 20]  # double check; 3 ergibt keinen Sinn bei so vielen Features

			space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
						 min_samples_leaf=min_samples_leaf, max_features=max_features)

			# define search
			search = GridSearchCV(model, space, scoring='accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

			# execute search
			print("Start fitting model...")
			result = search.fit(X_train, y_train)
			print("Model fitted.")
			print("Best: %f using %s" % (result.best_score_, result.best_params_))

			# get the best performing model fit on the whole training set
			best_model = result.best_estimator_
			# save the best model
			pickle.dump(model, open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_feature_timeperiod-" + str(timeperiod) + "_timeperiod_around_event-" + str(label_segment) + "_test_proband-" + str(i) + "_model.sav", "wb"))
			print("Best model: ", best_model)

			#apply permutation test
			print("Start permutation test...")
			## create dataframe which contains all data and delete some stuff
			data_permutation = df.copy()
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
			print("Permutation test done.")

			## visualize permutation test restuls
			fig, ax = plt.subplots(figsize=(10, 5))
			plt.title("Permutation Test results with Proband " + str(i) + " as Test-Data")
			ax.hist(perm_scores_model, bins=20, density=True)
			ax.axvline(score_model, ls='--', color='r')
			score_label = (f"Score on original\ndata: {score_model:.2f}\n"
						   f"(p-value: {pvalue_model:.3f})")
			ax.text(0.14, 125, score_label, fontsize=16)
			ax.set_xlabel("Accuracy score")
			ax.set_ylabel("Probability")
			plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_feature_timeperiod-" + str(timeperiod) + "_timeperiod_around_event-" + str(label_segment) + "_test_proband-" + str(i) + "_Permutation.png")
			#plt.show()

			# evaluate model on the hold out dataset
			print("Start evaluating model...")
			yhat = best_model.predict(X_test)

			# evaluate the model
			acc = accuracy_score(y_test, yhat)
			acc_balanced = balanced_accuracy_score(y_test, yhat)
			print('Balanced Accuracy: %.3f' % acc_balanced)
			f1 = f1_score(y_true=y_test, y_pred=yhat,  average="weighted")
			precision = precision_score(y_true=y_test, y_pred=yhat, average="weighted")
			recall = recall_score(y_true=y_test, y_pred=yhat,  average="weighted")
			#TODO document this: why I am using balanced accuracy (https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)
			# and weighted f1, precision and recall (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html=

			#Visualize Confusion Matrix
			# create 10x5 figure
			fig, ax = plt.subplots(figsize=(10, 5))
			mat = confusion_matrix(y_test, yhat)
			sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
			# change xticks and yticks to string labels
			plt.title('Confusion Matrix where Proband ' + str(i) + " was test-data" )
			plt.xlabel('true label')
			plt.ylabel('predicted label')
			plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_feature_timeperiod-" + str(timeperiod) + "_timeperiod_around_event-" + str(label_segment) + "_test_proband-" + str(i) + "_ConfusionMatrix.png")
			#plt.show()

			#apply binomial test
			#TODO implement correctly for our multiclass problem
			print("Start binomial test...")
			pvalue_binom = binom_test(x=mat[0][0]+mat[1][1], n=len(y_test), p=0.5, alternative='greater')
			print("P-value binomial test: ", pvalue_binom)
			print("Binomial test done.")

			#feature importance: compute SHAP values
			print("Start computing SHAP values...")
			explainer = shap.Explainer(best_model)
			shap_values = explainer.shap_values(X_test)
			#plt.figure(figsize=(10, 5))
			fig, ax = plt.subplots(figsize=(10, 5))
			plt.title("Feature Importance for iteration with proband " +str(i) + " as test set")
			# plot summary_plot with tight layout
			shap.summary_plot(shap_values[1], X_test_df, plot_type="bar", show=False, plot_size=(20, 10) )
			#fig = plt.gcf()
			fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name +"/sensors_included-" + sensors_included + "_feature_timeperiod-" + str(timeperiod) + "_timeperiod_around_event-" + str(label_segment) + "_test_proband-" + str(i) + "_SHAPFeatureImportance.png")
			#plt.show()

			# store statistical test results (p-value permutation test, accuracy of that permutation iteration, pvalue binomial test) in list
			print("Start storing statistical test results...")
			permutation_pvalue.append(pvalue_model)
			permutation_modelaccuracy.append(score_model)
			pvalues_binomial.append(pvalue_binom)

			# store the resulted metrics
			test_proband.append(i)
			outer_results_acc.append(acc)
			outer_results_acc_balanced.append(acc_balanced)
			outer_results_f1.append(f1)
			outer_results_precision.append(precision)
			outer_results_recall.append(recall)
			best_parameters.append(str(result.best_params_))

			# store the y_test and yhat for the final accuracy computation
			test_proband_array = np.concatenate((test_proband_array, np.array((i,)*len(y_test))))
			y_test_array = np.concatenate((y_test_array, y_test))
			y_pred_array = np.concatenate((y_pred_array, yhat))

			# report progress
			t1_inner = time.time()
			#print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
			#print("Permutation-Test p-value was " + str(pvalue_model) + " and Binomial Test p-values was " + str(pvalue_binom))
			#print("The proband taken as test-data for this iteration was " + str(i))
			#print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

		# Save the resulting metrics:
		results = pd.DataFrame()
		results["Test-Proband"] = test_proband
		results["Accuracy"] = outer_results_acc
		results["Balanced Accuracy"] = outer_results_acc_balanced
		results["Accuracy by PermutationTest"] = permutation_modelaccuracy
		results["F1"] = outer_results_f1
		results["Precision"] = outer_results_precision
		results["Recall"] = outer_results_recall
		results["P-Value Permutation Test"] = permutation_pvalue
		results["P-Value Binomial Test"] = pvalues_binomial
		# add best parameters
		results["Best Parameters"] = best_parameters
		# add label column name as column
		results["Label Column Name"] = label_column_name
		# add seconds around event as column
		results["Seconds around Event"] = label_segment
		# add timeperiod of features as column
		results["Feature Segment Length"] = timeperiod
		results.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name + "/sensors_included-" + sensors_included + "_feature_timeperiod-" + str(timeperiod) + "_timeperiod_around_event-" + str(label_segment) +  "_results.csv")

		# save the y_test and yhat for the final accuracy computation
		df_labels_results = pd.DataFrame({"y_test": y_test_array, "y_pred": y_pred_array, "test_proband": test_proband_array})
		df_labels_results.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name + "/sensors_included-" + sensors_included + "_feature_timeperiod-" + str(timeperiod) + "_timeperiod_around_event-" + str(label_segment) +  "_results_labelsRealAndPredicted.csv")

		#TODO: document this!
		# compute the final metrics for this LOSOCV: here the cumulated y_test and yhat are used in order to account
		# for the fact that some test-participants have more data than others AND that some participants more label-classes
		# were present then for other participants
		timeperiod_list.append(timeperiod)
		label_list.append(label_column_name)
		acc_final_list.append(accuracy_score(y_test_array, y_pred_array))
		bal_acc_final_list.append(balanced_accuracy_score(y_test_array, y_pred_array))
		f1_final_list.append(f1_score(y_test_array, y_pred_array, average = "weighted"))
		precision_final_list.append(precision_score(y_test_array, y_pred_array, average = "weighted"))
		recall_final_list.append(recall_score(y_test_array, y_pred_array, average = "weighted"))
		#TODO include here also a binomial test for the final accuracy
		#TODO include here also a confusion matrix
		#TODO think how to apply permutation test here

		# summarize the estimated performance of the model
		print("The results for timeperiod " + str(timeperiod) + " and label " + str(label_column_name) + " are:")
		print('Mean Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
		print('Mean F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
		print('Mean Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
		print('Mean Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
		print("Mean p-value of Permutation Test: %.3f (%.3f)" % (mean(permutation_pvalue), std(permutation_pvalue)))
		print("Mean of p-value of Binomial Test: %.3f (%.3f)" % (mean(pvalues_binomial), std(pvalues_binomial)))

		print("Computing for timeperiod " + str(timeperiod) + " and label " + label_column_name + "has been minutes: " + str((time.time() - t0_label)/60))

	# add together the results of all activities
	timeperiod_list_outside.append(timeperiod_list)
	label_list_outside.append(label_list)
	acc_final_list_outside.append(acc_final_list)
	bal_acc_final_list_outside.append(bal_acc_final_list)
	f1_final_list_outside.append(f1_final_list)
	precision_final_list_outside.append(precision_final_list)
	recall_final_list_outside.append(recall_final_list)

	#save these intermediate results to a csv
	df_results_intermediate = pd.DataFrame()
	df_results_intermediate["Timeperiod"] = timeperiod_list
	df_results_intermediate["Label"] = label_list
	df_results_intermediate["Accuracy"] = acc_final_list
	df_results_intermediate["Balanced Accuracy"] = bal_acc_final_list
	df_results_intermediate["F1"] = f1_final_list
	df_results_intermediate["Precision"] = precision_final_list
	df_results_intermediate["Recall"] = recall_final_list
	df_results_intermediate.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/sensors_included-" + sensors_included + "_feature_timeperiod-" + str(timeperiod) + "_timeperiod_around_event-" + str(label_segment) +  "_results_for_this_label_segment.csv")

#join the combined results of all activities in a dataframe
df_results_final = pd.DataFrame()
df_results_final["Feature Segment Length (in s)"] = timeperiod_list_outside
df_results_final["Activity"] = label_list_outside
df_results_final["Accuracy"] = acc_final_list_outside
df_results_final["Balanced Accuracy"] = bal_acc_final_list_outside
df_results_final["F1"] = f1_final_list_outside
df_results_final["Precision"] = precision_final_list_outside
df_results_final["Recall"] = recall_final_list_outside
df_results_final.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/sensors_included-" + sensors_included + "_timeperiod_around_event-" + str(label_segment) +  "_results_for_all_activities_and_feature_segments.csv")
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
	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="label", task=tfdf.keras.Task.CLASSIFICATION)
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, label="label", task=tfdf.keras.Task.CLASSIFICATION)
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
	tuner.search(train_ds, epochs=1, validation_data=test_ds)

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

