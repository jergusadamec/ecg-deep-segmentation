import random
from torch.utils import data
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
import numpy as np
import pandas as pd


def train_val_test_data_split (
		X: np.ndarray,
		y: np.ndarray,
		val=.05,
		test=.2
):
	assert len(X) == len(y), 'Length of X and y must be the same'
	n = len(X)
	n_test = int(n * test)
	n_val = int(n * val)
	n_train = n - (n_test + n_val)

	idx = list(range(n))
	random.shuffle(idx)

	train_idx = idx[:n_train]
	val_idx = idx[n_train:(n_train + n_val)]
	test_idx = idx[(n_train + n_val):]

	train_set = np.array([X[ix] for ix in train_idx]), np.array([y[ix] for ix in train_idx])
	val_set = np.array([X[ix] for ix in val_idx]), np.array([y[ix] for ix in val_idx])
	test_set = np.array([X[ix] for ix in test_idx]), np.array([y[ix] for ix in test_idx])

	return train_set, val_set, test_set


def restore_net (ckpt):
	with open(ckpt, 'rb') as f:
		net = torch.load(f, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	return net


def load_json (filename):
	with open(filename) as f:
		data = json.load(f)
		f.close()

		return data


def save_as_pkl (filename_path, data):
	with open(filename_path, 'wb') as f:
		pickle.dump(data, f)
		f.close()


def plotecg (x, y, start, end):
	x = x[start:end, 0]
	y = y[start:end]
	cmap = ['k', 'r', 'g', 'b']
	start = end = 0
	for i in range(len(y) - 1):
		if y[i] != y[i + 1]:
			end = i
			plt.plot(np.arange(start, end + 1), x[start:end + 1], cmap[int(y[i])])
			start = i + 1
	plt.show()


def plot_rewards_with_std (reward, std_reward_plus, std_reward_minus, xlabel, ylabel):
	x = [i for i in range(len(reward))]
	y = reward

	avg_color = 'black'
	std_color = '#DDDDDD'

	plt.plot(x, y, color=avg_color, linewidth=1.5, label='mean')
	plt.fill_between(x, std_reward_plus, std_reward_minus, color=std_color, label='std')

	plt.tick_params('y', labelsize=20)
	plt.tick_params('x', labelsize=20)

	plt.xlabel(xlabel, size=20)
	plt.ylabel(ylabel, size=20)

	plt.legend(loc='best', prop={'size': 15})
	plt.show()


def plot_learning_curve (data, xlabel, ylabel):
	means = list(map(lambda x: np.mean(x, axis=0), data))
	std = list(map(lambda x: np.std(x, axis=0), data))

	std_plus = [
		means[i] + std[i]
		for i in range(len(means))
	]

	std_minus = [
		means[i] - std[i]
		for i in range(len(means))
	]

	plot_rewards_with_std(
			reward=means,
			std_reward_plus=std_plus,
			std_reward_minus=std_minus,
			xlabel=xlabel,
			ylabel=ylabel
	)


def plot_dist_with_stats (
		data,
		labels=None,
		title='Distribution of ECG Signal',
		ax=None,
		stats=True
):
	mean = data.mean(skipna=True)
	std = data.std(skipna=True)

	if ax is None:
		fig, ax = plt.subplots()

	sns.distplot(data, bins=200, fit=norm, kde=True, ax=ax, norm_hist=True, hist=True)

	if stats:
		ax.axvline(mean.item(), color='w', linestyle='dashed', linewidth=2)
		ax.axvline(std.item(), color='r', linestyle='dashed', linewidth=2)
		ax.axvline(-std.item(), color='r', linestyle='dashed', linewidth=2)

	ax.set_xlabel("Samples")
	ax.set_ylabel("Probability density")
	ax.set_title(title)
	ax.text(-7, 0.1, "Extreme negatives")
	ax.text(7, 0.1, "Extreme positives")
	if labels is not None:
		plt.legend(labels=labels)
	plt.show()

	return ax


def print_confusion_matrix (confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
	# plt.imshow(confusion_matrix, cmap='binary', interpolation='None')
	# plt.show()


	"""Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

	borrowed from: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

	Arguments
	---------
	confusion_matrix: numpy.ndarray
		The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
		Similarly constructed ndarrays can also be used.
	class_names: list
		An ordered list of class names, in the order they index the given confusion matrix.
	figsize: tuple
		A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
		the second determining the vertical size. Defaults to (10,7).
	fontsize: int
		Font size for axes labels. Defaults to 14.

	Returns
	-------
	matplotlib.figure.Figure
		The resulting confusion matrix figure
	"""
	df_cm = pd.DataFrame(
			confusion_matrix, index=class_names, columns=class_names,
	)
	fig = plt.figure(figsize=figsize)
	try:
		heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
	except ValueError:
		raise ValueError("Confusion matrix values must be integers.")
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()
	return fig
