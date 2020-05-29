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


def train_val_test_data_split(
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


def restore_net(ckpt):
	with open(ckpt, 'rb') as f:
		net = torch.load(f, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	return net


def load_json(filename):
	with open(filename) as f:
		data = json.load(f)
		f.close()

		return data


def save_as_pkl(filename_path, data):
	with open(filename_path, 'wb') as f:
		pickle.dump(data, f)
		f.close()


def plotecg(x, y, start, end):
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


def plot_rewards_with_std(reward, std_reward_plus, std_reward_minus, xlabel, ylabel):
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


def plot_learning_curve(data, xlabel, ylabel):
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


def plot_dist_with_stats(
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


def plot_confusion_matrix(confusion_matrix, target_names, title='Confusion matrix', cmap=None, normalize=True):
	"""
	source: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

	given a sklearn confusion matrix (cm), make a nice plot

	Arguments
	---------
	confusion_matrix: confusion matrix from sklearn.metrics.confusion_matrix

	target_names: given classification classes such as [0, 1, 2]
				  the class names, for example: ['high', 'medium', 'low']

	title:        the text to display at the top of the matrix

	cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
				  see http://matplotlib.org/examples/color/colormaps_reference.html
				  plt.get_cmap('jet') or plt.cm.Blues

	normalize:    If False, plot the raw numbers
				  If True, plot the proportions

	Usage
	-----
	plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
															  # sklearn.metrics.confusion_matrix
						  normalize    = True,                # show proportions
						  target_names = y_labels_vals,       # list of names of the classes
						  title        = best_estimator_name) # title of graph

	Citiation
	---------
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

	"""
	import matplotlib.pyplot as plt
	import numpy as np
	import itertools

	accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
	misclass = 1 - accuracy

	if cmap is None:
		cmap = plt.get_cmap('Blues')

	plt.figure(figsize=(8, 6))
	plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45, size=19)
		plt.yticks(tick_marks, target_names, size=19)

	if normalize:
		cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		color = "white" if cm[i, j] > thresh else "black"
		if i == j:
			if i == 0:
				color = 'white'
			else:
				color = 'black'
		if normalize:
			plt.text(
				j, i, "{:0.4f}".format(cm[i, j]),
				horizontalalignment="center",
				color=color,
				size=23
			)
		else:
			plt.text(
				j, i, "{:,}".format(cm[i, j]),
				horizontalalignment="center",
				color=color,
				size=23
			)

	plt.tight_layout()
	plt.ylabel('True label', size=23)
	plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), size=23)
	plt.show()
