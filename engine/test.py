import pickle

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics._plot import confusion_matrix

import config
from util.utils import plot_learning_curve
from util.utils import plot_confusion_matrix
from visualization import visualise_ecg


# def eval(val_loader, model, criterion, params, epoch, logger):
# 	"""Main method to evaluate model."""
# 	model.eval()
# 	print("================================")
# 	print("Evaluating...")
#
# 	loss = 0
# 	num_classes = params["num_classes"]
#
# 	confusion = np.zeros((num_classes, num_classes))
# 	for iteration, (steps, targets, _) in enumerate(tqdm(val_loader)):
# 		if params["use_cuda"]:
# 			steps = steps.cuda()
# 			targets = targets.cuda()
#
# 		output = model(steps)
#
# 		rows = targets.cpu().numpy()
# 		cols = output.max(1)[1].cpu().numpy()
#
# 		confusion[rows, cols] += 1
#
# 		loss += criterion(output, targets)
#
# 	loss = loss / iteration
# 	acc = np.trace(confusion) / np.sum(confusion)
#
# 	# Plot confusion matrix in visdom
# 	logger.heatmap(confusion, win='4', opts=dict(
# 		title="Confusion_Matrix_epoch_{}".format(epoch),
# 		columnnames=["A","B","C","D","E"],
# 		rownames=["A","B","C","D","E"])
# 	)
#
# 	return loss, acc


def test(net, test_loader, device, batch_size, plot_ecg=False, plot_ecg_windows_size=None):
	with torch.no_grad():
		ecgs_list = []
		labels_list = []
		predicted_list = []
		right = 0.0
		total = 0.0
		net.eval()
		for ix, sample in enumerate(test_loader):
			ecgs = sample['ecg']
			labels = sample['label']

			ecgs = ecgs.to(device)
			labels = labels.to(device)

			if ecgs.shape[0] < batch_size:
				batch_size = ecgs.shape[0]

			output = net(ecgs)
			output = output.to(device)

			_, predicted = torch.max(output.data, 1)
			label_true = labels.contiguous().view(-1).long()

			total += label_true.size(0)
			right += (predicted == label_true).sum().item()

			if ecgs.shape[0] == 32:
				ecgs_list.extend(ecgs.numpy())
				labels_list.extend(label_true.numpy())
				predicted_list.extend(predicted.numpy())

			if plot_ecg:
				visualise_ecg(
					ecg=ecgs.numpy(),
					labels=labels.numpy(),
					pred_vec=predicted.numpy(),
					plot_window=plot_ecg_windows_size,
					max_plots=3
				)

		print("{} ACC: {:.4f}".format('testing', right / total))

		return ecgs, labels_list, predicted_list


def eval (ecgs, y_true, y_pred, labels, target_names, plot_acc=False, plot_loss=False, plot_conf_matrix=False, plot_ecg=False, plot_ecg_windows_size=None):

	if plot_acc:
		with open(config.RESOURCES_DIR + '/loss.pkl', 'rb') as f:
			loss = pickle.load(f)
			plot_learning_curve(loss, xlabel='Episode', ylabel='Loss')

	if plot_loss:
		with open(config.RESOURCES_DIR + '/accuracy.pkl', 'rb') as f:
			acc = pickle.load(f)
			plot_learning_curve(acc, xlabel='Episode', ylabel='Accuracy')

	# if plot_ecg:
	# 	visualise_ecg(
	# 		ecg=ecgs.numpy(),
	# 		labels=y_true.numpy(),
	# 		pred_vec=y_pred.numpy(),
	# 		plot_window=plot_ecg_windows_size,
	# 		max_plots=3
	# 	)

	if plot_conf_matrix:
		conf_mat = confusion_matrix.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

		plot_confusion_matrix(
			confusion_matrix=conf_mat,
			target_names=target_names,
			title='Confusion matrix',
			normalize=True
		)

	report = classification_report(y_true=y_true, y_pred=y_pred, labels=labels, target_names=target_names)

	print('Report: ' + str(report))