import torch
import torch.nn as nn

import config
from util import load_json


class SegModel(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, out_size):
		super().__init__()
		self.features = torch.nn.Sequential(
			torch.nn.LSTM(
					input_size=input_size,
					hidden_size=hidden_size,
					num_layers=num_layers,
					batch_first=True,
					bidirectional=True
			),
		)
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(2*hidden_size, 2*hidden_size),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(),

			torch.nn.Linear(2 * hidden_size, 2 * hidden_size),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(),
		)
		self.output = torch.nn.Linear(2*hidden_size, out_size)

	def forward(self, x):
		batch, seq_len, nums_fea = x.size()
		features, _ = self.features(x)
		output = self.classifier(features)
		output = self.output(output.view(batch * seq_len, -1))

		return output


class CnnSegModel(nn.Module):
	"""
		attr 'input_size': number of channels in input signal;
			For example for image recognizition there are 3 channels for R,G and B value;
		attr 'kernel_size': size of the convolving kernel;

		Conv1D excepts the input to be of the shape - [batch_size, input_channels, signal_length]
	"""

	def __init__(self, input_size=1, hidden_size=32, kernel_size=3, out_size=5, batch_size=32):
		super().__init__()
		self.batch_size = batch_size
		self.features = nn.Sequential(
			nn.Conv1d(in_channels=input_size, out_channels=1, kernel_size=kernel_size, padding=0),
			nn.ReLU(),
			# nn.MaxPool1d(3, stride=1),
			nn.Dropout()
		)
		# Classify output, fully connected layers
		self.memory = nn.Sequential(
			torch.nn.LSTM(
					input_size=input_size,
					hidden_size=hidden_size,
					num_layers=1,
					batch_first=True,
					bidirectional=True
			)
		)
		self.deconv = nn.ConvTranspose1d(in_channels=2*hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=0)
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(hidden_size, 2 * hidden_size),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(),
		)
		self.output = torch.nn.Linear(2 * hidden_size, out_size)

	def forward(self, x):
		"""
		:param x: shape(batch, seq_len, input_size) -->  (batch_size, features/input_channels, timesteps/signal_length)
		:return:
		"""
		batch, seq_len, nums_fea = x.size()
		x = x.reshape((batch, 1, seq_len))

		batch, nums_fea, seq_len = x.size()

		features = self.features(x)
		features_res = features.reshape((batch, 398, 1))

		memory, _ = self.memory(features_res)
		deconv = self.deconv(memory.reshape((batch, 64, 398)))
		deconv = deconv.reshape((batch, seq_len, 32))
		output = self.classifier(deconv)
		out = self.output(output.view(batch * seq_len, -1))

		return out


def model_factory (model_name):
	model_params = load_json(config.ROOT_DIR + '/model_params.json')[model_name]

	if model_name == 'seg-net':
		return SegModel(
				input_size=model_params['input_size'],
				hidden_size=model_params['hidden_size'],
				num_layers=model_params['num_layers'],
				out_size=model_params['out_size']
		)
	elif model_name == 'cnn-seg-net':
		return CnnSegModel(
				input_size=model_params['input_size'],
				hidden_size=model_params['hidden_size'],
				out_size=model_params['out_size'],
				kernel_size=model_params['kernel_size']
		)