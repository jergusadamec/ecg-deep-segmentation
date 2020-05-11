import pickle

import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data.dataset import Dataset


class ECGDataset (Dataset):

	def __init__(self, data_path, transform=None):
		self.transform = transform
		with open(data_path, 'rb') as f:
			self.x, self.y = pickle.load(f)

	def __len__ (self):
		return len(self.x)

	def __getitem__ (self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		ecg = torch.from_numpy(self.x[idx]).float()
		label = torch.from_numpy(self.y[idx]).float()

		sample = {
			'ecg': ecg,
			'label': label
		}

		if self.transform:
			sample['ecg'] = self.transform(sample['ecg'])

		return sample


class PyTorchMinMaxScalerVectorized(object):

	def __init__(self, fitted_min_max_scaler: MinMaxScaler):
		self.fitted_min_max_scaler: MinMaxScaler = fitted_min_max_scaler

	def __call__(self, tensor):
		return torch.tensor(self.fitted_min_max_scaler.transform(tensor.numpy()))


def fit_min_max_scaler (path, type='stad'):
	with open(path, 'rb') as f:
		x, y = pickle.load(f)

		reshaped_x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))

		if type == 'stad':
			scaler = StandardScaler()
			scaler.fit(reshaped_x)
		elif type == 'norm':
			scaler = MinMaxScaler(feature_range=(0, 1))
			scaler.fit(reshaped_x)

		f.close()

		return scaler
