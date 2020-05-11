import json
import os
import pickle

import BaselineWanderRemoval
import numpy as np
import pandas as pd

import config

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


INSPECT = False
WINDOW_LEN = 220
SAVE_TO_FILE = False


def extract_name (text, start_marker, end_marker):
	start = text.index(start_marker) + len(start_marker)
	end = text.index(end_marker, start)

	return text[start:end]


def load_json (filename):
	with open(filename) as f:
		data = json.load(f)
		f.close()

		return data


def split_ecg_lead_by_window (ecg, window=200):
	tmp = [x for x in np.array_split(np.array(ecg), int(len(ecg) / window)) if x.size > 0]
	res = np.zeros((len(tmp), window, 1))

	try:
		for ix in range(len(tmp)):
			if len(tmp[ix]) // window:
				tmp[ix] = tmp[ix][:window]

				BaselineWanderRemoval.fix_baseline_wander(tmp[ix], sr=127)

			res[ix] = np.array(tmp[ix]).reshape((window, 1))
	except ValueError as e:
		print(e)

	return res


def split_ecg_label_by_window (ecg, window=200):
	tmp = [x for x in np.array_split(np.array(ecg), int(len(ecg) / window)) if x.size > 0]
	res = np.zeros((len(tmp), window,))

	for ix in range(len(tmp)):
		if len(tmp[ix]) // window:
			tmp[ix] = tmp[ix][:window]
		res[ix] = np.array(tmp[ix]).reshape((window,))

	return res


def inspect_db (df):
	patients = set()
	leads = set()
	ecgs = []
	fss = set()
	max_ecg_raw_value = -np.inf
	min_ecg_raw_value = np.inf
	total_length = 0
	for index, row in df.iterrows():
		json_filename = os.path.join(config.ECG_DATA_DIR, row['name'])
		json_data = load_json(json_filename)

		ecg = json_data['data'][row['filename']]['ecg'][0]
		fs = json_data['data'][row['filename']]['fs']

		if max(ecg) > max_ecg_raw_value:
			max_ecg_raw_value = max(ecg)
		if min(ecg) < min_ecg_raw_value:
			min_ecg_raw_value = min(ecg)

		total_length += len(ecg)
		patients.add(row['Patient'])
		leads.add(row['Lead'])
		ecgs.append(len(ecg))
		fss.add(fs)

	print('Number of ECG leads: ', len(df.values.tolist()))
	print('Number of distinct patients: ', len(patients))
	print('Distinct sampling rates: ', fss)
	print('Number of lead types: ', len(leads))
	print('Max size of ECG: ', max(ecgs))
	print('Min size of ECG: ', min(ecgs))
	print('Mean size of ECG: ', np.mean(ecgs))
	print('Distinct sizes of ECG: ', set(ecgs))
	print('Max raw value of ECG: ', max_ecg_raw_value)
	print('Min raw value of ECG: ', min_ecg_raw_value)
	print('Total length in database: ', total_length)


def preprocess (df, window_len):
	x_data = np.empty((1, window_len, 1))
	y_data = np.zeros((1, window_len,))

	for index, row in df.iterrows():
		filename = os.path.join(config.ECG_DATA_DIR, row['name'])
		ecg_csv_name = row['filename']

		ecg = load_json(filename)

		ecg_lead = ecg['data'][ecg_csv_name]['ecg'][0]
		ecg_labels = ecg['data'][ecg_csv_name]['label'][0]

		assert len(ecg_lead) == len(ecg_labels)

		ecg_lead_splitted = split_ecg_lead_by_window(ecg_lead, window=window_len)
		ecg_label_splitted = split_ecg_label_by_window(ecg_labels, window=window_len)

		x_data = np.vstack((x_data, ecg_lead_splitted))
		y_data = np.vstack((y_data, ecg_label_splitted))

	return x_data, y_data


def preprocess_baseline_wander_removal (df, window_len):
	x_data = np.empty((1, window_len, 2))
	y_data = np.zeros((1, window_len,))

	for index, row in df.iterrows():
		filename = os.path.join(config.ECG_DATA_DIR, row['name'])
		ecg_csv_name = row['filename']

		ecg = load_json(filename)

		ecg_lead = ecg['data'][ecg_csv_name]['ecg'][0]
		ecg_labels = ecg['data'][ecg_csv_name]['label'][0]
		fs = ecg['data'][ecg_csv_name]['fs']

		assert len(ecg_lead) == len(ecg_labels)

		ecg_lead_splitted = split_ecg_lead_by_window(ecg_lead, window=window_len)
		try:
			shape = ecg_lead_splitted.shape[0]*ecg_lead_splitted.shape[1]
			ecg_lead_splitted_filtered = np.array(
					BaselineWanderRemoval.fix_baseline_wander(ecg_lead_splitted.reshape((shape,)), sr=fs)
			).reshape((ecg_lead_splitted.shape[0], 220, 1))

			a = ecg_lead_splitted.reshape((shape,))
			b = ecg_lead_splitted_filtered.reshape((shape,))

			if ecg_lead_splitted.shape[0] > 1:
				ddde = 5
			c = np.dstack((a, b)).reshape((ecg_lead_splitted.shape[0], 220, 2))
		except ValueError:
			saa = 5

		ecg_label_splitted = split_ecg_label_by_window(ecg_labels, window=window_len)

		x_data = np.vstack((x_data, c))
		y_data = np.vstack((y_data, ecg_label_splitted))

	return x_data, y_data


def fun (df, mask_lambda, window_len=200, bwr=False):
	masked_df = df[mask_lambda]

	if bwr:
		return preprocess_baseline_wander_removal(masked_df, window_len=window_len)

	return preprocess(masked_df, window_len=window_len)


def save_as_pkl (filename, data):
	with open(config.RESOURCES_DIR + filename, 'wb') as f:
		pickle.dump(data, f)
		f.close()


def prepare_split_df (split_scv_path):
	split_df = pd.read_csv(split_scv_path)
	split_df[['Database', 'filename']] = split_df.name.str.split('/', expand=True)
	split_df['filename'] = split_df.apply(lambda x: extract_name(x['filename'], '', '.json'), axis=1)

	return split_df


if __name__ == "__main__":

	###############################################################################
	# temporary
	# with open(config.RESOURCES_DIR + '/training_log.txt', 'r') as f:
	# 	train_acc = []
	# 	val_acc = []
	# 	while f.readline():
	# 		line = f.readline().split()
	# 		if len(line) == 3:
	# 			if line[0].split(',')[1] == 'train':
	# 				train_acc.append(float(line[2]))
	# 			if line[0].split(',')[1] == 'val':
	# 				val_acc.append(float(line[2]))
	#
	# 	mean_train_acc = np.mean(np.array(train_acc).reshape((len(train_acc), 1)))
	# 	mean_val_acc = np.mean(np.array(val_acc).reshape((len(val_acc), 1)))
	# 	print('mean_train_acc', mean_train_acc)
	# 	print('mean_val_acc', mean_val_acc)
	###############################################################################

	split_df = prepare_split_df(config.ECG_DATA_DIR + '/split.csv')

	print(split_df.head(5))

	if INSPECT:
		distinct_databases = set(split_df['Database'].values.tolist())
		for distinct_database in distinct_databases:
			print(str(distinct_database).upper())

			current_db_df = split_df[split_df['Database'] == distinct_database]
			inspect_db(current_db_df)
			print()
			print()

	x_train, y_train = fun(
			split_df,
			split_df.apply(
					lambda x: True, axis=1
			),
			window_len=WINDOW_LEN,
			bwr=False

	)

	from util import train_val_test_data_split

	train_set, val_set, test_set = train_val_test_data_split(x_train, y_train)

	print('train set: ', train_set[0].shape, train_set[1].shape)
	print('val set: ', val_set[0].shape, val_set[1].shape)
	print('test set: ', test_set[0].shape, test_set[1].shape)

	if SAVE_TO_FILE:
		save_as_pkl('/train_set_bwr.pkl', train_set)
		save_as_pkl('/val_set_bwr.pkl', val_set)
		save_as_pkl('/test_set_bwr.pkl', test_set)
