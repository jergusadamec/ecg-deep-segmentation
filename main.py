import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from engine import test
from engine import train
from engine import eval
from modeling import ECGDataset, PyTorchMinMaxScalerVectorized, fit_min_max_scaler
from modeling import model_factory
from util import restore_net

TRAIN = False
CONTINUE_TRAIN = False
TEST = True
EPOCHS = 100
BATCH_SIZE = 32
NUM_SEGS_CLASS = 5
WINDOWS_SIZE = 220
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

qtdb_pkl = 'resources/qtdb_pkl/'
save_path = 'resources/ckpt'

PATH_TO_TEST_MODEL = 'resources/z-score-data/ckpt' + '/epoch_99.ckpt'

if __name__ == '__main__':
    print(DEVICE)
    if DEVICE == 'cuda':
        print(torch.cuda.get_device_name(0))

    print('TEST: ' + str(TEST))
    print('PATH_TO_TEST_MODEL: '+ str(PATH_TO_TEST_MODEL))

    train_data_path = '/train_set.pkl'
    val_data_path = '/val_set.pkl'
    test_data_path = '/test_set.pkl'

    fitted_min_max_scaler_train = fit_min_max_scaler(path=config.RESOURCES_DIR + train_data_path)
    fitted_min_max_scaler_val = fit_min_max_scaler(path=config.RESOURCES_DIR + val_data_path)
    fitted_min_max_scaler_test = fit_min_max_scaler(path=config.RESOURCES_DIR + test_data_path)

    normalizer_train = PyTorchMinMaxScalerVectorized(fitted_min_max_scaler_train)
    normalizer_val = PyTorchMinMaxScalerVectorized(fitted_min_max_scaler_val)
    normalizer_test = PyTorchMinMaxScalerVectorized(fitted_min_max_scaler_test)

    # loading data
    ecg_train_db = ECGDataset(data_path=config.RESOURCES_DIR + train_data_path, transform=normalizer_train)

    train_loader = DataLoader(
            ecg_train_db,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
    )

    ecg_val_db = ECGDataset(data_path=config.RESOURCES_DIR + val_data_path, transform=normalizer_val)
    val_loader = DataLoader(
            ecg_val_db,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
    )

    ecg_test_db = ECGDataset(data_path=config.RESOURCES_DIR + test_data_path, transform=normalizer_test)
    test_loader = DataLoader(
            ecg_test_db,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
    )

    if TRAIN:
        if CONTINUE_TRAIN:
            # continue training
            net = restore_net(save_path + '/epoch_35.ckpt')
            net.to(DEVICE)
        else:
            # model
            model_name = 'seg-net'
            net = model_factory(model_name)
            net.to(DEVICE)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()

        train(
                train_loader=train_loader,
                val_loader=val_loader,
                net=net,
                epochs=EPOCHS,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE,
                batch_size=BATCH_SIZE
        )

    if TEST:
        net = restore_net(PATH_TO_TEST_MODEL)
        net.eval()
        net.to(DEVICE)

        # after the training run function for train/val/test loader
        loader = test_loader

        ecgs, y_true, y_pred = test(
                net=net,
                test_loader=loader,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                plot_ecg=False,
                plot_ecg_windows_size=WINDOWS_SIZE
        )

        eval(
            ecgs=ecgs,
            y_true=y_true,
            y_pred=y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=['none', 'p_wave', 'qrs', 't_wave', 'extrasystole'],
            plot_acc=True,
            plot_loss=True,
            plot_conf_matrix=True,
            plot_ecg=True,
            plot_ecg_windows_size=WINDOWS_SIZE
        )









