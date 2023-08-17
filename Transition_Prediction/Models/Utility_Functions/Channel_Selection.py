import numpy as np

## select specific channels for model training and testing
def select1dFeatureChannels(group_value, channel_to_calculate):
    train_emg1_x = group_value['train_feature_x'][:, :, 0: 65]
    test_emg1_x = group_value['test_feature_x'][:, :, 0: 65]
    train_emg2_x = group_value['train_feature_x'][:, :, 65: 130]
    test_emg2_x = group_value['test_feature_x'][:, :, 65: 130]
    num_features = int(group_value['train_feature_x'].shape[-1] / 130)  # which features to compute

    for i in range(1, num_features):  # extract emg1 data and emg2 data
        train_emg1_x = np.concatenate((train_emg1_x, group_value['train_feature_x'][:, :, 0 + 130 * i: 65 + 130 * i]), axis=-1)
        test_emg1_x = np.concatenate((test_emg1_x, group_value['test_feature_x'][:, :, 0 + 130 * i: 65 + 130 * i]), axis=-1)
        train_emg2_x = np.concatenate((train_emg2_x, group_value['train_feature_x'][:, :, 65 + 130 * i: 130 + 130 * i]), axis=-1)
        test_emg2_x = np.concatenate((test_emg2_x, group_value['test_feature_x'][:, :, 65 + 130 * i: 130 + 130 * i]), axis=-1)
    train_set_y = group_value['train_onehot_y'][:, 0, :]
    test_set_y = group_value['test_onehot_y'][:, 0, :]

    if channel_to_calculate == 'emg_1':
        train_set_x = train_emg1_x[:, :, 0: 65 * num_features]
        test_set_x = test_emg1_x[:, :, 0: 65 * num_features]
    elif channel_to_calculate == 'emg_2':
        train_set_x = train_emg2_x[:, :, 0: 65 * num_features]
        test_set_x = test_emg2_x[:, :, 0: 65 * num_features]
    elif channel_to_calculate == 'emg_all':
        train_set_x = group_value['train_feature_x'][:, :, 0: 130 * num_features]
        test_set_x = group_value['test_feature_x'][:, :, 0: 130 * num_features]
    elif channel_to_calculate == 'emg_bipolar':
        pass
        # bipolar input data
        # train_bipolar_x = group_value['train_feature_x'][:, 33].reshape(len(group_value['train_int_y']), 1)
        # for i in range(1, 16):
        #     emg_feature_bipolar_x = np.concatenate((train_bipolar_x, group_value[:, 33+65*i].reshape(len(group_value['train_int_y']), 1)), axis=1)
    else:
        raise Exception("No Such Channels")

    return train_set_x, train_set_y, test_set_x, test_set_y