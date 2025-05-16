import numpy as np
import os

def MinMaxScaler(data):
    """Min Max normalizer."""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data
import os

def real_data_loading(data_names, seq_len):
    """Load and preprocess real-world datasets."""
    data = []
    base_path = '/home/langleman/Conditional_TimeGAN/data/'  # 절대 경로로 수정

    for data_name in data_names:
        file_path = os.path.join(base_path, f'{data_name}.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        
        ori_data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        ori_data = MinMaxScaler(ori_data)
        
        temp_data = []
        for i in range(0, len(ori_data) - seq_len):
            _x = ori_data[i:i + seq_len]
            temp_data.append(_x)
        
        data.append(np.array(temp_data))
    
    # Concatenate all datasets
    concatenated_data = np.concatenate(data, axis=0)
    
    return concatenated_data



# data_names = [
#     'BL_weekly_new_train', 'CT_weekly_new_train', 'JK_weekly_new_train',
#     'JP_weekly_new_train', 'KC_weekly_new_train', 'TS_weekly_new_train',
#     'KT_weekly_new_train', 'OP_weekly_new_train', 'PD_weekly_new_train',
#     'SK_weekly_new_train', 'SL_weekly_new_train', 'TC_weekly_new_train',
#     'VT_weekly_new_train'
# ]
