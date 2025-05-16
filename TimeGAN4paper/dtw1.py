
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "NanumGothic"





def find_best_shift_and_replace(ori_data, df_renorm):
    min_mse = float('inf')
    best_shifted_df = None

    for shift in range(len(ori_data) - len(df_renorm) + 1):
        shifted_df = np.pad(df_renorm, (shift, len(ori_data) - len(df_renorm) - shift), 'constant', constant_values=(0, 0))
        mse = mean_squared_error(ori_data, shifted_df[:len(ori_data)])
        
        if mse < min_mse:
            min_mse = mse
            best_shifted_df = shifted_df

    # Replace padded zeros with corresponding values from ori_data
    replaced_df = best_shifted_df.copy()
    for i in range(len(replaced_df)):
        if replaced_df[i] == 0:
            replaced_df[i] = ori_data[i]

    return replaced_df


import numpy as np
from scipy.spatial.distance import euclidean

def dtw_distance(series1, series2):
    """
    Calculate Dynamic Time Warping (DTW) distance between two time series.

    Args:
    - series1, series2: Input time series as 1-D numpy arrays.

    Returns:
    - DTW distance between series1 and series2.
    """
    # Calculate distance matrix
    n, m = len(series1), len(series2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf

    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(series1[i - 1], series2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Insertion
                                           dtw_matrix[i, j - 1],    # Deletion
                                           dtw_matrix[i - 1, j - 1])  # Match

    return dtw_matrix[n, m]

#category = ['BL']
category = ['BL', 'CT', 'JK', 'JP', 'KC', 'KT', 'OP', 'PD', 'SK', 'SL', 'TC', 'TS', 'VT']

for cat in category:
    

    ori_data = pd.read_csv(f'/home/langleman/TimeGAN4paper/data/{cat}_weekly_new_train.csv')
    array_ori_data = ori_data.values
    gen_data = pd.read_csv(f'/home/langleman/TimeGAN4paper/generated_data/data/generated_data_new_{cat}.csv')

    min_value = ori_data.min()  # 변수 이름을 min에서 min_value로 변경
    max_value = ori_data.max()  # 변수 이름을 max에서 max_value로 변경

    gen_data = gen_data * max_value[0] + min_value[0]

    cat_dict = dict()
    for idx, row in gen_data.iterrows():
        
        dtw_score = dtw_distance(array_ori_data, row.values)
        cat_dict[idx] = dtw_score
        # 값이 가장 작은 8개의 키를 추출
    sorted_keys = sorted(cat_dict, key=cat_dict.get)[:8]
        # sorted_keys에 해당하는 행만 남기기
    gen_data_filtered = gen_data.loc[sorted_keys]

    print(gen_data_filtered)

    
    ori = ori_data.values

    pad_list = []
    for i in range(int(gen_data_filtered.shape[0])):

        df = gen_data_filtered.iloc[i].to_list()

    # 최적의 shifting 계산
        result = find_best_shift_and_replace(ori, df)
        pad_list.append(result)


    df = pd.DataFrame(pad_list)

    print(df)

    # 새로운 CSV 파일로 저장
    df.to_csv(f'/home/langleman/TimeGAN4paper/generated_data/data4inference_dtw/padded_weekly_{cat}_dtw.csv', index=False)
    

