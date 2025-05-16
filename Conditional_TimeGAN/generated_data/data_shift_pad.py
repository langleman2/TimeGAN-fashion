
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




#cat = 'BL'



category = ['BL', 'CT', 'JK', 'JP', 'KC', 'TS', 'KT', 'OP', 'PD', 'SK', 'SL', 'TC', 'VT']
#category = ['SK', 'SL', 'TC', 'VT']

for cat in category:

    ori_data = pd.read_csv(f'/home/langleman/Conditional_TimeGAN/data/{cat}_weekly_new_train.csv')
    ori_data.plot(figsize= (16,9))

    # %%
    #df = pd.read_csv(f'/home/langleman/Conditional_TimeGAN/generated_data/data/generated_data_new_{cat}_fm100_iter6000_hid24_numlayer9_batch102.csv')
    df = pd.read_csv(f'/home/langleman/Conditional_TimeGAN/generated_data/data/generated_data_new_{cat}_1121_all.csv')
    #print(df.shape)

    min = ori_data.min()
    max = ori_data.max()

    df_renorm = df * max[0] + min[0]
    df_renorm




    ori = ori_data.판매수량.to_list()

    pad_list = []
    for i in range(int(df_renorm.shape[0])):

        df = df_renorm.loc[i].to_list()

    # 최적의 shifting 계산
        result = find_best_shift_and_replace(ori, df)
        pad_list.append(result)


    df = pd.DataFrame(pad_list)

    #print(df)

    folder_path = '/home/langleman/Conditional_TimeGAN/generated_data/data4inference'

    df.to_csv(os.path.join(folder_path, f'padded_weekly_{cat}_1121_all.csv'), index=False)


    vars()[f'pad_{cat}'] = pd.read_csv(os.path.join(folder_path, f'padded_weekly_{cat}_1121_all.csv'))

    fig,ax = plt.subplots(1)
    
    # for i in range(12):
    #     vars()[f'pad_{cat}'].loc[i].plot(ax=ax,figsize = (16,9), color = 'b', alpha = 0.2)
    ori_data.plot(ax=ax, color = 'g', label = '원본 데이터')

    # 저장할 파일 경로 및 파일 이름 설정
    save_path = f'/home/langleman/Conditional_TimeGAN/generated_data/only_original/{cat}_original.png'



    # 그림 저장
    plt.savefig(save_path)



