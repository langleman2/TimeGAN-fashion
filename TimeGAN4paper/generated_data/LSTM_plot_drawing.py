import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from torch import cuda
from LSTM_2022_weekly_timegan import Encoder, category_data_multi3


device = 'cuda' if cuda.is_available() else 'cpu'



data_load_path = '/home/langleman/TimeGAN4paper/data/marimonte_weekly_new.csv'
category_list = ['JK', 'JP', 'BL','VT','SL','SK','OP','TC','CT','KT','KC','TS','PD']
category_dict = {'JK': 'JACKET', 'JP': 'JUMPER', 'BL': 'BLOUSE', 'VT': 'VEST', 'SL': 'SLACKS',
                  'SK': 'SKIRT', 'OP': 'ONE PIECE', 'TC': 'TRENCH COAT', 'CT': 'COAT', 'KT': 'KNIT', 'KC': 'CARDIGAN', 'TS': 'T-SHIRT', 'PD': 'PADDING'}
plot_save_path = '/home/langleman/TimeGAN4paper/result/1119_4paper/figures'
#csv_save_path = '/home/langleman/TimeGAN4paper/result/1119_4paper/LSTM_2022_weekly_1119_4paper.csv'

data = pd.read_csv(data_load_path)
epochs_num = 5000




def make_x_axis(train_y, test_y):
    train_x = []
    test_x = []

    year = 2015
    week = 1
    current_date = datetime(year, 1, 1)

    while len(train_x) < len(train_y):
        train_x.append(current_date)
        current_date += timedelta(weeks=1)
        if current_date.weekday() == 0:  # If it's Monday
            week += 1
        if week == 53:
            year += 1
            week = 1

    # Set the start date of test data to be the week after the end of train data
    current_date = train_x[-1] + timedelta(weeks=1)

    while len(test_x) < len(test_y):
        test_x.append(current_date)
        current_date += timedelta(weeks=1)
        if current_date.weekday() == 0:  # If it's Monday
            week += 1
        if week == 53:
            year += 1
            week = 1

    return train_x, test_x


def set_vline(ax): # 시즌별로 수직선 생성
    for i in ['2015','2016','2017','2018','2019','2020','2021', '2022']:
        '''
        ax.axvline(i + '-01-01',color='g')
        ax.axvline(i + '-04-01',color='r')
        ax.axvline(i + '-08-01',color='y')
        ax.axvline(i + '-10-01',color='r')
        '''

        
        '''
        # 시즌별로 색칠하기 
        ax.axvspan(i + '-01-01',i + '-04-01',alpha=0.3, facecolor='g')
        ax.axvspan(i + '-04-01',i + '-08-01',alpha=0.3, facecolor='r')
        ax.axvspan(i + '-08-01',i + '-10-01',alpha=0.3, facecolor='y')
        ax.axvspan(i + '-10-01',str(int(i)+1) + '-01-01',alpha=0.3, facecolor='b')
        '''
 

'''
def make_plot(train_y,test_y,pred_y, category, train_pred = [], save = False, save_path = plot_save_path): # real, predict값으로 그래프 생성
    train_x, test_x = make_x_axis(train_y, test_y)
    fig = plt.figure(figsize=[10,7])

    plt.plot(train_x,train_y,color='C0')
    #plt.plot([train_x[-1]]+ test_x,[train_y[-1]] +  test_y, color='black')
    
    if len(train_pred) == 0:
        plt.plot([train_x[-1]] + test_x, [train_y[-1]] + pred_y,'--',color='r')
    else :
        plt.plot(train_x[-len(train_pred):] + test_x, train_pred + pred_y,'--',color='r')     

    set_vline(plt)
    plt.legend(['train','test','pred'])
    #plt.axvline(date(2021,11,15), c = 'black')
    plt.title(f'category_dict[category]' + f'({category})')
    if save == True:
        plt.savefig(save_path +str(category)+'_(1STEP).png',bbox_inches='tight', facecolor='w')
    plt.show()

'''
def make_plot(train_y, test_y, pred_y, category, train_pred=[], save=False, save_path=plot_save_path):
    train_x, test_x = make_x_axis(train_y, test_y)
    fig = plt.figure(figsize=[10, 7])

    # 학습 데이터 플롯
    plt.plot(train_x, train_y, color='C0')

    # 예측값 플롯
    if len(train_pred) == 0:
        # 예측값의 길이와 x축 길이 맞추기
        plt.plot(test_x, pred_y, '--', color='r')
    else:
        train_pred_x = train_x[-len(train_pred):]  # train_pred에 맞는 x축 데이터 생성
        plt.plot(train_pred_x + test_x, train_pred + pred_y, '--', color='r')

    set_vline(plt)
    plt.legend(['train', 'test', 'pred'])
    plt.title(f'{category_dict[category]} ({category})')
    if save:
        plt.savefig(os.path.join(save_path, f'{category}_(1STEP).png'), bbox_inches='tight', facecolor='w')
    plt.show()





# 모델 로드 및 평가 후 플롯 생성
def load_and_plot(model_path, data, category, test_size=12, sequence_length=12, lag=1, input_size=1, hidden_size=64, num_layers=1):

    num_epochs = epochs_num
    learning_rate = 1e-3
    input_size = 1 # feature 갯수
    hidden_size = 64
    num_layers = 1
    num_classes = 1
    test_size = 52 # weekly
    threshold = 5000
    sequence_length = 52
    lag = 1
    
    # 1. 데이터 준비
    x_seq, y_seq = category_data_multi3(data, category, sequence_length, lag)
    x_test_seq = x_seq[-test_size:]
    y_test_seq = y_seq[-test_size:]

    test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=x_test_seq.shape[0], shuffle=False)

    # 2. 모델 초기화 및 가중치 로드
    model = Encoder(num_classes=1, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # 3. 예측값 계산
    test_pred = None
    with torch.no_grad():
        for x, y in test_loader:
            x_test = x.to(device)
            y_test = y.to(device)

            pred = model(x_test)
            test_pred = pred[0].cpu().numpy()  # 예측값 저장

    # 4. 결과 시각화
    y_test = y_test_seq.cpu().numpy()
    test_pred = test_pred.flatten()
    y_test = y_test.flatten()
    
    print(f"train_y length: {len(train_y)}, test_y length: {len(y_test)}, pred_y length: {len(test_pred)}")


    # Plot 그리기
    make_plot(
        train_y=np.zeros(len(x_seq) - test_size),  # 학습 데이터는 빈 리스트로 처리
        test_y=y_test,
        pred_y=test_pred,
        category=category,
        save=True,  # 저장 옵션
        save_path=plot_save_path
    )
    print(f"Plot saved for category {category}.")

# 모델 가중치 파일 경로
model_save_path = '/home/langleman/TimeGAN4paper/result/iter6000/model/JK_lag1_total1/torch_best_model.pt'

# 특정 카테고리로 플롯 생성 실행
load_and_plot(model_path=model_save_path, data=data, category='JK')