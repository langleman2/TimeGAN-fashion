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
device = 'cuda' if cuda.is_available() else 'cpu'

print(device)

data_load_path = '/home/langleman/TimeGAN4paper/data/marimonte_weekly_new.csv'
category_list = ['JK', 'JP', 'BL','VT','SL','SK','OP','TC','CT','KT','KC','TS','PD']
plot_save_path = '/home/langleman/TimeGAN4paper/result/original/figures/'
csv_save_path = '/home/langleman/TimeGAN4paper/result/original/LSTM_2022_weekly_iter6000.csv'

data = pd.read_csv(data_load_path)
epochs_num = 5000

os.makedirs(plot_save_path, exist_ok=True)


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
        ax.axvspan(i + '-01-01',i + '-04-01',alpha=0.3, facecolor='g')
        ax.axvspan(i + '-04-01',i + '-08-01',alpha=0.3, facecolor='r')
        ax.axvspan(i + '-08-01',i + '-10-01',alpha=0.3, facecolor='y')
        ax.axvspan(i + '-10-01',str(int(i)+1) + '-01-01',alpha=0.3, facecolor='b')
        
def make_plot(train_y,test_y,pred_y, category, train_pred = [], save = False, save_path = plot_save_path): # real, predict값으로 그래프 생성
    train_x, test_x = make_x_axis(train_y, test_y)
    fig = plt.figure(figsize=[10,7])

    plt.plot(train_x,train_y,color='C0')
    plt.plot([train_x[-1]]+ test_x,[train_y[-1]] +  test_y, color='black')
    
    if len(train_pred) == 0:
        plt.plot([train_x[-1]] + test_x, [train_y[-1]] + pred_y,'--',color='r')
    else :
        plt.plot(train_x[-len(train_pred):] + test_x, train_pred + pred_y,'--',color='r')
    set_vline(plt)
    plt.legend(['train','test','pred'])
    plt.axvline(date(2021,11,15), c = 'black')
    plt.title(category+'(1STEP)')
    if save == True:
        plt.savefig(save_path +str(category)+'_(1STEP).png',bbox_inches='tight', facecolor='w')
    plt.show()
    

    
def category_data_multi(data, x, sequence_length, lag):
    d = data[data['item_category']==x].drop(['item_category', 'base_date'], axis=1)
    # d = d[['판매수량']]
    d = d[['판매수량']]

    x = d.values
    y = d.values

    x_seq = []
    y_seq = []

    for i in range(len(x) - sequence_length):
        if len(y[i+sequence_length:(i+sequence_length+lag)])==lag:
            x_seq.append(x[i: (i+sequence_length+lag-1)])
            y_seq.append(y[i+sequence_length:(i+sequence_length+lag)])
        else:
            break
    
    return torch.FloatTensor(x_seq), torch.FloatTensor(y_seq).view([-1, lag]) # float형 tensor로 변형

# metrics(평가지표)
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
def MAE(y_test, y_pred): 
    return np.abs(y_test - y_pred).mean()

# SMAPE proposed by Makridakis (1993): 0%-200%
def smape_original(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

# adjusted SMAPE version to scale metric from 0%-100%
def smape_adjusted(a, f):
    return (1/a.size * np.sum(np.abs(f-a) / (np.abs(a) + np.abs(f))*100))


class Encoder(nn.Module): # LSTM 모델/ 마지막 LSTM layer의 output에서 다음 시점의 예측값 산출
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.num_classes = num_classes # output 개수
        self.num_layers = num_layers # layer
        self.input_size = input_size # feature 개수
        self.hidden_size = hidden_size # hidden unit 개수

        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size,
                      num_layers=num_layers, batch_first=True)
        self.fc_layer =  nn.Linear(hidden_size, hidden_size) 
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
       
        output, (hn, cn) = self.lstm(x) #lstm with input, hidden, and internal state
        out = output[:,-1] 
        out = out.view(-1, self.hidden_size)
        out = self.fc_layer(out) # 비선형성 더하기 위해 linear layer + relu 추가
        out = self.relu(out)
        out = self.output_layer(out)
        return out, hn.view(1*self.num_layers, -1, self.hidden_size), cn.view(1*self.num_layers, -1, self.hidden_size)


    
def train_and_plot(data, i, test_size = 12, sequence_length = 12, lag = 1, input_size = 6, hidden_size = 64, num_layers = 1,
                   num_classes = 1, num_epochs = 1500,learning_rate = 1e-3, plot = True, save = False, model_save = False, train_loss_threshold=5000):
    
    model_save_path = '/home/langleman/TimeGAN4paper/result/original/model/{}_lag{}_total1/'.format(i, lag)
    os.makedirs(model_save_path, exist_ok=True)
    x_seq, y_seq = category_data_multi(data, i, sequence_length, lag)
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    
    x_train_seq = x_seq[:-test_size]
    y_train_seq = y_seq[:-test_size]
    
    x_test_seq = x_seq[-test_size:]
    y_test_seq = y_seq[-test_size:]

    train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
    test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size = x_train_seq.shape[0], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size = x_test_seq.shape[0], shuffle=False)



    model = Encoder(num_classes=num_classes, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    train_loss_epoch = np.inf
    best_test_loss = np.inf
    best_epoch = 0

    for epoch in range(num_epochs):
 # train part

        train_loss = 0
        model.train()
        for x, y in train_loader:
            x_train = x.to(device)
            y_train = y.to(device)

            pred = model(x_train)
            pred = pred[0]
            y_pred = pred.view(-1)
            y_real = y_train.view(-1)

            loss = loss_function(y_pred,y_real)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            train_loss += loss.item()

        train_loss_epoch = train_loss/len(train_loader)

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for x, y in test_loader:
                model.to(device)
                x_test = x.to(device)
                y_test = y.to(device)

                test_pred = model(x_test)
                test_pred = test_pred[0]
                y_pred_test = test_pred.view(-1)
                y_real_test = y_test.view(-1)

                loss = loss_function(y_pred_test,y_real_test)      

                test_loss += loss.item()

        test_loss_epoch = test_loss/len(test_loader)

        best_state_dict = model.state_dict()                                       #수정
        


        if train_loss_epoch>train_loss_threshold:
            print(f'epoch{epoch+1}\tTrain_loss: {train_loss_epoch}')
            
        else:
            if best_test_loss>test_loss_epoch: # 모델 save(default=False)
                best_test_loss = test_loss_epoch
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

                print(f'Epoch{epoch+1}      train_loss: {train_loss_epoch}      test loss: {test_loss_epoch}')
                print()

                if epoch - best_epoch>=2000:
                    print(f'best_epoch: {best_epoch+1}')
                    break

    torch.save(best_state_dict,f'{model_save_path}/torch_best_model.pt')
    
    model = Encoder(num_classes=num_classes, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    # model = Encoder(num_classes=num_classes, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(f'{model_save_path}/torch_best_model.pt'))
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            model.to(device)
            x_test = x.to(device)
            y_test = y.to(device)
            
        
            
            test_pred = model(x_test)
            test_pred = test_pred[0]
            y_pred_test = test_pred
            y_real_test = y_test

            loss = loss_function(y_pred_test,y_real_test)      
            test_loss += loss.item()

    y_real_test = y_test[:,-1].detach().cpu().numpy()
    y_pred_test = test_pred[:,-1].detach().cpu().numpy()
    y_real_train = y_train[:,-1].detach().cpu().numpy()
    y_pred_train = pred[:,-1].detach().cpu().numpy()
    
    print(i + '_' + 'SMAPE :', smape_adjusted(y_real_test, y_pred_test))
    print(i + '_' + 'MAE :', MAE(y_real_test, y_pred_test))
    print(i + '_' + 'R^2 :', r2_score(y_real_test, y_pred_test))
    
    mape_r = smape_original(y_real_test, y_pred_test)
    mape_a = smape_adjusted(y_real_test, y_pred_test)
    mae_r = MAE(y_real_test, y_pred_test)
    rsquare_r = r2_score(y_real_test, y_pred_test)

    train_mape_r = smape_original(y_real_train, y_pred_train)
    train_mape_a = smape_adjusted(y_real_train, y_pred_train)
    train_mae_r = MAE(y_real_train, y_pred_train)
    train_rsquare_r = r2_score(y_real_train, y_pred_train)

    if plot==True:
        # y_real_train = y_train[:,-1].detach().numpy().reshape(-1)
        y_real_train = data[data['item_category']==i]['판매수량'][:-test_size]
        
        y_pred_train = y_pred_train.reshape(-1)
        y_real_test = y_real_test.reshape(-1)
        y_pred_test = y_pred_test.reshape(-1)

        make_plot(list(y_real_train), list(y_real_test), list(y_pred_test), i,save = save)
  
    return y_real_train,y_pred_train,y_real_test,y_pred_test, mape_r, mape_a, mae_r, rsquare_r, train_mape_r, train_mape_a, train_mae_r, train_rsquare_r

step1_metrics = np.zeros((13,8))
step1_metrics = pd.DataFrame(step1_metrics, index = category_list, columns = ['smape', 'smape_a', 'mae', 'R', 'train_smape','train_smape_a', 'train_mae', 'train_R'])
step1_metrics = step1_metrics.loc[['JK', 'JP', 'BL','VT','SL','SK','OP','TC','CT','KT','KC','TS','PD']]

def main():
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
    

    #epoch_dict = {'BL' : 1, 'PD' : 1, 'CT' : 8, 'JP' : 1, 'KC' : 2, 'TC' :1, 'SK' : 3, 'VT' : 4, 'KT' : 8, 'OP' : 1, 'JK' : 2, 'SL' : 7, 'TS' : 1}
    epoch_dict = {'BL' : 15000, 'PD' : 15000, 'CT' : 80000, 'JP' : 10000, 'KC' : 20000, 'TC' :10000, 'SK' : 30000, 'VT' : 40000, 'KT' : 80000, 'OP' : 10000, 'JK' : 20000, 'SL' : 70000, 'TS' : 100000}
    learning_rate_dict = {'BL' : 1e-4, 'PD' : 5e-5, 'CT' : 0.00002, 'JP' : 1e-4, 'KC' : 0.00005, 'TC' : 1e-4, 'SK' : 0.00001, 'VT' : 0.00001, 'KT' : 0.00001, 'OP' : 1e-4, 'JK' : 1e-4, 'SL' : 0.0001, 'TS' : 0.0001}

    for cate in tqdm(category_list):
        real_train, pred_train, real_test, pred_test, mape, mape_a, mae, r, tmape, tmape_a, tmae, tr = train_and_plot(data, cate, test_size = test_size, sequence_length = sequence_length, lag = lag, input_size = input_size, 
               hidden_size = hidden_size, num_layers = num_layers, num_classes = num_classes, num_epochs = epoch_dict[cate], learning_rate = learning_rate_dict[cate], plot = True, save = True, model_save = True,train_loss_threshold=threshold)
        step1_metrics.loc[cate] = [mape, mape_a, mae, r, tmape, tmape_a, tmae, tr]
    step1_metrics.to_csv(csv_save_path)
if __name__ == '__main__':
    main() 








