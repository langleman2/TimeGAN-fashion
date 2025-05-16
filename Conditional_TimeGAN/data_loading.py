"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np



def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','BL_weekly_new_train', 'CT_weekly_new_train', 'JK_weekly_new_train', 'JP_weekly_new_train', 'KC_weekly_new_train', 'TS_weekly_new_train',
               'KT_weekly_new_train','OP_weekly_new_train','PD_weekly_new_train','SK_weekly_new_train','SL_weekly_new_train','TC_weekly_new_train','VT_weekly_new_train' ]
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)

  elif data_name == 'BL':                                                    #추가
    ori_data = np.loadtxt('data/BL.csv', delimiter = ",",skiprows = 1)  
  elif data_name == 'JP':                                                    #추가
    ori_data = np.loadtxt('data/JP.csv', delimiter = ",",skiprows = 1)  

  elif data_name == 'BL_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/BL_weekly_new_train.csv', delimiter = ",",skiprows = 1)  
  elif data_name == 'CT_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/CT_weekly_new_train.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'JK_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/JK_weekly_new_train.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'JP_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/JP_weekly_new_train.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'KC_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/KC_weekly_new_train.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'KT_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/KT_weekly_new_train.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'OP_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/OP_weekly_new_train.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'PD_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/PD_weekly_new_train.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'SK_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/SK_weekly_new_train.csv', delimiter = ",",skiprows = 1)  
  elif data_name == 'SL_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/SL_weekly_new_train.csv', delimiter = ",",skiprows = 1)  
  elif data_name == 'TC_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/TC_weekly_new_train.csv', delimiter = ",",skiprows = 1)  
  elif data_name == 'TS_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/TS_weekly_new_train.csv', delimiter = ",",skiprows = 1)  
  elif data_name == 'VT_weekly_new_train':                                                    #추가
    ori_data = np.loadtxt('data/VT_weekly_new_train.csv', delimiter = ",",skiprows = 1)  

    
  # Flip the data to make chronological data
  #ori_data = ori_data[::-1]  #수정... ####################주의

  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
  # print(ori_data[:10])    #10개만 원본 데이터 출력

  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data