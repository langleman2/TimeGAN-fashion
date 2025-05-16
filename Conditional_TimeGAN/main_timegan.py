"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from conditional_timegan_gpt import timegan_conditional
# 2. Data loading
from data_loading_conditional import real_data_loading
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """

#  cat = args.cat


  ## Data loading

  #ori_data = real_data_loading(['SK_weekly_new_train', 'SL_weekly_new_train', 'TC_weekly_new_train', 'VT_weekly_new_train'], args.seq_len)
    
  ori_data = real_data_loading(['BL_weekly_new_train', 'CT_weekly_new_train', 'JK_weekly_new_train',
    'JP_weekly_new_train', 'KC_weekly_new_train', 'TS_weekly_new_train',
    'KT_weekly_new_train', 'OP_weekly_new_train', 'PD_weekly_new_train',
    'SK_weekly_new_train', 'SL_weekly_new_train', 'TC_weekly_new_train',
    'VT_weekly_new_train'], args.seq_len)
  #breakpoint()
  print(f'ori_data shape : {ori_data.shape}')
    
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size

  categories = [0]*12 + [1]*12 + [2]*12 + [3]*12 +[4]*12 +[5]*12 +[6]*12 +[7]*12 +[8]*12 +[9]*12 +[10]*12 +[11]*12 +[12]*12
  #categories = [0]*12 + [1]*12 + [2]*12 + [3]*12

  generated_data = timegan_conditional(ori_data, categories, parameters)   


  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  # 생성된 데이터
  print('generated_data')
  #print(generated_data)
  print(len(generated_data))

    
  # 데이터 구조 확인
  print('generated_data')
  #print(generated_data)  # 첫 두 항목만 출력하여 구조 확인
  print(f'Number of elements: {len(generated_data)}')
  print(type(generated_data))

  ###################################################################333 
      
  # 데이터 구조 확인
  print('generated_data')
  print(f'Number of elements: {len(generated_data)}')
  print(type(generated_data))
  for i in range(len(generated_data)):
      print(generated_data[i].shape)

  # 3D 배열을 2D 배열로 변환
  # 각 항목의 shape가 (354, 1)이므로, reshape을 통해 (354,)로 변환
  generated_data_array = np.array([arr.reshape(-1) for arr in generated_data])

  # 변환된 배열의 형태 확인
  print(f'Array shape: {generated_data_array.shape}')

  # NumPy 배열을 DataFrame으로 변환
  generated_df = pd.DataFrame(generated_data_array)

  # CSV 파일 경로 지정
  output_file_path = '/home/langleman/Conditional_TimeGAN/generated_data/data/generated_data_new_Condition_1118_b156_h18_nl18.csv'

  # DataFrame을 CSV 파일로 저장 (숫자만 포함된 형태)
  generated_df.to_csv(output_file_path, index=False, header=False)



  print('저장 완료')



  
# metric & visual 자리

# Convert metric_results dictionary to DataFrame
  df_metric_results = pd.DataFrame(metric_results, index=[0])

  csv_save_path =f'/home/langleman/Conditional_TimeGAN/metric_generated/condition_metric_results.csv'
  df_metric_results.to_csv(csv_save_path)





  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['BL_weekly_new_train', 'CT_weekly_new_train', 'JK_weekly_new_train', 'JP_weekly_new_train', 'KC_weekly_new_train', 'TS_weekly_new_train',
               'KT_weekly_new_train','OP_weekly_new_train','PD_weekly_new_train','SK_weekly_new_train','SL_weekly_new_train','TC_weekly_new_train','VT_weekly_new_train'],
      default='BL_weekly_new_train',
      type=str)
  parser.add_argument(
      '--seq_len', 
      help='sequence length',
      default=354, #12
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=12, #12
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=100, #50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default= 16, #24
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10, #10
      type=int)
  

# #추가함 내가
#   parser.add_argument(
#       '--cat',
#       choices = ['BL', 'CT', 'JK', 'JP', 'KC', 'KT', 'OP', 'PD', 'SK', 'SL', 'TC', 'TS', 'VT'],
#       help='catagory for clothes',
#       default='BL', 
#       type=str) 


  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)





