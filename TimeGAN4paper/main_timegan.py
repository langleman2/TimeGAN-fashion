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
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
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

  cat = args.cat


  ## Data loading
  if args.data_name in [f'{cat}_weekly_new_train']:
    ori_data = real_data_loading(args.data_name, args.seq_len)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
    
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size

  generated_data = timegan(ori_data, parameters)   


  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()

  ###########################################################################
  generated_data1 = generated_data.reshape(generated_data.shape[0], -1)

  generated_df = pd.DataFrame(generated_data1)

  # Define the file path where you want to save the CSV file
  output_file_path = f'/home/langleman/TimeGAN4paper/generated_data/data/generated_data_new_{cat}_1231.csv'

  # Save the DataFrame to a CSV file
  generated_df.to_csv(output_file_path, index=False)

  print('Finish saving')
 
  ###################################################################333 
  
#   # 1. Discriminative Score
#   discriminative_score = list()
#   for _ in range(args.metric_iteration):
#     temp_disc = discriminative_score_metrics(ori_data, generated_data)
#     discriminative_score.append(temp_disc)
      
#   metric_results['discriminative'] = np.mean(discriminative_score)
      
#   # 2. Predictive score
#   predictive_score = list()
#   for tt in range(args.metric_iteration):
#     temp_pred = predictive_score_metrics(ori_data, generated_data)
#     predictive_score.append(temp_pred)   
      
#   metric_results['predictive'] = np.mean(predictive_score)     
          
#   # 3. Visualization (PCA and tSNE)
#   visualization(ori_data, generated_data, 'pca', cat)
#   visualization(ori_data, generated_data, 'tsne', cat)
  
#   ## Print discriminative and predictive scores
#   print(metric_results)

# # Convert metric_results dictionary to DataFrame
#   df_metric_results = pd.DataFrame(metric_results, index=[0])

#   csv_save_path =f'/home/langleman/TimeGAN4paper/metric_generated/{cat}/{cat}_metric_results.csv'
#   df_metric_results.to_csv(csv_save_path)





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
      default=72, #12
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
  

#추가함 내가
  parser.add_argument(
      '--cat',
      choices = ['BL', 'CT', 'JK', 'JP', 'KC', 'KT', 'OP', 'PD', 'SK', 'SL', 'TC', 'TS', 'VT'],
      help='catagory for clothes',
      default='BL', 
      type=str) 


  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)





