import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def calc_nse(obs: np.array, sim: np.array) -> float:
 
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_FHV(obs, preds):
  indices = np.argsort(obs)
  sorted_obs = obs[indices]
  sorted_preds = preds[indices]
  top_2_percent_index = int(0.02 * len(sorted_obs))
  top_obs = sorted_obs[-top_2_percent_index:]
  top_preds = sorted_preds[-top_2_percent_index:]
  num = np.sum(top_preds-top_obs)
  den = np.sum(top_obs)
  FHV = num/den*100
  return FHV




def calcolo_soglia_1(obs, preds, indeces, counter, soglia_1):
    df = indeces[counter]
    diff = df[n_timesteps_in:-n_timesteps_out].shape[0] - obs.shape[0]
    sku = pd.to_datetime(df.ISTANTE[n_timesteps_in:-n_timesteps_out-diff].values)
    dffone = pd.DataFrame(index=sku, data={'obs': obs, 'preds': preds})
    PP = 0
    FP = 0
    FN = 0
    NN = 0
    for day, day_data in dffone.groupby(dffone.index.date):
      superamento_previsto = np.any(day_data.preds.values > soglia_1)
      superamento_reale = np.any(day_data.obs.values > soglia_1)
      if superamento_reale and superamento_previsto:
        PP += 1
      if superamento_reale and not superamento_previsto:
        FN += 1
      if not superamento_reale and superamento_previsto:
        FP += 1
      if not superamento_reale and not superamento_previsto:
        NN += 1
    return dffone, (PP,FP,FN,NN)

def calcolo_soglia_2(final_obs, final_preds, soglia, margine):
  window = 24*7
  prob = np.where(final_obs > soglia)
  con = np.array([])
  eventi_pos = []
  for i in prob[0]:
    if not np.isin(i,con):
      missing = np.arange(i-window,i+window)
      con = np.concatenate([con,missing])
      eventi_pos.append(missing)
  window = 24*7
  prob = np.where(final_preds > soglia)
  con = np.array([])
  eventi_pos_preds = []
  for i in prob[0]:
    if not np.isin(i,con):
      missing = np.arange(i-window,i+window)
      con = np.concatenate([con,missing])
      eventi_pos_preds.append(missing)
  PP = 0
  FN = 0
  for i in eventi_pos:
    is_predicted = np.any(final_preds[i] > soglia - margine)
    if is_predicted:
      PP += 1
    else:
      FN += 1
  FP = 0
  for i in eventi_pos_preds:
    is_predicted = np.any(final_obs[i] > soglia - margine)
    if not is_predicted:
      FP += 1
  return PP, FN, FP
    
def final_plot(obs, preds, n_timesteps_out, soglia, margine):
  cat_obs = []
  cat_preds = []
  c = 0
  for i in range(obs.shape[1]//24):
    cat_obs.append(obs[n_timesteps_out-24:,c*24,:])
    cat_preds.append(preds[n_timesteps_out-24:,c*24,:])
    c += 1
  final_obs = torch.cat(cat_obs, dim=0).flatten().numpy()
  final_preds = torch.cat(cat_preds, dim=0).flatten().numpy()
  plt.figure(figsize=(20,10))
  plt.plot(final_obs, alpha=0.4)
  plt.plot(final_preds)
  plt.axhline(y=soglia-margine, color='r', linestyle='--')
  plt.show()
  return final_obs, final_preds