def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    #sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    #obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    #sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    #obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

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