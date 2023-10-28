from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def reshape_with_predictions(train, test, n_timesteps_in, n_timesteps_out):
  sc = StandardScaler()
  X_train = sc.fit_transform(train.values)
  X_test = sc.transform(test.values)
  X_train = torch.tensor(X_train).float()
  X_test = torch.tensor(X_test).float()

  y_train = torch.tensor(train.values)[:,-1].float()
  y_test = torch.tensor(test.values)[:,-1].float()
  y_train = y_train.view(y_train.shape[0], 1)
  y_test = y_test.view(y_test.shape[0], 1)

  def reshape_data_with_predictions(x, y, past, fut, lag):
      num_samples, num_features = x.shape
      x_new = np.zeros((num_samples - past + 1 -fut -lag, past, num_features)).astype('float32')
      preds = np.zeros((num_samples - past + 1 -fut -lag, fut, num_features - 1)).astype('float32')
      y_new = np.zeros((num_samples - past + 1 -fut -lag, fut, 1)).astype('float32')
      print(x_new.shape)
      for i in range(0, x_new.shape[0]):
          x_new[i, :, :num_features] = x[i:i + past, :]
          preds[i, :, :] = x[i + past - 1 + lag : i + past -1 + lag + fut , :-1]
          y_new[i, :, :] = y[i + past - 1 + lag : i + past -1 + lag + fut , 0].view(fut,1)
      return x_new, y_new, preds

  X_test_new, y_test_new, preds_test = reshape_data_with_predictions(X_test, y_test, n_timesteps_in, n_timesteps_out, 1)
  X_train_new, y_train_new, preds_train = reshape_data_with_predictions(X_train, y_train, n_timesteps_in, n_timesteps_out, 1)
  return X_train_new, y_train_new, preds_train, X_test_new, y_test_new, preds_test