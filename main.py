import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import reshape_with_predictions
from metrics import calc_nse, calc_FHV, calcolo_soglia_1, calcolo_soglia_2, final_plot
from loader import ManualDataLoader
from model import Model_preds

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_model_preds(model, loader):
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, fut, ys in loader:
            # push data to GPU (if available)

            xs = xs.transpose(0,1).to(DEVICE)
            fut = fut.transpose(0,1).to(DEVICE)
            ys = ys.transpose(0,1).to(DEVICE)
            # get model predictions
            y_hat = model.forward(xs,fut, xs.shape[1], n_features_out)
            y_hat = torch.cat(y_hat,0)

            obs.append(ys)
            preds.append(y_hat)


    return torch.cat(obs,1), torch.cat(preds,1)

def train_preds(X_train_new, preds_train, y_train_new, n_epochs, n_timesteps_out):
  tr_loader = ManualDataLoader(X_train_new, preds_train, y_train_new, batch_size=batch_size, shuffle = True)
  model = Model_preds(input_size = input_size, hidden_size = hidden_size, n_timesteps_out = n_timesteps_out).to(DEVICE)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  for epoch in range(n_epochs):
      model.train()
      for  i, (data, preds, labels) in enumerate(tr_loader):  #iterate through each batch
                  data = data.transpose(0,1)
                  preds = preds.transpose(0,1)
                  data = data.to(DEVICE)
                  labels = labels.to(DEVICE)
                  preds = preds.to(DEVICE)
                  y_pred = model.forward(data,preds,data.shape[1], n_features_out)
                  y_pred = torch.cat(y_pred,0).transpose(0,1)
                  loss = criterion(y_pred, labels)
                  loss.backward()
                  optimizer.step()
                  optimizer.zero_grad()

      obs, preds = eval_model_preds(model, tr_loader)
      obs = obs.cpu()
      preds = preds.cpu()
      nse2 = calc_nse(obs.numpy(),preds.numpy())
      print(f'Epoch - {epoch+1},  NSE_train - {round(nse2,4)} Loss - {round(loss.item(),3)}')
  return model

def test_preds(model, X_test_new, preds_train, y_test_new, soglia, margine):
  test_loader = ManualDataLoader(X_test_new, preds_test, y_test_new, batch_size=batch_size)
  obs, preds = eval_model_preds(model, test_loader)
  obs = obs.cpu()
  preds = preds.cpu()
  nse = calc_nse(obs.numpy(),preds.numpy())
  final_obs, final_preds = final_plot(obs, preds, n_timesteps_out, soglia, margine)
  PP, FN, FP = calcolo_soglia_2(final_obs, final_preds, soglia, margine)
  FHV = calc_FHV(final_obs, final_preds)
  return (nse, FHV, PP, FN, FP), final_obs, final_preds

if __name__ == '__main__':
    df3 = pd.read_csv('K1319_15_s687281_index.csv')
    df2 = pd.read_csv('K10_16_21_s687281_index.csv')
    df1 = pd.read_csv('K1218_17_22_s687281_index.csv')
    df4 = pd.read_csv('K1420_11_s687281_index.csv')

    K10_16_21 = pd.read_csv('K10_16_21_s687281.csv')
    K1218_17_22 = pd.read_csv('K1218_17_22_s687281.csv')
    K1319_15 = pd.read_csv('K1319_15_s687281.csv')
    K1420_11 = pd.read_csv('K1420_11_s687281.csv')

    n_epochs = 2
    n_timesteps_in = 72
    n_timesteps_out = 24
    input_size = 8
    n_features_out = 1
    hidden_size = 32
    batch_size = 2048
    soglia_1 = 1
    soglia_2 = 1.7
    margine = 0.05

    indeces = [df1, df2, df3, df4]
    todos = [K1218_17_22,K1319_15,K10_16_21,K1420_11]

    for i, K in enumerate(todos):

        train = pd.concat([K for j, K in enumerate(todos) if j!=i])
        test = K
        X_train_new, y_train_new, preds_train, X_test_new, y_test_new, preds_test = reshape_with_predictions(train, test, n_timesteps_in, n_timesteps_out)
        model = train_preds(X_train_new, preds_train, y_train_new, n_epochs, n_timesteps_out)
        results, obs, preds = test_preds(model, X_test_new,preds_train, y_test_new, soglia_2, margine)
        listone.append(results)
        dffone, matrix = calcolo_soglia_1(obs, preds, indeces, counter, soglia_1)
        matrice.append(matrix)
