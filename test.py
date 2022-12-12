import matplotlib.pyplot as plt
import numpy as np
import yaml

import torch
from torch import nn, optim
from tqdm import tqdm
from load import addition_task, copy_task
from model import SVDRnn, LSTM, RNN


def visualize(data):
    X = np.arange(0, len(data))
    plt.plot(X, data)
    plt.show()


def train(model, paras):
    optimizer = optim.SGD(model.parameters(), lr=paras['lr'])
    loss_function = nn.MSELoss()
    if paras['task_name'] == 'addition_task':
        train_data_loader, _ = addition_task(paras['addition_task']['data_dir'],
                                             batch_size=paras['addition_task']['batch_size'],
                                             train_num=paras['addition_task']['train_num'],
                                             test_num=paras['addition_task']['test_num'])
    else:
        train_data_loader, _ = copy_task(paras['copy_task']['data_dir'],
                                         batch_size=paras['copy_task']['batch_size'],
                                         train_num=paras['copy_task']['train_num'],
                                         test_num=paras['copy_task']['test_num'])
    save_dir = paras[paras['task_name']]['save_dir']

    model.train()
    grads = []
    weights = []
    losses = []
    for i in range(paras['epoch']):
        total_loss = []
        # for batch in tqdm(train_data_loader):
        for batch in train_data_loader:
            x, y = batch
            x = x.to(paras['device'])
            y = y.to(paras['device'])
            optimizer.zero_grad()
            if paras['task_name'] == 'addition_task':
                y_hat = model(x)[:, -1]
                # y_hat = torch.squeeze(model(x), dim=-1)
            else:
                y_hat = model(x)[:, -10:]
            loss = loss_function(y, y_hat)
            loss.backward()
            grads.append(torch.norm(model.rnn.weight_hh_l0.grad, p=2).to('cpu').detach())
            optimizer.step()
            total_loss.append(float(loss))

        print(f"Epoch: {i}, MSE: {sum(total_loss) / len(total_loss)}")
        losses.append(sum(total_loss) / len(total_loss))
    grads = np.array(grads)
    losses = np.array(losses)
    visualize(grads)
    visualize(losses)
    np.savetxt(f'visualization/{paras["task_name"]}/{name}_grads', grads)
    np.savetxt(f'visualization/{paras["task_name"]}/{name}_losses', losses)


def evaluate(model, paras):
    loss_function = nn.MSELoss()
    if paras['task_name'] == 'addition_task':
        _, test_data_loader = addition_task(paras['addition_task']['data_dir'],
                                            batch_size=paras['addition_task']['batch_size'],
                                            train_num=paras['addition_task']['train_num'],
                                            test_num=paras['addition_task']['test_num'])
    else:
        _, test_data_loader = copy_task(paras['copy_task']['data_dir'],
                                        batch_size=paras['copy_task']['batch_size'],
                                        train_num=paras['copy_task']['train_num'],
                                        test_num=paras['copy_task']['test_num'])
    model.eval()
    total_loss = []
    for batch in tqdm(test_data_loader):
        x, y = batch
        x = x.to(paras['device'])
        y = y.to(paras['device'])
        if paras['task_name'] == 'addition_task':
            y_hat = model(x)[:, -1]
        else:
            y_hat = model(x)[:, -10:]
        loss = loss_function(y, y_hat)
        total_loss.append(loss)
    print(f"MSE: {sum(total_loss) / len(total_loss)}")


def main():
    if name == 'RNN':
        rnn = RNN
    else:
        rnn = LSTM
    with open('config/config.yaml', 'r') as f:
        paras = yaml.safe_load(f)
    model = rnn(paras[paras['task_name']]['input_size'],
                paras['hidden_size'],
                paras[paras['task_name']]['output_size']
                ).to(paras['device'])
    train(model, paras)
    evaluate(model, paras)


if __name__ == '__main__':
    # name = 'RNN'
    name = 'LSTM'
    main()
