import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

import torch
from torch import nn, optim
from tqdm import tqdm
from load import addition_task, copy_task
from model import SVDRnn, LSTM, RNN


def visualize(data):
    plt.plot(np.linspace(0, len(data)), data)
    plt.show()


def train(model, paras):
    optimizer = optim.Adam(model.parameters())
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
    grads_W = []
    batch_loss = []
    for i in range(paras['epoch']):
        total_loss = []
        for batch in tqdm(train_data_loader):
            x, y = batch
            x = x.to(paras['device'])
            y = y.to(paras['device'])
            optimizer.zero_grad()
            if paras['task_name'] == 'addition_task':
                y_hat = model(x)[:, -1]
            else:
                y_hat = model(x)[:, -10:]
            loss = loss_function(y, y_hat)
            # model.W.register_hook(lambda g: grads_W.append(torch.norm(g).to('cpu')))
            loss.backward()
            optimizer.step()
            model.control_sigma(paras['r'])
            total_loss.append(float(loss))
            # batch_loss.append(loss.to('cpu'))

        print(f"Epoch: {i}, MSE: {sum(total_loss) / len(total_loss)}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_{i}.pt"))
    torch.save(model.state_dict(), os.path.join(save_dir, f"final.pt"))
    # visualize(np.array(grads_W))
    # visualize(np.arraybatch_loss)


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
            y_hat = model(x)
        loss = loss_function(y, y_hat)
        total_loss.append(loss)
    print(f"MSE: {sum(total_loss) / len(total_loss)}")


def main():
    with open('config/config.yaml', 'r') as f:
        paras = yaml.safe_load(f)
    model = SVDRnn(paras[paras['task_name']]['input_size'],
                   paras['hidden_size'],
                   paras[paras['task_name']]['output_size'],
                   k1=paras['k1'],
                   k2=paras['k2'],
                   device=paras['device'])
    if paras['eval_only']:
        model.load_state_dict(torch.load(paras[paras['task_name']]['model_dir']))
    else:
        if not paras['train_from_scratch']:
            model.load_state_dict(torch.load(paras[paras['task_name']]['model_dir']))
        train(model, paras)
    evaluate(model, paras)


if __name__ == '__main__':
    main()

    # parameters = {
    #     'k1': 2,
    #     'k2': 3,
    #     'r': 0.01,
    #     'hidden_size': 10,
    #     'device': 'cuda',
    #     'epoch': 50,
    #     'task_name': 'addition_task',
    #     'eval': True,
    #     'train_from_scratch': True,
    #     'addition_task': {
    #         'input_size': 2,
    #         'output_size': 1,
    #         'save_dir': 'addition_task_weights',
    #         'data_dir': 'data/Adding_task/data100',
    #         'model_dir': 'addition_task_weights/final.pt',
    #         'train_num': 10000,
    #         'test_num': 1000,
    #         'batch_size': 64
    #     },
    # }
    #
    # with open('config.yaml', 'w') as f:
    #     data = yaml.dump(parameters, f)
