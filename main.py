import os

import yaml

import torch
from torch import nn, optim
from tqdm import tqdm
from load import addition_task
from model import SVDRnn


def train(model, paras):
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.MSELoss()
    train_data_loader, _ = addition_task(paras[paras['task_name']]['data_dir'],
                                         batch_size=paras[paras['task_name']]['batch_size'],
                                         train_num=paras[paras['task_name']]['train_num'],
                                         test_num=paras[paras['task_name']]['test_num'])
    save_dir = paras[paras['task_name']]['save_dir']

    model.train()
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
                y_hat = model(x)
            loss = loss_function(y, y_hat)
            loss.backward()
            optimizer.step()
            model.control_sigma(paras['r'])
            total_loss.append(loss)

        print(f"Epoch: {i}, MSE: {sum(total_loss) / len(total_loss)}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_{i}.pt"))
    torch.save(model.state_dict(), os.path.join(save_dir, f"final.pt"))


def evaluate(model, paras):
    loss_function = nn.MSELoss()
    _, test_data_loader = addition_task(paras[paras['task_name']]['data_dir'],
                                        batch_size=paras[paras['task_name']]['batch_size'],
                                        train_num=paras[paras['task_name']]['train_num'],
                                        test_num=paras[paras['task_name']]['test_num'])
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
    with open('config.yaml', 'r') as f:
        paras = yaml.safe_load(f)
    model = SVDRnn(paras[paras['task_name']]['input_size'],
                   paras['hidden_size'],
                   paras[paras['task_name']]['output_size'],
                   k1=paras['k1'],
                   k2=paras['k2'],
                   device=paras['device'])
    if paras['eval']:
        model.load_state_dict(torch.load(paras[paras['task_name']]['model_dir']))
        evaluate(model, paras)
    else:
        if not paras['train_from_scratch']:
            model.load_state_dict(torch.load(paras[paras['task_name']]['model_dir']))
        train(model, paras)


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
