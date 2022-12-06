import matplotlib.pyplot as plt
import numpy as np

names = ['RNN', 'SVD', 'LSTM']


def grad(task_name='addition_task'):
    grads = {}
    for name in names:
        grads[name] = np.loadtxt(f'visualization/{task_name}/{name}_grads')
        plt.plot(grads[name], label=name)

    plt.yscale('log')
    plt.legend()
    plt.xlabel('batches x 256')
    plt.ylabel('L2-norm')
    plt.title('gradient of transition matrix')
    plt.show()


def loss(task_name='addition_task'):
    losses = {}
    for name in names:
        losses[name] = np.loadtxt(f'visualization/{task_name}/{name}_losses')
        plt.plot(losses[name], label=name)

    plt.yscale('log')
    plt.legend()
    plt.xlabel('batches x 256')
    plt.ylabel('')
    plt.title('loss')
    plt.show()


grad('copy_task')
loss('copy_task')

