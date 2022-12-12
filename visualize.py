import matplotlib.pyplot as plt
import numpy as np

names = ['RNN', 'SVD', 'LSTM']


def grad(task_name='addition_task'):
    grads = {}
    for name in names:
        grads[name] = np.mean(np.loadtxt(f'visualization/{task_name}/10/{name}_grads').reshape(196, -1), axis=0)
        # grads[name] = np.loadtxt(f'visualization/{task_name}/{name}_grads')
        plt.plot(grads[name], label=name)
    # plt.yscale('log')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('L2-norm')
    plt.title('gradient of transition matrix')
    plt.show()


def loss(task_name='addition_task'):
    losses = {}
    for name in names:
        losses[name] = np.loadtxt(f'visualization/{task_name}/10/{name}_losses')
        plt.plot(losses[name], label=name)
    if task_name == 'addition_task':
        plt.plot([0.166666667 for _ in range(1000)], label='baseline')
    else:
        plt.plot([0.166666667 for _ in range(1000)], label='baseline')

    # plt.yscale('log')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('')
    plt.title('loss')
    plt.show()


task = 'addition_task'
# task = 'copy_task'
grad(task)
loss(task)
