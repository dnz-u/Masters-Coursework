import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cm_hm(name, cm):
    _ = plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
                yticklabels=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
    plt.title(f"{name}: Confusion Matrix")
    return _

def plot_loss(name, train_data, test_data, comparison_data_name: str):
    _ = plt.figure(figsize=(12, 8))

    x_arr = [i for i in range(len(train_data))]

    plt.plot(x_arr, train_data, label='Training')
    plt.plot(x_arr, test_data, label=f'{comparison_data_name}')

    plt.xlabel("n'th Epoch")
    plt.title(f'{name} : Average Loss over Epochs\n(Total Loss / # of Samples )')
#    plt.xticks(x_arr)

    plt.legend()

    return _

def plot_acc(name, train_data, test_data, comparison_data_name: str):
    _ = plt.figure(figsize=(12, 8))

    train_data = np.array(train_data).T
    test_data = np.array(test_data).T

    x_arr = [i for i in range(len(train_data[0]))]

    # Plotting accuracy in blue
    for i in reversed(range(len(train_data))):
        plt.plot(x_arr, train_data[i], label=f'Top {i+1}, Training')
        plt.plot(x_arr, test_data[i], label=f'Top {i+1}, {comparison_data_name}')

    plt.xlabel("n'th Epoch")
    plt.title(f'{name} : Accuracy (%) over Epochs')
    #plt.xticks(x_arr)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    return _

def plot_outputs(data_dict, name:str):

    _ = plt.figure()

    first_columns = [value[:, 0] for key, value in data_dict.items()]

    data_matrix = np.vstack(first_columns)

    cmap = 'magma'

    plt.imshow(data_matrix, aspect='auto', cmap=cmap)

    plt.colorbar(label='First Column Values')

    plt.xlabel('Hidden Layer (Tanh) Outputs')
    plt.ylabel('Time Step')
    plt.title(f'{name}: Activations for Each Time Step ')

    return _

def plot_weights(data_matrix, name: str):
    _ = plt.figure()

    cmap = 'magma'
    plt.imshow(data_matrix, aspect='auto', cmap=cmap, extent=[0, data_matrix.shape[1], 0, data_matrix.shape[0]])
    cbar = plt.colorbar(label='Values')

    plt.xlabel('Neuron weights: w1,w2, ... wn')
    plt.ylabel('Neurons')
    plt.title(f'{name}: Outputs Heatmap')

    return _

def close_p(f):
    plt.close(f)