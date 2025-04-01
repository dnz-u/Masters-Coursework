import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import gc  # garbage collector
import time

os.environ["OMP_NUM_THREADS"] = "2"

gc.collect()

FEATURE_SIZE = 3
OUTPUT_SIZE = 6
NUM_TIME_STEPS = 150

cur_run_seed = 682
np.random.seed(cur_run_seed)

def plot_loss(train_data, test_data):
    _ = plt.figure(figsize=(12, 8))

    x_arr = [i for i in range(len(train_data))]

    plt.plot(x_arr, train_data, label='Training')
    plt.plot(x_arr, test_data, label='Test')
    
    plt.xlabel("n'th Epoch")
    plt.title('Average Loss over Epochs\n(Total Loss / # of Samples )')
    plt.xticks(x_arr)

    plt.legend()

    return _

def plot_acc(train_data, test_data):
    _ = plt.figure(figsize=(12, 8))

    train_data = np.array(train_data).T
    test_data = np.array(test_data).T
    x_arr = [i for i in range(len(train_data[0]))]

    # Plotting accuracy in blue
    for i in reversed(range(len(train_data))):
            plt.plot(x_arr, train_data[i], label=f'Top {i+1}, Training')
            plt.plot(x_arr, test_data[i], label=f'Top {i+1}, Test')

    plt.xlabel("n'th Epoch")
    plt.title('Accuracy (%) over Epochs')
    plt.xticks(x_arr)
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


class DATA:
    def __init__(self):
        # input type (feature, sample)
        # input size 28 * 28 = 784
        self.x_train_orig = None
        self.y_train_orig = None
            
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        
        # Open the HDF5 file
        with h5py.File('data-Mini Project 2.h5', 'r') as file:
            # file.keys() = ['trX', 'trY', 'tstX', 'tstY']
            self.x_train = file['trX'][:]
            self.y_train = file['trY'][:]
            self.x_test = file['tstX'][:]
            self.y_test = file['tstY'][:]
            
  
# =============================================================================
#             
#         from sklearn.model_selection import train_test_split
#         self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
#                 self.x_train, self.y_train, 
#                 test_size=0.1, 
#                 stratify=self.y_train, random_state=cur_run_seed
#                 )
#       
#         self.x_val = np.transpose(self.x_val, (2,0,1))
#         self.y_val = self.y_val.T
# =============================================================================
        
        self.x_train = np.transpose(self.x_train, (2,0,1))
        self.y_train = self.y_train.T
        self.x_test = np.transpose(self.x_test, (2,0,1))
        self.y_test = self.y_test.T

        class_indices = np.argmax(self.y_train, axis=0)
        unique_classes, class_counts = np.unique(class_indices, return_counts=True)
        class_occurrences = dict(zip(unique_classes, class_counts))
        print("Number of occurrences: ", class_occurrences)
        gc.collect()
        self.test_data()
        
    def test_data(self):
        assert self.x_train.shape == (3, 3000, 150), self.x_train.shape
        assert self.y_train.shape == (OUTPUT_SIZE, 3000), self.y_train.shape
        assert self.x_test.shape == (3, 600, 150), self.x_test.shape
        assert self.y_test.shape == (OUTPUT_SIZE, 600), self.y_test.shape
# =============================================================================
#         assert self.x_val.shape == (3, 300, 150), self.x_val.shape
#         assert self.y_val.shape == (OUTPUT_SIZE, 300), self.y_val.shape
# =============================================================================


def mini_batch(xt, yt, mini_batch_size):
        """
        It is a generator functions that generates&yields mini batches

        """
        # m: number of examples
        m = yt.shape[1]

        number_of_mini_batches = int(m/mini_batch_size)

        for k in range(number_of_mini_batches): 
            mini_batch_X = xt[:, k*mini_batch_size : (k+1)*mini_batch_size, :]
            mini_batch_Y = yt[:, k*mini_batch_size : (k+1)*mini_batch_size]
            yield (mini_batch_X, mini_batch_Y)

        if (m % number_of_mini_batches) != 0:
            mini_batch_X = xt[:, number_of_mini_batches*mini_batch_size : m, :]
            mini_batch_Y = yt[:, number_of_mini_batches*mini_batch_size : m]
            yield (mini_batch_X, mini_batch_Y)
            
def sigmoid(z):
    return 1./(1 + np.exp(-z))

def cce_loss(y_pred, y_true, m):
    #_m = y_true.shape[1]
    #assert _m == m
    return -1 * np.sum( y_true *np.log(y_pred))

def _mini_batch(xt, yt, mini_batch_size, shuffle = None):
        """
        It is a generator functions that generates&yields mini batches

        """
        # Shuffle dataset
        def create_permutation(x, y):
            perm = np.random.permutation(x.shape[1])
            return x[:, perm, :], y[:, perm]

        if shuffle is not None:
            x_shuffle, y_shuffle = create_permutation(xt, yt)
        else:
            x_shuffle, y_shuffle = xt[:], yt[:] 

        # m: number of examples
        m = yt.shape[1]

        number_of_mini_batches = int(m/mini_batch_size)

        for k in range(number_of_mini_batches): 
            mini_batch_X = xt[:, k*mini_batch_size : (k+1)*mini_batch_size, :]
            mini_batch_Y = yt[:, k*mini_batch_size : (k+1)*mini_batch_size]
            yield (mini_batch_X, mini_batch_Y)

        if (m % number_of_mini_batches) != 0:
            mini_batch_X = xt[:, number_of_mini_batches*mini_batch_size : m, :]
            mini_batch_Y = yt[:, number_of_mini_batches*mini_batch_size : m]
            yield (mini_batch_X, mini_batch_Y)
            
class RNN:
    _shared_container = dict()
    _exp_no = 0
    
    @staticmethod
    def get_shared_container():
        return RNN._shared_container
    
    @staticmethod
    def get_exp_no():
        return RNN._exp_no
    
    def update_exp_no(self):
        RNN._exp_no += 1

    def __init__(self, name, N, FEATURE_SIZE, OUTPUT_SIZE, NUM_TIME_STEPS):
        self.update_exp_no()
        
        self.name = name
        
        self.data = None
        self.m = None
        self.batch_size = None
        
        self.hidden_size = N
        self.feature_size = FEATURE_SIZE
        self.output_size = OUTPUT_SIZE
        self.time_step = NUM_TIME_STEPS
        
        self.a_steps = dict()
        self.y_preds = dict()
        self.lr = None
        self.epochs = None
        
        self.Waa = None
        self.Wax = None
        self.Wya = None
        self.by = None
        self.ba = None
        
        self.is_parameters_initialized = False
        
        # training data
        self.batch_loss = []
        self.batch_acc = []
        
        self.epoch_train_loss = []
        self.epoch_train_acc = []
        
        self.epoch_val_loss = []
        self.epoch_val_acc = []
        
        self.epoch_test_loss = []
        self.epoch_test_acc = []
        
        self.best_Waa = None
        self.best_Wax = None
        self.best_Wya = None
        self.best_by = None
        self.best_ba = None
        
        self.best_acc = -1
        self.best_epoch = -1

        self.act1 = "tanh" 
        self.act2 = "sigmoid"
        
    def save_parameters(self, dir_name):
        np.save(f'{dir_name}/Waa_matrix', self.best_Waa)
        np.save(f'{dir_name}/Wax_matrix', self.best_Wax)
        np.save(f'{dir_name}/Wya_matrix', self.best_Wya)
        np.save(f'{dir_name}/by_matrix', self.best_by)
        np.save(f'{dir_name}/ba_matrix', self.best_ba)
      
    def load_best_params(self, dir_name):
        self.Waa = np.load(f'{dir_name}/Waa_matrix.npy')
        self.Wax = np.load(f'{dir_name}/Wax_matrix.npy')
        self.Wya = np.load(f'{dir_name}/Wya_matrix.npy')
        self.by = np.load(f'{dir_name}/by_matrix.npy')
        self.ba = np.load(f'{dir_name}/ba_matrix.npy')

    def copy_to_best(self):
        self.best_Waa = self.Waa.copy()
        self.best_Wax = self.Wax.copy()
        self.best_Wya = self.Wya.copy()
        self.best_by = self.by.copy()
        self.best_ba = self.ba.copy()
        
    def init_weights(self):
        interval = [-0.1, 0.1]
        self.Waa = np.random.uniform(*interval, (self.hidden_size, self.hidden_size))
        self.Wax = np.random.uniform(*interval, (self.hidden_size, self.feature_size))
        self.Wya = np.random.uniform(*interval, (self.output_size, self.hidden_size))
        
        self.ba = np.random.uniform(*interval, (self.hidden_size, 1))
        self.by = np.random.uniform(*interval, (self.output_size, 1))

    def init_a(self, sample_size):
        # initialize zero matrix
        a_init = np.zeros((self.hidden_size, sample_size))
        return a_init
    
    def prediction(self, x, y):
        # Forward Pass
        total_loss = 0  
        a_prev = self.init_a(y.shape[1])
        for t in range(self.time_step):
            a = np.tanh(np.dot(self.Waa, a_prev) + self.ba 
                        + np.dot(self.Wax, x[:, :, t]))
            a_prev = a.copy()
            
            y_hat = sigmoid(np.dot(self.Wya, a) + self.by)
            total_loss += cce_loss(y_hat, y, self.m)
        return y_hat, total_loss
        
    def get_num_correct(self, preds, y):
        pred_labels = np.argmax(preds, axis=0)
        y_labels = np.argmax(y, axis=0)
        # Calculate accuracy
        #acc = np.mean(np.equal(pred_labels, y_labels))
        acc = np.sum(np.equal(pred_labels, y_labels))
        return acc
    
    def get_top_n_correct(self, preds, y, n):
        "returns total number of correct predictions"
        top_n = np.argsort(preds, axis=0)[-n:, :]
        correct_preds = np.sum(top_n == np.argmax(y, axis=0), axis=0)
        total_correct = np.sum(correct_preds)
        return total_correct
        
    def forward_pass(self, x, y):
        # Forward Pass
        total_loss = 0  
        a_prev = self.init_a(self.m)
        for t in range(self.time_step):
            a = np.tanh(np.dot(self.Waa, a_prev) + self.ba 
                        + np.dot(self.Wax, x[:, :, t]))
            self.a_steps[t] = a.copy()
            
            y_hat = sigmoid(np.dot(self.Wya, a) + self.by)
            self.y_preds[t] = y_hat.copy()
            
            a_prev = a.copy()
            
            total_loss += cce_loss(y_hat, y, self.m)
        return y_hat, total_loss

    def bptt(self, x, y):
        # BPTT
        a_steps = self.a_steps
        y_preds = self.y_preds
        
        da_next = self.init_a(self.m)
        
        dWya = np.zeros_like(self.Wya)
        dWaa = np.zeros_like(self.Waa)
        dWax = np.zeros_like(self.Wax)
        dby = np.zeros_like(self.by)
        dba = np.zeros_like(self.ba)
        
        
        for t in reversed(range(self.time_step)):
            a_next = a_steps[t]
            
            if t == 0:
                a_prev = self.init_a(self.m)
            else:    
                a_prev = a_steps[t-1]
            
            dL_dz = y_preds[t] - y              # (output_size, sample_size)
            dWya += (1.0/self.m)*np.dot(dL_dz, a_next.T)     # (output_size, sample_size) x (sample_size, hidden_size)
            dby += (1.0/self.m)*np.sum(dL_dz, axis=1, keepdims=True)       # (output_size, 1)
            dya = np.dot(self.Wya.T, dL_dz)     # (hidden_size, output_size) x (output_size, sample_size)
            dL_da = dya + da_next               # (hidden_size, sample_size)
            dL_dz = np.multiply(dL_da, (1-np.square(a_next)))     # (hidden_size, sample_size)
            
            dWaa += (1.0/self.m)*np.dot(dL_dz, a_prev.T)     # (hidden_size, sample_size) x (sample_size, hidden_size)
            dba += (1.0/self.m)*np.sum(dL_dz, axis=1, keepdims=True)       # (hidden_size, 1)
            dWax += (1.0/self.m)*np.dot(dL_dz, x[:, :, t].T) # (hidden_size, sample_size) x (sample_size, feature_size)
            
            da_next = np.dot(self.Waa.T, dL_dz) # (hidden_size, hidden_size) x (hidden_size, sample_size)
        
        # gradient scaling
        f = 1./self.time_step # np.sqrt(1./self.m)
        dWya = f * dWya
        dWaa = f * dWaa
        dWax = f * dWax
        dby = f * dby
        dba = f * dba
        
         # gradient clipping
# =============================================================================
#         np.clip(dWya, -5, 5, out=dWya)
#         np.clip(dWaa, -5, 5, out=dWaa)
#         np.clip(dWax, -5, 5, out=dWax)
#         np.clip(dby, -5, 5, out=dby)
#         np.clip(dba, -5, 5, out=dba)
# =============================================================================
        
        # gradient updates
        self.Wya += -self.lr * dWya * f 
        self.Waa += -self.lr * dWaa * f
        self.Wax += -self.lr * dWax * f
        self.by += -self.lr * dby * f
        self.ba += -self.lr * dba * f
    
    def prompt_statistics(self, name:str, cur_epoch:int, top1_avg_loss:float,
                          top1_acc:float, top2_acc:float, top3_acc:float):
        print(f"\nEpoch [{cur_epoch}/{self.epochs}], "
              f"{name} \nAvg. (Top1) Loss: {top1_avg_loss:.4f}, "
              f"Total Accuracy -> "
              f"(Top1): {top1_acc:.4f}%, "
              f"(Top2): {top2_acc:.4f}%, "
              f"(Top3): {top3_acc:.4f}%")
        
    def calculate_statistics(self, x, y):
        preds, loss = self.prediction(x, y)
        m = y.shape[1]
        
        _acc = self.get_num_correct(preds, y)
        top1_acc = self.get_top_n_correct(preds, y, 1)
        assert np.isclose(_acc, top1_acc) == True
        
        top2_acc = self.get_top_n_correct(preds, y, 2)
        top3_acc = self.get_top_n_correct(preds, y, 3)

        return loss/m, 100*top1_acc/m, 100*top2_acc/m, 100*top3_acc/m
    
    def eval_all(self, epoch, train=True, val=True, test=True):
        if train:          
            training_metrics = self.calculate_statistics(self.data.x_train, self.data.y_train)
            self.prompt_statistics("Training", epoch, *training_metrics)
            # add
            self.epoch_train_loss.append(training_metrics[0])
            self.epoch_train_acc.append(training_metrics[1:])
            
        if test:
            test_metrics = self.calculate_statistics(self.data.x_test, self.data.y_test)
            self.prompt_statistics("Test", epoch, *test_metrics)                
            print("---")
            self.epoch_test_loss.append(test_metrics[0])
            self.epoch_test_acc.append(test_metrics[1:])
            
        if val:
            self.epoch_val_loss = [-1]
            self.epoch_val_acc = [-1]
        
        return test_metrics[1]
            
            
    def fit(self, data:DATA, lr, batch_size, epochs, early_stopping=60):
        if not self.is_parameters_initialized:
            self.init_weights()
            self.is_parameters_initialized = True
            
        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        print("Initial metrics...")
        self.eval_all(0)
        
        start = time.perf_counter()
        for epoch in range(1, self.epochs+1):
            batch_no = 0
            
            total_loss = 0
            batch_correct = 0
            
            for xb, yb in mini_batch(self.data.x_train, self.data.y_train, mini_batch_size = self.batch_size):
                m = xb.shape[1]    
                self.m = m
                
                # forward pass
                preds, batch_loss = self.forward_pass(xb, yb)
                
                total_loss += batch_loss
                
                # top 1 acc
                batch_correct = self.get_num_correct(preds, yb)
                b_acc_p = (batch_correct / m)*100
                #print(f"Epoch [{epoch}/{self.epochs}], {batch_no}/{self.data.x_train.shape[1]//m}], Total Loss: {loss:.10f}, B.Acc: {b_acc_p:.4f}%")      
                self.batch_loss.append(batch_loss)
                self.batch_acc.append(b_acc_p)

                ###
                batch_no += 1
                
                #backward pass
                self.bptt(xb, yb)
            
            # epoch evaluation                
            test_top1_acc = self.eval_all(epoch)
            
            if test_top1_acc > self.best_acc:
                self.copy_to_best()
                self.best_acc = test_top1_acc
                self.best_epoch = epoch
                
            # early stopping condition
            if (epoch - self.best_epoch) > early_stopping:
                break
        
        end = time.perf_counter()
        elapsed_time = end - start # in seconds
        
        # Finalize
        cont = RNN.get_shared_container()
        exp_no = RNN.get_exp_no()
        cont[exp_no] = dict()
        
        # Store meta-data types
        cont[exp_no]["seed"] = cur_run_seed
        cont[exp_no]["params"] = dict() 
        cont[exp_no]["results"] = dict()
        cont[exp_no]["quick_results"] = None
         
        # Store Parameters
        cont[exp_no]["params"]["case"] = "case"
        cont[exp_no]["params"]["N"] = self.hidden_size
        cont[exp_no]["params"]["lr"] = self.lr
        cont[exp_no]["params"]["lambda"] = -1
        cont[exp_no]["params"]["epochs"] = self.epochs
        cont[exp_no]["params"]["best_epoch"] = self.best_epoch
        cont[exp_no]["params"]["batch_size"] = self.batch_size         
        cont[exp_no]["params"]["act1"] = self.act1
        cont[exp_no]["params"]["act2"] = self.act2
        cont[exp_no]["params"]["time"] = elapsed_time
        
        
        # Store Meta-data parameters
        cont[exp_no]["results"]["losses"] = list()
        cont[exp_no]["results"]["losses"].append(self.epoch_train_loss.copy())
        cont[exp_no]["results"]["losses"].append(self.epoch_val_loss.copy())
        cont[exp_no]["results"]["losses"].append(self.epoch_test_loss.copy())
        cont[exp_no]["results"]["acc"] = list()
        cont[exp_no]["results"]["acc"].append(self.epoch_train_acc.copy())
        cont[exp_no]["results"]["acc"].append(self.epoch_val_acc.copy())
        cont[exp_no]["results"]["acc"].append(self.epoch_test_acc.copy())
        
        # Store Meta-data parameters
        # this contains the final train, validation, test accuracies
        _accs_test = np.array(self.epoch_test_acc)
        _accs_train = np.array(self.epoch_train_acc)
        _idx = _accs_test[:,0].argmax(axis=0)
        cont[exp_no]["quick_results_top1"] = [_accs_train[_idx, 0], -1, _accs_test[_idx, 0]]
        cont[exp_no]["quick_results_top2"] = [_accs_train[_idx, 1], -1, _accs_test[_idx, 1]]
        cont[exp_no]["quick_results_top3"] = [_accs_train[_idx, 2], -1, _accs_test[_idx, 2]]


def plots(model):
    dir_name = "mini_p2_results/"
    dir_name += "res_" + model.name + "/"
    os.mkdir(dir_name)
    
    pp = plot_outputs(model.a_steps, "a")
    pp.savefig(dir_name + "hidden_a")
    
    pp = plot_weights(model.Waa, "Waa")
    pp.savefig(dir_name + "hidden_weights")

    pp = plot_weights(model.Wax, "Wax")
    pp.savefig(dir_name + "input_weights")

    pp = plot_weights(model.Wya, "Wya")
    pp.savefig(dir_name + "output_weights")
    
    pp = plot_loss(model.epoch_train_loss, model.epoch_test_loss)
    pp.savefig(dir_name + "losses")

    pp = plot_acc(model.epoch_train_acc, model.epoch_test_acc)
    pp.savefig(dir_name + "accs")
    
    model.save_parameters(dir_name)

def confusion_matrix(y_pred, y_true):
    # Convert one-hot encoded arrays to class indices
    y_pred_idxs = np.argmax(y_pred, axis=0)
    y_true_idxs = np.argmax(y_true, axis=0)

    # Calculate confusion matrix
    num_class = y_true.shape[0]
    conf_matrix = np.zeros((num_class, num_class), dtype=np.int64)

    for pred, true_val in zip(y_pred_idxs, y_true_idxs):
        conf_matrix[true_val, pred] += 1

    return conf_matrix

early_stopping = 60
epochs = 50
data = DATA()

# CASE - diff. lr
N = 50
learning_rate = 0.05
batch_size = 10
rnn1 = RNN("rnn1", N, FEATURE_SIZE, OUTPUT_SIZE, NUM_TIME_STEPS)
rnn1.fit(data, lr=learning_rate, batch_size=batch_size, epochs=epochs)
plots(rnn1)
del rnn1
gc.collect()

import json
d = RNN._shared_container
# Store the data in a JSON file
with open('data682.json', 'w') as json_file:
    json.dump(d, json_file)

