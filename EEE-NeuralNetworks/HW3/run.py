import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # garbage collector
import time
import os

cur_run_seed = 42
np.random.seed(cur_run_seed)


def plot_cm_hm(name, cm):
    _ = plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
                yticklabels=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
    plt.title(f"{name}: Confusion Matrix")
    return _

def plot_loss(name, train_data, test_data, val_data):
    _ = plt.figure(figsize=(12, 8))

    x_arr = [i for i in range(len(train_data))]

    plt.plot(x_arr, train_data, label='Training')
    plt.plot(x_arr, val_data, label='Validation')
    plt.plot(x_arr, test_data, label='Test')

    plt.xlabel("n'th Epoch")
    plt.title(f'{name} : Average Loss over Epochs\n(Total Loss / # of Samples )')
#    plt.xticks(x_arr)

    plt.legend()

    return _

def plot_acc(name, train_data, test_data, val_data):
    _ = plt.figure(figsize=(12, 8))

    train_data = np.array(train_data).T
    val_data = np.array(val_data).T
    test_data = np.array(test_data).T

    x_arr = [i for i in range(len(train_data[0]))]

    # Plotting accuracy in blue
    for i in reversed(range(len(train_data))):
            plt.plot(x_arr, train_data[i], label=f'Top {i+1}, Training')
            if val_data is not None:
                plt.plot(x_arr, val_data[i], label=f'Top {i+1}, Validation')
            plt.plot(x_arr, test_data[i], label=f'Top {i+1}, Test')

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

class DATA:
    def __init__(self):
        # input type (feature, sample)
        # input size 28 * 28 = 784
        self.x_train_orig = None
        self.y_train_orig = None

        self.x_train = np.loadtxt("datas/X_train.txt")
        self.y_train = np.loadtxt("datas/y_train.txt", dtype=np.int64) - 1
        self.x_test = np.loadtxt("datas/X_test.txt")
        self.y_test = np.loadtxt("datas/y_test.txt", dtype=np.int64) - 1
        self.x_val = None
        self.y_val = None

        from sklearn.model_selection import train_test_split
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train, self.y_train,
                test_size=0.1,
                stratify=self.y_train, random_state=cur_run_seed
                )

        self.y_train = self.create_onehot(self.y_train, 6)
        self.y_test = self.create_onehot(self.y_test, 6)
        self.y_val = self.create_onehot(self.y_val, 6)

        self.x_val = self.x_val.T
        self.x_train = self.x_train.T
        self.x_test = self.x_test.T

        #self.calc_occurence(self.y_train)

        gc.collect()
        self.test_data()

    @staticmethod
    def calc_occurence(y):
        count_Y = np.argmax(y, axis=0)
        unique_classes, class_counts = np.unique(count_Y, return_counts=True)
        class_occurrences = dict(zip(unique_classes, class_counts))
        print("Number of occurrences: ", class_occurrences)

    def create_onehot(self, Y, num_class):
        m = len(Y)
        y_onehot = np.zeros((m, num_class), dtype=np.int64)
        for i,label in enumerate(Y, start=0):
            #print(f"{i=}, {label=}\n")
            y_onehot[i][label] = 1
        return y_onehot.T

    def test_data(self):
        assert self.x_train.shape == (561, 6616), self.x_train.shape
        assert self.y_train.shape == (6, 6616), self.y_train.shape
        assert self.x_test.shape == (561, 2947), self.x_test.shape
        assert self.y_test.shape == (6, 2947), self.y_test.shape
        assert self.x_val.shape == (561, 736), self.x_val.shape
        assert self.y_val.shape == (6, 736), self.y_val.shape

def mini_batch(xt, yt, mini_batch_size, shuffle = None):
        """
        It is a generator functions that generates&yields mini batches

        """
        # Shuffle dataset
        def create_permutation(x, y):
            perm = np.random.permutation(len(x))
            return x[perm], y[perm]

        if shuffle is not None:
            x_shuffle, y_shuffle = create_permutation(xt, yt)
        else:
            x_shuffle, y_shuffle = xt[:], yt[:]

        # m: number of examples
        m = yt.shape[1]

        number_of_mini_batches = int(m/mini_batch_size)

        for k in range(number_of_mini_batches):
            mini_batch_X = x_shuffle[:, k*mini_batch_size : (k+1)*mini_batch_size]
            mini_batch_Y = y_shuffle[:, k*mini_batch_size : (k+1)*mini_batch_size]
            yield (mini_batch_X, mini_batch_Y)

        if (m % number_of_mini_batches) != 0:
            mini_batch_X = x_shuffle[:, number_of_mini_batches*mini_batch_size : m]
            mini_batch_Y = y_shuffle[:, number_of_mini_batches*mini_batch_size : m]
            yield (mini_batch_X, mini_batch_Y)

class ReLu:
    @staticmethod
    def forward(z):
        return np.maximum(z, 0)

    @staticmethod
    def derivative(a):
        return np.where(a > 0, 1, 0)

class Softmax:
    @staticmethod
    def forward(z):
        val = np.exp(z-np.max(z, axis=0, keepdims=True))
        return val / np.sum(val, axis=0, keepdims=True)

    @staticmethod
    def derivative(a):
        return a * (1-a)

class _Layer:
    "Bir layer da neler olur ? "
    def __init__(self, weight_arr, bias_arr, activation,
                 beta, dropout):
        """
        Default momentum value is 0. Which mean there is no momentum.
        """
        # output size is the number of neurons in the receiving layer
        # input size is the number of features in the input
        self.W = weight_arr
        self.b = bias_arr

        # activation function
        self.act = activation

        # initialize momentum parameters
        self.beta = beta
        self.v_W = 0
        self.v_b = 0
        if beta > 1e-8:
            self.v_W = np.zeros_like(self.W)
            self.v_b = np.zeros_like(self.b)

        # initialize dropout
        self.dropout = dropout
        self.mask = 1


    def apply_dropout(self, a):
        if abs(1-self.dropout) < 1e-5:
            return a
        else:
            self.mask = (np.random.rand(*a.shape) < self.dropout) / (self.dropout)
            return a * self.mask

    def gradient_update_with_momentum(self, dw, db, lr):
            if self.beta > 1e-8: # not zero condition
                self.v_W = self.beta*self.v_W + (1-self.beta)*dw
                self.v_b = self.beta*self.v_b + (1-self.beta)*db
            else:
                self.W += -1 * lr*dw
                self.b += -1 * lr*db

            self.W += -1 * lr*self.v_W
            self.b += -1 * lr*self.v_b

class Initializer:
    @staticmethod
    def gaussian_uniform(a, b, shape):
        return np.random.uniform(a, b, shape)

    @staticmethod
    def zeros(_, shape):
        return np.zeros(shape)

    @staticmethod
    def ones(_, shape):
        return np.ones(shape)

    @staticmethod
    def he_normal(input_size, shape):
        return np.random.randn(*shape) * np.sqrt(2.0 / input_size)

    @staticmethod
    def he_uniform(input_size, shape):
        limit = np.sqrt(6.0 / input_size)
        print(f"he limit = {limit}")
        return np.random.uniform(-limit, limit, size=shape)

    @staticmethod
    def glorot_uniform(_, shape):
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        print(f"glorot limit = {limit}")
        return np.random.uniform(-limit, limit, size=shape)

    @staticmethod
    def glorot(_, shape):
        return np.random.randn(*shape) * np.sqrt(2.0 / (shape[0] + shape[1]))


class NN:
    _shared_container = dict()
    _exp_no = 0

    @staticmethod
    def get_shared_container():
        return NN._shared_container

    @staticmethod
    def get_exp_no():
        return NN._exp_no

    def update_exp_no(self):
        NN._exp_no += 1

    def __init__(self, name: str):
        self.update_exp_no()

        self.name = name

        self.layers = list()
        self.a = [0] # 0 is a dummy object for 2d input array

        self.lr = None
        self.m = None
        self.batch_size = None
        self.epochs = None

        self.is_parameters_initialized = False

        # training data
        self.epoch_train_loss = []
        self.epoch_train_acc = []

        self.epoch_val_loss = []
        self.epoch_val_acc = []

        self.epoch_test_loss = []
        self.epoch_test_acc = []

        self.dummy_W = None
        self.dummy_b = None

        self.best_acc = -1
        self.best_epoch = -1

    def __str__(self):
        z = ""
        for idx, layer in enumerate(self.layers, start=1):
            z += f"Layer {idx}-> N={layer.W.shape[0]}, Act.={layer.act.__name__}\n"
        return z

    def save_parameters(self, dir_name):
        if self.dummy_W is not None:
            np.savez(f'{dir_name}/W_matrix', *self.dummy_W)
            np.savez(f'{dir_name}/b_matrix', *self.dummy_b)

    def load_best_params(self, dir_name):
        W = np.load(f'{dir_name}W_matrix.npz')
        b = np.load(f'{dir_name}b_matrix.npz')

        for (layer, _W, _b) in zip(self.layers, W, b):
            layer.W = W[_W].copy()
            layer.b = b[_b].copy()

    def copy_to_best(self):
        if not self.dummy_W:
            self.dummy_W = list()
            self.dummy_b = list()
            for layer in self.layers:
                self.dummy_W.append(
                    np.zeros_like(layer.W)
                    )
                self.dummy_b.append(
                    np.zeros_like(layer.b)
                    )

        for l, layer in enumerate(self.layers, start=0):
            self.dummy_W[l] = layer.W.copy()
            self.dummy_b[l] = layer.b.copy()

    def add_layer(self, input_size, layer_size, activation, beta, dropout,
                  weight_initializer, bias_initializer,
                  weight_initializer_params, bias_initializer_params):
        """
        input_size, layer_size
        when initializer is called, shape is (layer_size, input_size)
        """
        # check if the sizes are valid
        if self.layers:
            assert self.layers[-1].W.shape[0] == input_size, f"{self.layers[-1].W.shape[-1]}, {input_size}"

        self.layers.append(
            _Layer(
                weight_initializer(*weight_initializer_params, (layer_size, input_size)),
                bias_initializer(*bias_initializer_params, (layer_size, 1)),
                activation, beta, dropout
                )
        )
        self.a.append(0) # initialize the activation list with dummy object

    def prediction(self, x, y):
        a = x.copy()
        for layer in self.layers:
            z = layer.W @ a + layer.b
            a = layer.act.forward(z)
        total_loss = self.calc_loss(a, y)
        return a, total_loss

    def forward(self, a):
        self.a[0] = a.copy()
        for l, layer in enumerate(self.layers, start=1):
            z = layer.W @ a + layer.b
            a = layer.act.forward(z)
            if layer.dropout != 1:
                a = layer.apply_dropout(a)
            self.a[l] = a.copy()

    def backward(self, y):
        num_layers = len(self.a)

        a_next = None
        dL_dz = self.a[-1] * self.layers[-1].mask - y
        l = num_layers-1

        # apply back prop for every layer
        for layer in reversed(self.layers):
            if l != (num_layers-1):
                # multiply  y 1 if there is no dropout where layer.dropout
                if layer.dropout != 1:
                    a_next = a_next * layer.mask
                dL_dz = a_next * layer.act.derivative(self.a[l])


            dw = (1./self.m) * dL_dz @ self.a[l-1].T
            db = (1./self.m) * np.sum(dL_dz, axis=1, keepdims=True)

            layer.gradient_update_with_momentum(dw, db, self.lr)

            a_next = layer.W.T @ dL_dz
            l -= 1

    def calc_loss(self, logits, y):
        total_loss = -1 * np.sum(y*np.log(logits))
        return total_loss

    def get_num_correct(self, preds, y):
        pred_labels = np.argmax(preds, axis=0)
        y_labels = np.argmax(y, axis=0)
        # Calculate accuracy
        #acc = np.mean(np.equal(pred_labels, y_labels))
        acc = np.sum(np.equal(pred_labels, y_labels))
        return acc

    def get_top_n_correct(self, preds, y, n=1):
        "returns total number of correct predictionAysGM"
        top_n = np.argsort(preds, axis=0)[-n:, :]
        correct_preds = np.sum(top_n == np.argmax(y, axis=0), axis=0)
        total_correct = np.sum(correct_preds)
        return total_correct



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
        #assert np.isclose(_acc, top1_acc) == True

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

        if val:
            val_metrics = self.calculate_statistics(self.data.x_val, self.data.y_val)
            self.prompt_statistics("Validation", epoch, *val_metrics)

            self.epoch_val_loss.append(val_metrics[0])
            self.epoch_val_acc.append(val_metrics[1:])

        if test:
            test_metrics = self.calculate_statistics(self.data.x_test, self.data.y_test)
            self.prompt_statistics("Test", epoch, *test_metrics)

            self.epoch_test_loss.append(test_metrics[0])
            self.epoch_test_acc.append(test_metrics[1:])
            print("---")
        return test_metrics[1]

    def fit(self, data, epochs, lr, batch_size,
            early_stopping):

        self.data = data
        self.epochs = epochs
        self.lr = lr
        if batch_size != -1:
            self.batch_size = batch_size
        else:
            self.batch_size = data.y_train.shape[1]


        print("Initial metrics...")
        DATA.calc_occurence(self.data.y_train)

        _ = self.eval_all(0)
        _logits, _l = self.prediction(self.data.x_train, self.data.y_train)
        print(confusion_matrix(_logits, self.data.y_train))

        start = time.perf_counter()
        for epoch in range(1, epochs+1):

            for xb, yb in mini_batch(self.data.x_train, self.data.y_train, mini_batch_size = self.batch_size):
                # m : number of samples
                m = yb.shape[1]
                self.m = m
                self.forward(xb)
                self.backward(yb)

            test_top1_acc = self.eval_all(epoch)

            if test_top1_acc > (self.best_acc + 1e-5):
                self.copy_to_best()
                self.best_acc = test_top1_acc
                self.best_epoch = epoch

            _loss = 1000
            if epoch == epochs:
                _logits, _loss = self.prediction(self.data.x_train, self.data.y_train)
                print(confusion_matrix(_logits, self.data.y_train))
                _logits, _loss = self.prediction(self.data.x_test, self.data.y_test)
                print(confusion_matrix(_logits, self.data.y_test))

            if _loss < 1e-1:
                break

            # early stopping condition
            if (epoch - self.best_epoch) > early_stopping:
                break

        end = time.perf_counter()
        elapsed_time = end - start # in seconds

        # Finalize
        cont = NN.get_shared_container()
        exp_no = NN.get_exp_no()
        cont[exp_no] = dict()

        # Store meta-data types
        cont[exp_no]["seed"] = cur_run_seed
        cont[exp_no]["params"] = dict()
        cont[exp_no]["results"] = dict()
        cont[exp_no]["quick_results"] = None

        # Store Parameters
        cont[exp_no]["params"]["case"] = "case"
        cont[exp_no]["params"]["N"] = [layer.b.shape[0] for layer in self.layers]
        cont[exp_no]["params"]["lr"] = self.lr

        cont[exp_no]["params"]["beta"] = self.layers[0].beta
        cont[exp_no]["params"]["dropout"] = self.layers[0].dropout

        cont[exp_no]["params"]["epochs"] = self.epochs
        cont[exp_no]["params"]["best_epoch"] = self.best_epoch
        cont[exp_no]["params"]["batch_size"] = self.batch_size
        cont[exp_no]["params"]["act1"] = "ReLu"
        cont[exp_no]["params"]["act2"] = "Softmax"
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

        # Store Meta-data parameterKsMsGp
        # this contains the final train, validation, test accuracies
        _accs_test = np.array(self.epoch_test_acc.copy())
        _accs_val = np.array(self.epoch_val_acc.copy())
        _accs_train = np.array(self.epoch_train_acc.copy())
        _idx = _accs_test[:,0].argmax(axis=0)
        cont[exp_no]["quick_results_top1"] = [_accs_train[_idx, 0], _accs_val[_idx, 0], _accs_test[_idx, 0]]
        cont[exp_no]["quick_results_top2"] = [_accs_train[_idx, 1], _accs_val[_idx, 1], _accs_test[_idx, 1]]
        cont[exp_no]["quick_results_top3"] = [_accs_train[_idx, 2], _accs_val[_idx, 2], _accs_test[_idx, 2]]


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

def pred_confusion_matrix(name: str, model: NN, x, y, verbose=False):
    preds, loss = model.prediction(x, y)
    conf_matrix = confusion_matrix(preds, y)
    if verbose:
        print(f"\nConfusion Matrix of {name}:")
        print(conf_matrix)
    return conf_matrix


def init_hw_nn_assgn(name: str, input_size, N1, N2, output_size, beta, dropout):
    _nn = NN(name)

    # hidden layer 1
    _nn.add_layer(input_size, N1, ReLu, beta, dropout,
                 Initializer.gaussian_uniform, Initializer.gaussian_uniform,
                 (-0.1, 0.1), (-0.1, 0.1)
                 )

    # hidden layer 2
    _nn.add_layer(N1, N2, ReLu, beta, dropout,
                 Initializer.gaussian_uniform, Initializer.gaussian_uniform,
                 (-0.1, 0.1), (-0.1, 0.1)
                 )

    # output layer
    _nn.add_layer(N2, output_size, Softmax, beta, dropout,
                 Initializer.gaussian_uniform, Initializer.gaussian_uniform,
                 (-0.1, 0.1), (-0.1, 0.1)
                 )

    return _nn

def init_hw_nn_he(name, input_size, N1, N2, output_size, beta, dropout):
    _nn = NN(name)

    # hidden layer 1
    _nn.add_layer(input_size, N1, ReLu, beta, dropout,
                 Initializer.he_uniform, Initializer.ones,
                 (input_size,), (input_size,)
                 )

    # hidden layer 2
    _nn.add_layer(N1, N2, ReLu, beta, dropout,
                 Initializer.he_uniform, Initializer.ones,
                 (N1,), (N1,)
                 )

    # output layer
    _nn.add_layer(N2, output_size, Softmax, beta, dropout,
                 Initializer.glorot_uniform, Initializer.ones,
                 (None,), (None,)
                 )
    return _nn

def find_good_seed(data, input_Size, model, rangee):
    losss = 0
    best_se = -1
    diff_max = float('inf')
    l = range(*rangee)
    from tqdm import tqdm
    for i in tqdm(l, desc="Processing"):
        if diff_max <= 1e-1:
            break
        np.random.seed(i)
        #print("\n----\nSoTa Init:")
        losss = model.fit(data.x_train,
                  data.y_train, epochs=0, lr=0)
        diff = abs(-np.log(1/data.y_train.shape[0]) - losss)
        if diff <= diff_max:
            best_se = i
            diff_max = diff
    return losss, best_se

def print_Res(model):
    #pred_confusion_matrix("Test Results", model, data.x_test, data.y_test)
    dir_name = "mini_p3_results/"
    dir_name += "res_" + model.name + "/"
    os.mkdir(dir_name)

    idx = model.name[2:]
    pp = plot_weights(model.layers[-1].W, "W_last")
    pp.savefig(dir_name + "W_last")

    pp = plot_weights(model.layers[-2].W, "W_h2")
    pp.savefig(dir_name + "W_h2")

    pp = plot_weights(model.layers[-3].W, "W_h1")
    pp.savefig(dir_name + "W_h3")

    pp = plot_loss(model.name, model.epoch_train_loss, model.epoch_test_loss, model.epoch_val_loss)
    pp.savefig(dir_name + f"losses{idx}")

    pp = plot_acc(model.name, model.epoch_train_acc, model.epoch_test_acc, model.epoch_val_acc)
    pp.savefig(dir_name + f"accs{idx}")

    model.save_parameters(dir_name)

    # confusion matrix for the best result
    model.load_best_params(dir_name)

    cm = pred_confusion_matrix("", model, model.data.x_train, model.data.y_train)
    pp = plot_cm_hm(model.name + ", Training", cm)
    pp.savefig(dir_name + f"trainingConfMatrix{idx}")

    cm = pred_confusion_matrix("", model, model.data.x_test, model.data.y_test)
    pp = plot_cm_hm(model.name + ", Test", cm)
    pp.savefig(dir_name + f"testConfMatrix{idx}")

def run_sims():

    data = DATA()
    input_Size = 561
    outputs_Size = 6
    epochs = 2
    _early_stopping = 20
    _N1 = 300

    ################# HYPERPARAMETERS #################
    np.random.seed(cur_run_seed)
    _N2 = 100
    _beta = 0
    _dp = 1
    _batch_size = 50
    _lr = 0.001
    nn1 = init_hw_nn_assgn("nn1", input_Size, _N1, _N2, outputs_Size, _beta, _dp)
    nn1.fit(data, epochs, lr=_lr, batch_size=_batch_size, early_stopping=_early_stopping)
    print_Res(nn1)
    del nn1
    gc.collect()

    import json
    d = NN._shared_container
    # Store the data in a JSON file
    with open('data42.json', 'w') as json_file:
        json.dump(d, json_file)


if __name__ == '__main__':
    run_sims()


