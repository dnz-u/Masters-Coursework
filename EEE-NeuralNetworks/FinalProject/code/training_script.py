import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import CIFAR10

from torchvision.models import resnet18

from torchvision.transforms import InterpolationMode # type: enum.EnumMeta
from PIL import Image
from tqdm import tqdm

import time
import random
import numpy as np

import utils
import plots

import cmodel
import json

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

cur_run_seed = 42

class CustomLrSheduler():
    def __init__(self, optimizer, mode, beta=0.3, patience=20, cool_down=5, improvement_threshold_coef=0.0001):
        self.optim = optimizer
        self.mode = mode
        self.patience = patience
        self.cool_down = cool_down
        self.improvement_threshold_coef = improvement_threshold_coef
        self.beta = beta
        self.best_val = None

        self.patience_counter = None
        self.cool_down_counter = None

        self.init_vals()

    def reset_counters(self):
        self.patience_counter = self.patience
        self.cool_down_counter = self.cool_down

    def init_vals(self):
        if self.mode == "max":
            self.best_val = -1
        if self.mode == "min":
            self.best_val = float("inf")

        self.reset_counters()

    def get_improvement_val(self, val):
        return val * (1+self.improvement_threshold_coef)

    def lr_update(self, cur_val, verbose=True):
        imp_val = self.get_improvement_val(cur_val)

        if self.mode == "max":
            if imp_val < self.best_val:
                if self.cool_down_counter < 1:
                    self.patience_counter -= 1

        if self.mode == "min":
            if imp_val > self.best_val:
                if self.cool_down_counter < 1:
                    self.patience_counter -= 1

        if self.patience_counter < 1:
            for g in self.optim.param_groups:
                lr_val = g['lr']
                if lr_val > 0.00009:
                    g['lr'] = lr_val * self.beta
                    if verbose:
                        print("! Learning Rate Updated. Now: ", g['lr'], "\n")

            self.reset_counters()

        self.cool_down_counter -= 1
        self.best_val = max(self.best_val, cur_val)

    def _invoke_update(self, cur_val, verbose=True):
        imp_val = self.get_improvement_val(cur_val)

        if self.mode == "max":
            if imp_val < self.best_val:
                if self.cool_down_counter < 1:
                    self.patience_counter -= 1

        if self.mode == "min":
            if imp_val > self.best_val:
                if self.cool_down_counter < 1:
                    self.patience_counter -= 1

        if True:
            for g in self.optim.param_groups:
                g['lr'] = g['lr'] * self.beta
                if verbose:
                    print("! Learning Rate Updated. Now: ", g['lr'], "\n")

            self.reset_counters()

        self.cool_down_counter -= 1
        self.best_val = max(self.best_val, cur_val)


    def help(self):
        s ="""
        Improvement threshold works as follows:
            bestVal > bestVal * (1 + improvement_threshold_coef)
            For Loss: // min case
                if loss bestVal is 0.094, minimum improvement should be less than
                0.094 * (1+0.0001) = 0.940094

            For Acc:  // max case
                if acc bestVal is 94.87%, minimum improvement should be bigger than
                94.870 * (1+0.0001) = 94.879487
        """
        print(s, "\n")

class CosineAnnealingLRWithWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, warmup_factor=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        super(CosineAnnealingLRWithWarmup, self).__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.warmup_factor + (1.0 - self.warmup_factor) * self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps))) / 2 for base_lr in self.base_lrs]

class DNN:
    _shared_container = dict()
    _exp_no = 0

    @staticmethod
    def get_shared_container():
        return DNN._shared_container

    @staticmethod
    def get_exp_no():
        return DNN._exp_no

    def update_exp_no(self):
        DNN._exp_no += 1

    def __init__(self, name: str, nn_model, nn_optim, nn_crit, device):
        self.update_exp_no()

        self.name: str = name
        self.model = nn_model
        self.optimizer = nn_optim
        self.criterion = nn_crit
        self.device = device

        self.lr = None
        self.batch_size = None
        self.epochs = None
        self.l2_lambda = None
        self.momentum = None
        self.opt = None
        self.keepdims = None

        # training, test datas
        self.epoch_train_loss = []
        self.epoch_train_acc = []

        self.epoch_val_loss = []
        self.epoch_val_acc = []

        self.epoch_test_loss = []
        self.epoch_test_acc = []

        # hold model parameters and optimizer parameters
        self.checkpoint = None

        self.best_acc = -1
        self.best_epoch = -1

    def eval_all(self, epoch, train_loader, val_loader, test_loader, train=True, val=True, test=True):
        def last_conversion(data_metrics):
            tm = []
            for val in data_metrics:
                tm.append(val.item())
                if isinstance(val, list):
                    tmp = []
                    for val_inner in val:
                        tmp.append(val_inner.item())
                    tm.append(tmp.copy())
            return tm

        self.model.eval()
        if train:
            training_metrics = utils.calculate_statistics(self.model, self.criterion, self.device, train_loader)
            training_metrics = utils.gpu_to_cpu_tensor_to_np(training_metrics)
            training_metrics = last_conversion(training_metrics)
            if (epoch%3) == 0:
                utils.prompt_statistics("Training", epoch, self.epochs, *training_metrics)

            self.epoch_train_loss.append(training_metrics[0])
            self.epoch_train_acc.append(training_metrics[1:])

        if val:
            val_metrics = utils.calculate_statistics(self.model, self.criterion, self.device, val_loader)
            val_metrics = list(utils.gpu_to_cpu_tensor_to_np(val_metrics))
            val_metrics = last_conversion(val_metrics)
            if (epoch%3) == 0:
                utils.prompt_statistics("Validation", epoch, self.epochs, *val_metrics)
            self.epoch_val_loss.append(val_metrics[0])
            self.epoch_val_acc.append(val_metrics[1:])
            #print("---")

        if test:
            test_metrics = utils.calculate_statistics(self.model, self.criterion, self.device, test_loader)
            test_metrics = list(utils.gpu_to_cpu_tensor_to_np(test_metrics))
            test_metrics = last_conversion(test_metrics)
            if (epoch%3) == 0:
                utils.prompt_statistics("Test", epoch, self.epochs, *test_metrics)
            self.epoch_test_loss.append(test_metrics[0])
            self.epoch_test_acc.append(test_metrics[1:])
            #print("---")

        return val_metrics[1]


    def fit(self, case:str, train_gen, val_gen, test_gen, epochs, lr, batch_size, early_stopping,
            momentum, l2_lambda, opt, keepdims):

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.opt = opt
        self.keepdims = keepdims

        #print("Initial metrics...")
        _ = self.eval_all(0, train_gen, val_gen, test_gen)

        # Train the model on the modified CIFAR-10 dataset

        #lr_scheduler = CosineAnnealingLRWithWarmup(self.optimizer, T_max=self.epochs, warmup_steps=5, eta_min=0.0001*self.lr)
        #lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', cooldown=3)

        # # Set milestones for learning rate decay in terms of batches
        # milestones = [32000, 48000]
        # milestones_in_batches = []
        # train_size =  45000
        # for mile in milestones:
        #      milestones_in_batches.append(int((mile*batch_size)/train_size))
        # print(f"{milestones_in_batches=}")

        #lr_sc = CustomLrSheduler(self.optimizer, "max", beta=1)
        #lr_sc._invoke_update(-5)

        start = time.perf_counter()
        for epoch in range(1, self.epochs+1):
            self.model.train()
            for inputs, labels in train_gen:
                self.optimizer.zero_grad()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Evaluate the model on the training and test set
            self.model.eval()
            val_top1_acc = self.eval_all(epoch, train_gen, val_gen, test_gen)

            if val_top1_acc > (self.best_acc + 1e-5):
                self.checkpoint = utils.get_best_params_checkpoint(self.model, self.optimizer)
                self.best_acc = val_top1_acc
                self.best_epoch = epoch

            _loss = 1000

            if _loss < 1e-7:
                break

            # early stopping condition
            if (epoch - self.best_epoch) > early_stopping:
                break


            # if epoch == 70:
            #     for g in self.optimizer.param_groups:
            #         g['lr'] = g['lr'] * 0.1
            #         print(g['lr'])

            #lr_scheduler.step(self.epoch_val_acc[-1])
            #lr_sc.lr_update(val_top1_acc, True) # [Top1 Top2 Top3][0]

        end = time.perf_counter()
        elapsed_time = end - start # in seconds

        self.record(case, elapsed_time)

    def record(self, case: str, elapsed_time: float):
        global cur_run_seed

        # Finalize
        cont = DNN.get_shared_container()
        exp_no = DNN.get_exp_no()
        cont[exp_no] = dict()

        # Store meta-data types
        cont[exp_no]["seed"] = cur_run_seed
        cont[exp_no]["params"] = dict()
        cont[exp_no]["results"] = dict()
        cont[exp_no]["quick_results"] = None

        # Store Parameters
        cont[exp_no]["params"]["case"] = "case"
        cont[exp_no]["params"]["keepdims"] = self.keepdims
        cont[exp_no]["params"]["lr"] = self.lr

        cont[exp_no]["params"]["beta"] = self.momentum
        cont[exp_no]["params"]["l2"] = self.l2_lambda
        cont[exp_no]["params"]["opt"] = self.opt


        cont[exp_no]["params"]["epochs"] = self.epochs
        cont[exp_no]["params"]["best_epoch"] = self.best_epoch
        cont[exp_no]["params"]["batch_size"] = self.batch_size
        cont[exp_no]["params"]["act1"] = "-1"
        cont[exp_no]["params"]["act2"] = "-1"
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
        _accs_test = np.array(self.epoch_test_acc.copy())
        _accs_val = np.array(self.epoch_val_acc.copy())
        _accs_train = np.array(self.epoch_train_acc.copy())
        _idx = _accs_test[:,0].argmax(axis=0)
        cont[exp_no]["quick_results_top1"] = [_accs_train[_idx, 0], _accs_val[_idx, 0], _accs_test[_idx, 0]]
        cont[exp_no]["quick_results_top2"] = [_accs_train[_idx, 1], _accs_val[_idx, 1], _accs_test[_idx, 1]]
        cont[exp_no]["quick_results_top3"] = [_accs_train[_idx, 2], _accs_val[_idx, 2], _accs_test[_idx, 2]]


def print_Res(inst, model, opt, crit, train_loader, test_loader):
    #pred_confusion_matrix("Test Results", model, data.x_test, data.y_test)
    dir_name = "mini_p3_results/"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    dir_name += "res_" + inst.name + "/"
    os.mkdir(dir_name)
    idx = inst.name[2:]

    last_layer_weights = utils.gpu_to_cpu_tensor_to_np(list(model.parameters())[-2])
    # print(f"{len(last_layer_weights)=}\n"
    #       f"{last_layer_weights=}"
    #       f"{last_layer_weights[0].shape=}")
    pp = plots.plot_weights(np.array(last_layer_weights).reshape(10, -1), "W_last")
    pp.savefig(dir_name + "W_last")
    plots.close_p(pp)

    pp = plots.plot_loss(inst.name, inst.epoch_train_loss, inst.epoch_val_loss, "Validation")
    pp.savefig(dir_name + f"losses{idx}")
    plots.close_p(pp)

    pp = plots.plot_acc(inst.name, inst.epoch_train_acc, inst.epoch_val_acc, "Validation")
    pp.savefig(dir_name + f"accs{idx}")
    plots.close_p(pp)

    #####################################################################
    # save best test accuracy parameters
    utils.save_best_parameters(dir_name, inst.checkpoint)

    # confusion matrix for the best result
    utils.load_best_params(dir_name, model, opt)

    _, _, preds, targets = utils.prediction(model, crit, inst.device, train_loader)
    cm = utils.pred_confusion_matrix("", preds, targets)
    pp = plots.plot_cm_hm(inst.name + ", Training", cm)
    pp.savefig(dir_name + f"trainingConfMatrix{idx}")
    plots.close_p(pp)

    _, _, preds, targets = utils.prediction(model, crit, inst.device, test_loader)
    cm = utils.pred_confusion_matrix("", preds, targets)
    pp = plots.plot_cm_hm(inst.name + ", Validation", cm)
    pp.savefig(dir_name + f"validationConfMatrix{idx}")
    plots.close_p(pp)

def set_seeds(seed=42):
    """
    This code copied from internet
    # Call this function before initializing your models and training loops
    """

    # Set seed for Python random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA operations if available
    if torch.cuda.is_available():
        #print("\nCUDA IS AVAILABLE.", end=" ")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random number generator to deterministic mode for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # If set to True, it may improve training speed for some configurations, but may not be reproducible
    print("Seeds are set.\n")

def transform_w_interpolate(mod: str):
    """
    mods: nearest, bilinear, bicubic, box, hamming, and lanczos

    """
    enum_meta = InterpolationMode
    d = {"bilinear":enum_meta.BILINEAR,
         "nearest":enum_meta.NEAREST,
         "bicubic":enum_meta.BICUBIC,
         "box":enum_meta.BOX,
         "hamming":enum_meta.HAMMING,
         "lanczos":enum_meta.LANCZOS
        }

    import norms
    ns = norms.d_norms[mod]

    if mod not in d:
        raise ValueError

    t = transforms.Compose([
            transforms.Resize((256, 256), interpolation=d[mod]),
            transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(ns['mu'], ns['std'])
            ])
    return t

def cifar32_transform():
    import norms
    n32 = norms.d_norms_32['32']
    t = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(n32['mu'], n32['std'])
            ])
    return t

def get_model():
    # Load pre-trained ResNet model
    resnet_model = resnet18(pretrained=True)

    # Modify the size of the output layer for CIFAR-10
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
    _ = resnet_model
    return resnet_model

def get_cmodel(*, dp_val):
    m = cmodel.myResnetStyleModel(keepdims=dp_val)
    m.initialize_weights()
    return m

def _gen_cond():
    batch_sizes = [32, 64, 128, 256]
    lr_values = [0.1, 0.01, 0.001, 0.0001]
    l2_reg = [0.001, 0.0001]
    dropout_vals = [0.0] #, 0.5, 0.8]
    d_optim = {"adamw": optim.AdamW}
    for opt_name_str, opt_foo in d_optim.items():
        for batch_size in batch_sizes:
            for lr in lr_values:
                for l2lambda in l2_reg:
                    for dp_rate in dropout_vals:
                        if False:
                            print(f"{batch_size=}\n"
                                  f"{momentum=}\n"
                                  f"{opt_foo=}\n"
                                  f"{lr=}\n"
                                  f"{l2lambda=}\n"
                                  f"{dp_rate=}\n")

                        yield {"BATCH_SIZE": batch_size,
                               "MOMENTUM"  : 0.9,
                               "OPT_FOO"   : (opt_name_str, opt_foo),
                               "LR"        : lr,
                               "L2LAMBDA"  : l2lambda,
                               "KEEP_DIMS" : dp_rate}


def gen_cond():
    batch_sizes = [64]
    lr_values = [0.1, 0.01]
    dropout_vals = [0.1, 0.2, 0.5, 0.7] #, 0.5, 0.8]
    d_optim = {"adamw": optim.AdamW}
    for opt_name_str, opt_foo in d_optim.items():
        for batch_size in batch_sizes:
            for lr in lr_values:
                for dp_rate in dropout_vals:
                    if False:
                        print(f"{batch_size=}\n"
                              f"{momentum=}\n"
                              f"{opt_foo=}\n"
                              f"{lr=}\n"
                              f"{l2lambda=}\n"
                              f"{dp_rate=}\n")

                    yield {"BATCH_SIZE": batch_size,
                           "MOMENTUM"  : 0.9,
                           "OPT_FOO"   : (opt_name_str, opt_foo),
                           "LR"        : lr,
                           "L2LAMBDA"  : lr*0.01,
                           "KEEP_DIMS" : dp_rate}


def resnet_gen_cond():
    batch_sizes = [128]
    lr_values = [0.1]
    dropout_vals = [0.0]
    d_optim = {"sgd": optim.SGD}
    for opt_name_str, opt_foo in d_optim.items():
        for batch_size in batch_sizes:
            for lr in lr_values:
                for dp_rate in dropout_vals:
                    if False:
                        print(f"{batch_size=}\n"
                              f"{momentum=}\n"
                              f"{opt_foo=}\n"
                              f"{lr=}\n"
                              f"{l2lambda=}\n"
                              f"{dp_rate=}\n")

                    yield {"BATCH_SIZE": batch_size,
                           "MOMENTUM"  : 0.9,
                           "OPT_FOO"   : (opt_name_str, opt_foo),
                           "LR"        : lr,
                           "L2LAMBDA"  : 0.0001,
                           "KEEP_DIMS" : dp_rate}


def run_sims(device, train_dataset, val_dataset, test_dataset, models:dict):

    NUM_EPOCHS = 200
    EARLY_STOPPING = 50
    DEBUG = 0

    total_runs = len(list(gen_cond()))
    c = 0
    for get_inputs in tqdm(resnet_gen_cond(), total=total_runs, desc='Run No'):
        c += 1
        set_seeds()

        BATCH_SIZE = get_inputs["BATCH_SIZE"]
        MOMENTUM   = get_inputs["MOMENTUM"]
        OPT_NAME   = get_inputs["OPT_FOO"][0]
        OPT_FOO    = get_inputs["OPT_FOO"][1]
        LR         = get_inputs["LR"]
        L2LAMBDA   = get_inputs["L2LAMBDA"]
        KEEP_DIMS  = get_inputs["KEEP_DIMS"]

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = get_model().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = None
        if OPT_NAME == "sgd":
            optimizer = OPT_FOO(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=L2LAMBDA)
        else:
            optimizer = OPT_FOO(model.parameters(), lr=LR, weight_decay=L2LAMBDA)

        nn_instance = DNN(f"nn{c}", model, optimizer, criterion, device)
        models[c] = nn_instance

        nn_instance.fit(f"nn{c}", train_loader, val_loader, test_loader, epochs=NUM_EPOCHS, lr=LR, batch_size=BATCH_SIZE, early_stopping=EARLY_STOPPING,
                        momentum=MOMENTUM, l2_lambda=L2LAMBDA, opt=OPT_NAME, keepdims=KEEP_DIMS)
        print_Res(nn_instance, model, optimizer, criterion, train_loader, test_loader)

        with open(f'./json_outputs/data42-{c}.json','w') as file_handle_final:
            entry = {
                str(c): {
                    'prm': get_inputs,
                    'res': DNN._shared_container[c]["results"],
                    'quick_res_t1': DNN._shared_container[c]["quick_results_top1"],
                    'quick_res_t2': DNN._shared_container[c]["quick_results_top2"],
                    'quick_res_t3': DNN._shared_container[c]["quick_results_top3"]
                        }
                    }
            entry[str(c)]['prm']['OPT_FOO'] = entry[str(c)]['prm']['OPT_FOO'][0]
            json.dump(entry, file_handle_final)


def dataset_split(full_dataset, bigger_chunk_size_coef):
    full_dataset_list = list(range(len(full_dataset)))
    labels = [label for _, label in full_dataset]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=bigger_chunk_size_coef, random_state=42)
    val_idx, train_idx = next(sss.split(full_dataset_list, labels))
    # Create Subset datasets using the selected indices
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    print(f"LENGTH {len(val_idx)=}, {len(train_idx)=}")
    return val_dataset, train_dataset

if __name__ == '__main__':
    # check and set cpu or gpu
    device_state = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_state)
    print("The device is ...", device_state.upper(), "...\n\n")


    transform = transform_w_interpolate("bilinear")
    #transform = cifar32_transform()

    # Download and Load CIFAR-10
    tv_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_dataset, train_dataset = dataset_split(tv_dataset, bigger_chunk_size_coef=0.9)

    models = dict()
    run_sims(device, train_dataset, val_dataset, test_dataset, models)

    d = DNN._shared_container
    # Store the data in a JSON file
    with open('data42.json', 'w') as json_file:
        json.dump(d, json_file)

    print("\nFinished.\n")


