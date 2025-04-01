import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


#####   METRICS   #####
def calc_occurence(data_loader):
    from collections import defaultdict
    class_counts = defaultdict(int)
    for data in data_loader:
        inputs, labels = data
        for label in labels:
            class_counts[label.item()] += 1
    class_counts = dict(sorted(class_counts.items()))
    print(class_counts)

def get_top_n_correct(preds: torch.Tensor, target: torch.Tensor, top_n: int):
    _class, indices = preds.topk(top_n, 1, True, True)
    correct_preds = indices.eq(target.view(-1,1).expand_as(indices))
    total_correct = correct_preds.sum(1).sum(0)
    return _class, total_correct

def calculate_statistics(_model, _criterion, _device, data_loader, debug=0):
    total_loss, total_samples, combined_preds, combined_targets = prediction(_model, _criterion, _device, data_loader)

    _, top_1_acc = get_top_n_correct(combined_preds, combined_targets, top_n=1)
    _, top_2_acc = get_top_n_correct(combined_preds, combined_targets, top_n=2)
    _, top_3_acc = get_top_n_correct(combined_preds, combined_targets, top_n=3)

    if debug:
        print(f"{total_loss=}\n"
              f"{total_samples=}\n"
              f"{top_1_acc=}\n"
              f"{top_2_acc=}\n"
              f"{top_3_acc=}\n"
        )

    return total_loss/total_samples, (top_1_acc*100)/total_samples, (top_2_acc*100)/total_samples, (top_3_acc*100)/total_samples

def prompt_statistics(name:str, cur_epoch:int, total_epochs:int, top1_avg_loss:float,
                      top1_acc:float, top2_acc:float, top3_acc:float):
        print(f"\nEpoch [{cur_epoch}/{total_epochs}], "
              f"{name} \nAvg. (Top1) Loss: {top1_avg_loss:.4f}, "
              f"Total Accuracy -> "
              f"(Top1): {top1_acc:.4f}%, "
              f"(Top2): {top2_acc:.4f}%, "
              f"(Top3): {top3_acc:.4f}%")
        return



###   PREDICTION   ###
def prediction(_model, _criterion, _device, data_loader):
    """
    RETURNS: loss, m:(total_sample_size), preds, labels
    """
    total_loss = 0
    total_samples = 0
    batch_preds = list()
    batch_targets = list()
    _model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(_device)
            labels = labels.to(_device)
            outputs = _model(inputs)
            total_loss += _criterion(outputs, labels)
            predicted = outputs.data
            total_samples += labels.size(0)
            batch_targets.append(labels)
            batch_preds.append(predicted)
    combined_targets = torch.cat(batch_targets, dim=0).view(-1,1)
    combined_preds = torch.cat(batch_preds, dim=0)
    return total_loss, total_samples, combined_preds, combined_targets


###   SAVE/LOAD   PARAMETERS ###
def save_parameters(dir_name, _model, _optimizer):
    torch.save(_model.state_dict(), dir_name+"/wb")
    torch.save(_optimizer.state_dict(), dir_name+"/optims")

def save_best_parameters(dir_name, checkpoint: dict):
    torch.save(checkpoint['model_state_dict'], dir_name+"/wb")
    torch.save(checkpoint['optimizer_state_dict'], dir_name+"/optims")

def load_best_params(dir_name, _model, _optimizer):
    _model.load_state_dict(torch.load(dir_name+"/wb"))
    _optimizer.load_state_dict(torch.load(dir_name+"/optims"))

def get_best_params_checkpoint(_model, _optimizer):
    checkpoint = {
        'model_state_dict': _model.state_dict(),
        'optimizer_state_dict': _optimizer.state_dict()
    }
    return checkpoint


###   CONFUSION MATRIX   ###
def gpu_to_cpu_tensor_to_np(arg1):
    """given a list of tensor it returns lis of numpy arrays"""
    arr = []
    for obj in arg1:
        if isinstance(obj, torch.Tensor):
            arr.append(obj.cpu().detach().numpy())
        else:
            arr.append(obj)
    return arr

def confusion_matrix(y_pred, y_true):
    # Calculate confusion matrix
    num_class = 10
    conf_matrix = np.zeros((num_class, num_class), dtype=np.int64)

    for pred, true_val in zip(y_pred, y_true):
        conf_matrix[true_val, pred] += 1

    return conf_matrix

def pred_confusion_matrix(name: str, preds, targets, verbose=False):
    _, indices = preds.topk(1, 1, True, True)
    conf_matrix = confusion_matrix(gpu_to_cpu_tensor_to_np(indices), gpu_to_cpu_tensor_to_np(targets))

    if verbose:
        print(f"\nConfusion Matrix of {name}:")
        print(conf_matrix)
    return conf_matrix


# watch -n 1 nvidia-smi
