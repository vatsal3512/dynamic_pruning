# -*- coding: utf-8 -*-
"""resnet.ipynb

Structured two-threshold pruning of a ResNet-18 (with an extra 64->256 skip) on CIFAR-10 / MNIST:
whole convolution filters are pruned and can regrow, in the style of Dynamic Network Surgery.

Originally developed in Google Colab; runs locally as well.
The original version of this file is kept in original_code/.
"""

import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount('/content/drive')
    basepath = '/content/drive/MyDrive/pruning test'
except ImportError:
    matplotlib.use('Agg')
    basepath = './runs/resnet18'

for sub in ('CheckPoints', 'Losses', 'Accuracies', 'Parameters'):
    os.makedirs(os.path.join(basepath, sub), exist_ok=True)


def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


SEED = 0
seed_everything(SEED)

class EarlyStopping:
    def __init__(self, patience=7, filename='checkpoint_0', verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.filename = filename

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(basepath, 'CheckPoints', self.filename + '.pt'))
        self.val_loss_min = val_loss

# VAL_SIZE = 0 reproduces the original protocol (the test loader is also used for early stopping
# and checkpoint selection).  Set e.g. VAL_SIZE = 5000 to hold out a validation split from the
# training set; the test set is then only evaluated once, at the end.
VAL_SIZE = 0

def get_data_loaders(dataset='CIFAR10', val_size=VAL_SIZE):
  if dataset == 'CIFAR10':
    transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    train_dataset = torchvision.datasets.CIFAR10('../data',transform=transform,train=True,download=True)
    test_dataset = torchvision.datasets.CIFAR10('../data',transform=transform,train=False,download=True)
  elif dataset == 'MNIST':
    transform = transforms.Compose([transforms.Resize((32,32)),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    train_dataset = torchvision.datasets.MNIST('../data',transform=transform,train=True,download=True)
    test_dataset = torchvision.datasets.MNIST('../data',transform=transform,train=False,download=True)

  workers = min(8, os.cpu_count() or 1)
  data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=workers)
  if val_size:
    g = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size], generator=g)
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=workers)
  else:
    data_loader_val = data_loader_test
  data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=workers)
  return(data_loader_train, data_loader_val, data_loader_test)

dataset_choice = 'CIFAR10' # Change to 'MNIST' to validate on MNIST
data_loader_train, data_loader_val, data_loader_test = get_data_loaders(dataset=dataset_choice)

import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)

class FeaturesWithSkip(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # CNN Skip connection: 64 -> 256, downsample spatial and increase channels
        self.skip_64_to_256 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm2d(256)
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.layer1(x)      # 64 channels
        x2 = self.layer2(x1)     # 128 channels
        x3 = self.layer3(x2)     # 256 channels

        skip = self.skip_64_to_256(x1)  # Adjust dimensions to match x3
        x3 = x3 + skip

        x4 = self.layer4(x3)     # 512 channels
        out = self.pool(x4)      # (B, 512, 1, 1)
        return out

class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = FeaturesWithSkip()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)        # (B, 512, 1, 1)
        x = torch.flatten(x, 1)     # (B, 512)
        return self.classifier(x)   # (B, 10)

num_classes = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_unpruned_model():
  model = ResNet18_CIFAR10()
  return(model)

model = get_unpruned_model()

def plot_losses(train_loss,test_loss,filename,i):
  x_axis = list(range(1,len(train_loss)+1))
  fig = plt.figure(i,figsize=(20,10))
  plt.axes(yscale='log')
  plt.plot(x_axis, train_loss, color='blue', label='Train Loss')
  plt.plot(x_axis, test_loss, color='green', label='Validation Loss')
  plt.legend(loc=1)
  plt.xlabel('Training step (batch)')
  plt.ylabel('Loss')
  plt.ylim(0.01,3)
  plt.savefig(os.path.join(basepath, 'Losses', filename + '.png'))
  plt.close(fig)

def plot_accuracies(train_accuracy,test_accuracy,filename,i):
  x_axis = list(range(1,len(train_accuracy)+1))
  fig = plt.figure(i,figsize=(20,10))
  plt.plot(x_axis, train_accuracy, color='blue', label='Train Accuracy')
  plt.plot(x_axis, test_accuracy, color='green', label='Validation Accuracy')
  plt.legend(loc=4)
  plt.xlabel('Training step (batch)')
  plt.ylabel('Accuracy')
  plt.ylim(0.1,1)
  plt.savefig(os.path.join(basepath, 'Accuracies', filename + '.png'))
  plt.close(fig)

def evaluate(model, loader, criterion):
  """Loss (averaged over images) and accuracy of `model` on `loader`."""
  correct_t = 0; data_size_t = 0; tt_loss = 0.0
  model.eval()
  with torch.no_grad():
    for val_images, val_labels in loader:
      val_images = val_images.to(device); val_labels = val_labels.to(device)
      val_outputs = torch.nn.functional.log_softmax(model(val_images), dim=1)
      _, val_predicted = torch.max(val_outputs.data, 1)
      correct_t += (val_predicted == val_labels).sum().item()
      tt_loss += criterion(val_outputs, val_labels).item() * len(val_labels)
      data_size_t += len(val_labels)
  return tt_loss / data_size_t, correct_t / data_size_t

def train(model,prune_itr,lr_multiplier=1):

  scale_factor = 16

  learning_rate = 1e-3*lr_multiplier; wt_dcy =5e-4; lr_patience = int(2000/scale_factor); lr_stepsize = int(3000/scale_factor); es_patience = int(5000/scale_factor)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_dcy)
  scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, threshold=1e-7)
  scheduler_stepLR = StepLR(optimizer, step_size=lr_stepsize, gamma=0.8)
  early_stopping = EarlyStopping(patience=es_patience,filename='checkpoint_'+str(prune_itr), verbose=False)

  if torch.cuda.is_available(): torch.cuda.empty_cache()
  model = model.to(device)
  num_epochs = 50
  es_flag = 0

  train_loss = []; test_loss = []; train_accuracy = []; test_accuracy = []

  best_loss = 100000; best_accuracy = 0

  for epoch in range(num_epochs):

    correct = 0; data_size = 0

    for i, (images, labels) in enumerate(data_loader_train):

      model.train()
      images = images.to(device); labels = labels.to(device)
      outputs = torch.nn.functional.log_softmax(model(images), dim=1)
      trainloss = criterion(outputs,labels)
      optimizer.zero_grad()
      trainloss.backward()
      optimizer.step()
      _, predicted = torch.max(outputs.data, 1)
      batch_correct = (predicted == labels).sum().item()
      correct += batch_correct; data_size += len(images)

      val_loss, val_acc = evaluate(model, data_loader_val, criterion)
      print('Epoch: {}, Batch: {}, Train Loss: {:.3f} Train Accuracy: {:.3f}%, Validation Loss: {:.3f} Validation Accuracy: {:.3f}%'.
            format(epoch+1, i+1, trainloss.item(), 100*batch_correct/len(images), val_loss, 100*val_acc))

      train_loss.append(trainloss.item()); test_loss.append(val_loss)
      train_accuracy.append(batch_correct/len(images)); test_accuracy.append(val_acc)

      if val_loss < best_loss:
        best_loss = val_loss
        best_accuracy = val_acc

      scheduler_stepLR.step()
      scheduler_plateau.step(val_loss)

      early_stopping(val_loss, model)
      if early_stopping.early_stop:
        print("Early stopping")
        es_flag = 1
        break
    print('Epoch {} training accuracy: {:.3f}%'.format(epoch+1, 100*correct/max(data_size,1)))
    if es_flag == 1: break

  model.load_state_dict(torch.load(os.path.join(basepath, 'CheckPoints', 'checkpoint_'+str(prune_itr)+'.pt'), map_location=device))

  return(model,train_loss,test_loss,train_accuracy,test_accuracy,best_loss,best_accuracy)

lr_multiplier = 1
dict_best_info = {}
model,train_loss,test_loss,train_accuracy,test_accuracy,best_loss,best_accuracy = train(model,0,lr_multiplier)
dict_best_info[0] = (best_loss,best_accuracy)
plot_losses(train_loss,test_loss,'l0',1)
plot_accuracies(train_accuracy,test_accuracy,'a0',2)

model = model.to(device)
print(model)

print('Best Accuracy is ' + str(round(best_accuracy*100,2)))
print('Best Loss is ' + str(round(best_loss,3)))

# =====================================================================================
# Pruning engine: two-threshold (hysteresis) masks with weight regrowth
# =====================================================================================
# For every prunable layer we keep
#   dense : the full weight tensor, which keeps training even where the mask is 0
#   mask  : 0/1 tensor; the network always computes with dense * mask
# Thresholds per layer:  a = c_a * max|importance| * k,   b = a + c_t * std(importance)
#   importance <  a              -> pruned
#   a <= importance <  b         -> keeps its previous state (hysteresis band)
#   importance >= b              -> kept, or regrown if it was pruned
# 'unstructured' importance = |w| per weight;  'structured' = mean |w| of each output filter/neuron.

class PrunableLayer:
  def __init__(self, name, module, c_a, c_t, k_growth, granularity='unstructured'):
    self.name, self.module = name, module
    self.c_a, self.c_t, self.k_growth, self.k = c_a, c_t, k_growth, 1.0
    self.granularity = granularity
    self.dense = module.weight.detach().clone()
    self.mask = torch.ones_like(module.weight)

def defineMasks(model, layer_config, granularity='unstructured'):
  """One PrunableLayer per Conv2d/Linear listed in layer_config {module name: (c_a, c_t, k_growth)}."""
  layers = []
  for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)) and name in layer_config:
      layers.append(PrunableLayer(name, module, *layer_config[name], granularity=granularity))
  return layers

def importance(weight, granularity):
  if granularity == 'structured':
    return weight.abs().flatten(1).mean(1)          # one score per output filter / neuron
  return weight.abs()

def expand(unit_mask, weight):
  return unit_mask.view(-1, *([1] * (weight.dim() - 1))).expand_as(weight)

@torch.no_grad()
def update_masks(layers):
  """Two-threshold update computed on the dense weights, so pruned weights can regrow."""
  regrown = 0; pruned = 0
  for L in layers:
    score = importance(L.dense, L.granularity)
    a = L.c_a * score.max() * L.k
    t = L.c_t * (score.std() if L.granularity == 'structured' else L.dense.std())
    b = a + t
    old = L.mask.flatten(1)[:, 0] if L.granularity == 'structured' else L.mask
    new = (((score > a).float() * old) + (score > b).float() >= 1.).float()
    regrown += int(((old == 0) & (new == 1)).sum()); pruned += int(((old == 1) & (new == 0)).sum())
    L.mask = expand(new, L.dense).clone() if L.granularity == 'structured' else new
    L.k = L.k * L.k_growth
  return regrown, pruned

@torch.no_grad()
def model_surgery(layers):
  """Make the network compute with W * M (dense weights are kept in layer.dense)."""
  for L in layers:
    L.module.weight.data.copy_(L.dense * L.mask)

@torch.no_grad()
def restore_dense(layers):
  for L in layers:
    L.module.weight.data.copy_(L.dense)

@torch.no_grad()
def store_dense(layers):
  for L in layers:
    L.dense.copy_(L.module.weight.data)

def count_nonzeros(model):
  return sum(torch.count_nonzero(param).item() for param in model.parameters())

def get_model_layerwise_analysis(layers):
  for L in layers:
    weights = torch.abs(L.module.weight.data)
    print(f'{L.name:<14} max {weights.max():.4f}  mean {weights.mean():.4f}  std {weights.std():.4f}')

initial_parameters = sum([p.numel() for p in model.parameters()])

# Every Conv2d is pruned filter-by-filter (structured); the classifier weight-by-weight.
# layer name: (c_a, c_t, k growth per surgery)
def resnet_layer_config(model):
  cfg = {}
  for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d): cfg[name] = (0.25, 2.0, 1.0)
    elif isinstance(module, nn.Linear): cfg[name] = (0.35, 2.0, 1.0)
  return cfg

def defineResNetMasks(model):
  cfg = resnet_layer_config(model)
  convs = defineMasks(model, {n: c for n, c in cfg.items() if not n.startswith('classifier')}, granularity='structured')
  head = defineMasks(model, {n: c for n, c in cfg.items() if n.startswith('classifier')}, granularity='unstructured')
  return convs + head

def perform_surgery_training(model, prune_itr, lr_multiplier, layers):

  scale_factor = 16

  learning_rate = 1e-3*lr_multiplier; wt_dcy =1e-7; lr_patience = int(2000/scale_factor); lr_stepsize = 1; es_patience = int(10000/scale_factor)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_dcy)
  scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, threshold=1e-7)
  scheduler_stepLR = StepLR(optimizer, step_size=lr_stepsize, gamma=0.995)
  early_stopping = EarlyStopping(patience=es_patience,filename='checkpoint_'+str(prune_itr), verbose=False)

  if torch.cuda.is_available(): torch.cuda.empty_cache()
  model = model.to(device)
  for L in layers:
    L.dense = L.dense.to(device); L.mask = L.mask.to(device)
  num_epochs = 70
  es_flag = 0

  train_loss = []; test_loss = []; train_accuracy = []; test_accuracy = []
  nonzero_parameter_list = []; regrowth_list = []

  best_loss = 100000; best_accuracy = 0; best_parameter_size = 0
  last_loss = last_accuracy = last_parameter_size = None
  final_model = deepcopy(model)

  prob_threshold = 1.0; total_regrown = 0

  for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(data_loader_train):

      model.train()
      images = images.to(device); labels = labels.to(device)
      model_surgery(layers)                       # forward pass uses W * M
      outputs = torch.nn.functional.log_softmax(model(images), dim=1)
      trainloss = criterion(outputs,labels)
      optimizer.zero_grad()
      trainloss.backward()

      restore_dense(layers)                       # gradient of W*M is applied to the dense W
      optimizer.step()
      store_dense(layers)

      choice = random.random()
      if choice <= prob_threshold:
        regrown, pruned = update_masks(layers)
        total_regrown += regrown
        print('Surgery (choice {:.4f} <= p {:.4f}): pruned {}, regrown {}'.format(choice, prob_threshold, pruned, regrown))

      model_surgery(layers)                       # the model now holds W * M for evaluation/checkpoints
      prob_threshold *= 0.9993
      nonzero_parameters = count_nonzeros(model)
      nonzero_parameter_list.append(nonzero_parameters); regrowth_list.append(total_regrown)

      _, predicted = torch.max(outputs.data, 1)
      batch_acc = (predicted == labels).sum().item() / len(images)

      val_loss, val_acc = evaluate(model, data_loader_val, criterion)
      print('Epoch: {}, Batch: {}, Params: {}, Train Loss: {:.3f} Train Accuracy: {:.3f}%, Validation Loss: {:.3f} Validation Accuracy: {:.3f}%'.
            format(epoch+1, i+1, nonzero_parameters, trainloss.item(), 100*batch_acc, val_loss, 100*val_acc))

      train_loss.append(trainloss.item()); test_loss.append(val_loss)
      train_accuracy.append(batch_acc); test_accuracy.append(val_acc)

      if val_loss < best_loss:
        best_loss = val_loss
        best_accuracy = val_acc
        best_parameter_size = nonzero_parameters

      if val_acc >= 0.9 * best_accuracy:          # sparsest model within 10% (relative) of the best accuracy
        last_loss, last_accuracy, last_parameter_size = val_loss, val_acc, nonzero_parameters
        final_model = deepcopy(model)

      scheduler_stepLR.step()
      scheduler_plateau.step(val_loss)

      early_stopping(val_loss, model)
      if early_stopping.early_stop:
        print("Early stopping")
        es_flag = 1
        break
    if es_flag == 1: break

  model.load_state_dict(torch.load(os.path.join(basepath, 'CheckPoints', 'checkpoint_'+str(prune_itr)+'.pt'), map_location=device))
  print('Total weights regrown during training:', total_regrown)

  return(model, final_model, train_loss, test_loss, train_accuracy, test_accuracy, best_loss, best_accuracy, best_parameter_size, last_loss, last_accuracy, last_parameter_size, nonzero_parameter_list, regrowth_list)

def plot_parameters(parameter_list,filename,i):
  x_axis = list(range(1,len(parameter_list)+1))
  fig = plt.figure(i,figsize=(20,10))
  plt.plot(x_axis, parameter_list, color='blue', label='Non-zero Parameters')
  plt.yscale('log')
  plt.legend(loc=1)
  plt.xlabel('Training step (batch)')
  plt.ylabel('Parameters')
  plt.savefig(os.path.join(basepath, 'Parameters', filename + '.png'))
  plt.close(fig)

lr_multiplier = 1
new_model = deepcopy(model)
layers = defineResNetMasks(new_model)
pruned_model, final_model, train_loss, test_loss, train_accuracy, test_accuracy, best_loss, best_accuracy, best_parameter_size, last_loss, last_accuracy, last_parameter_size, parameter_list, regrowth_list = perform_surgery_training(new_model,1,lr_multiplier,layers)
dict_best_info[1] = (best_loss,best_accuracy)
plot_losses(train_loss,test_loss,'l1',1)
plot_accuracies(train_accuracy,test_accuracy,'a1',2)
plot_parameters(parameter_list,'p1',3)

pruned_model = pruned_model.to(device)
print(pruned_model)

get_model_layerwise_analysis(defineResNetMasks(pruned_model))

criterion = nn.CrossEntropyLoss()
remaining_parameters = count_nonzeros(pruned_model)
test_loss_final, test_acc_final = evaluate(pruned_model, data_loader_test, criterion)
dense_test_loss, dense_test_acc = evaluate(model, data_loader_test, criterion)
print('For Compression without loss...')
print('Number of Initial Parameters are ' + str(initial_parameters))
print('Number of Remaining Parameters are ' + str(remaining_parameters))
print('Compression Rate without loss is ' + str(initial_parameters/remaining_parameters))
print('Parameter reduction is {:.2f}%'.format(100*(1 - remaining_parameters/initial_parameters)))
print('Dense test accuracy {:.2f}%  |  pruned test accuracy {:.2f}%'.format(100*dense_test_acc, 100*test_acc_final))
print('Best Loss is ' + str(round(best_loss,3)))

def layer_wise_comparison(model, pruned_model):
  dense_layers = defineResNetMasks(model); sparse_layers = defineResNetMasks(pruned_model)
  for d, s in zip(dense_layers, sparse_layers):
    total = torch.count_nonzero(d.module.weight).item(); left = torch.count_nonzero(s.module.weight).item()
    print('Layer {}: Parameters: {}, Parameters Left: {}, Percentage Left: {}'.format(d.name, total, left, round(left*100/total,2)))

print('For Compression without loss...')
layer_wise_comparison(model,pruned_model)

print('For Best Compression...')
layer_wise_comparison(model,final_model)

if last_parameter_size:
  print('Trained for '+str(last_parameter_size)+' non-zero parameters reaching '+str(initial_parameters/last_parameter_size)+' times compression with loss '+str(round(last_loss,3))+' and accuracy '+str(round(last_accuracy*100,2))+'%.')
