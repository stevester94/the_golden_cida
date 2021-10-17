#! /usr/bin/env python3
import os
from steves_models.configurable_cida import Configurable_CIDA
from steves_utils.cida_train_eval_test_jig import  CIDA_Train_Eval_Test_Jig
from steves_utils.dummy_cida_dataset import Dummy_CIDA_Dataset
from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.lazy_map import Lazy_Map
from steves_utils.sequence_aggregator import Sequence_Aggregator
import torch
import numpy as np
import os
import sys
import json
import time
from math import floor
import matplotlib.pyplot as plt
import matplotlib.gridspec



# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
NUM_LOGS_PER_EPOCH = 5


###################################
# Parse Args, Set paramaters
###################################


if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    fake_args = {}
    fake_args["experiment_name"] = "Manual Experiment"
    fake_args["lr"] = 0.001
    fake_args["n_epoch"] = 3
    fake_args["batch_size"] = 1280
    fake_args["patience"] = 10
    fake_args["seed"] = 1337
    fake_args["device"] = "cuda"

    fake_args["source_snrs"] = [0, 2, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
    fake_args["target_snrs"] = [4]

    fake_args["x_net"] = [
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},
    ]
    fake_args["u_net"] = [
        {"class": "Identity", "kargs": {}},
    ]
    fake_args["merge_net"] = [
        {"class": "Linear", "kargs": {"in_features": 50*58+1, "out_features": 256}},
    ]
    fake_args["class_net"] = [
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
    ]
    fake_args["domain_net"] = [
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
        {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
    ]

    fake_args["device"] = "cuda"

    fake_args["alpha"] = "sigmoid"

    parameters = fake_args


experiment_name = parameters["experiment_name"]
lr              = parameters["lr"]
n_epoch         = parameters["n_epoch"]
batch_size      = parameters["batch_size"]
patience        = parameters["patience"]
seed            = parameters["seed"]
device          = parameters["device"]
alpha           = parameters["alpha"]
source_snrs     = parameters["source_snrs"]
target_snrs     = parameters["target_snrs"]

start_time_secs = time.time()

###################################
# Clear out results if it already exists
###################################
os.system("rm -rf "+RESULTS_DIR)
os.mkdir(RESULTS_DIR)


###################################
# Set the RNGs and make it all deterministic
###################################
import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True) 


###################################
# Build the network(s)
# Note: It's critical to do this AFTER setting the RNG
###################################
x_net           = build_sequential(parameters["x_net"])
u_net           = build_sequential(parameters["u_net"])
merge_net       = build_sequential(parameters["merge_net"])
class_net       = build_sequential(parameters["class_net"])
domain_net      = build_sequential(parameters["domain_net"])

###################################
# Build the dataset
###################################

# We append a 1 or 0 to the source and target ds respectively
# This gives us a final tuple of
# (Time domain IQ, label, domain, source?<this is effectively a bool>)
source_ds = Dummy_CIDA_Dataset(
    normalize_domain=True,
    num_classes=16,
    num_unique_examples_per_class=250,
    domains=source_snrs,
    x_shape=(2,128)
)

source_ds = Lazy_Map(
    source_ds, lambda i: i + (1,)
)

target_ds = Dummy_CIDA_Dataset(
    normalize_domain=True,
    num_classes=16,
    num_unique_examples_per_class=250,
    domains=target_snrs,
    x_shape=(2,128)
)
target_ds = Lazy_Map(
    target_ds, lambda i: i + (0,)
)

def wrap_in_dataloader(ds):
    return torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)


# Split our source and target datasets, wrap them in dataloaders. BUT NOT TRAIN
source_train_len = floor(len(source_ds)*0.7)
source_val_len   = floor(len(source_ds)*0.15)
source_test_len  = len(source_ds) - source_train_len - source_val_len
source_train, source_val, source_test = torch.utils.data.random_split(source_ds, [source_train_len, source_val_len, source_test_len], generator=torch.Generator().manual_seed(seed))
source_val, source_test = (
    wrap_in_dataloader(source_val), wrap_in_dataloader(source_test)
)

target_train_len = floor(len(target_ds)*0.7)
target_val_len   = floor(len(target_ds)*0.15)
target_test_len  = len(target_ds) - target_train_len - target_val_len
target_train, target_val, target_test = torch.utils.data.random_split(target_ds, [target_train_len, target_val_len, target_test_len], generator=torch.Generator().manual_seed(seed))
target_val, target_test = (
    wrap_in_dataloader(target_val), wrap_in_dataloader(target_test)
)

# HERE'S the clincher!
# We combine our source and target train set. This lets us use unbalanced datasets (IE if we have more source than target)
train = Sequence_Aggregator([target_train, source_train])
train = wrap_in_dataloader(train)


if alpha == "sigmoid":
    def sigmoid(epoch, total_epochs):
        # This is the same as DANN except we ignore batch
        x = epoch/total_epochs
        gamma = 10
        alpha = 2. / (1. + np.exp(-gamma * x)) - 1

        return alpha


    alpha_func = sigmoid
elif alpha == "null":
    alpha_func = lambda e,n: 0 # No alpha
else:
    raise Exception("Unknown alpha requested: " + str(alpha))
###################################
# Build the model
###################################
model = Configurable_CIDA(
    x_net=x_net,
    u_net=u_net,
    merge_net=merge_net,
    class_net=class_net,
    domain_net=domain_net,
    label_loss_object=torch.nn.NLLLoss(),
    domain_loss_object=torch.nn.L1Loss(),
    learning_rate=lr
)


###################################
# Build the tet jig, train
###################################
cida_tet_jig = CIDA_Train_Eval_Test_Jig(
    model=model,
    path_to_best_model=BEST_MODEL_PATH,
    device=torch.device(device),
    label_loss_object=torch.nn.NLLLoss(),
    domain_loss_object=torch.nn.L1Loss(),
)

cida_tet_jig.train(
    train_iterable=train,
    source_val_iterable=source_val,
    target_val_iterable=target_val,
    patience=patience,
    learning_rate=lr,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
    alpha_func=alpha_func
)


###################################
# Colate experiment results
###################################
source_test_label_accuracy, source_test_label_loss, source_test_domain_loss = cida_tet_jig.test(source_test)
target_test_label_accuracy, target_test_label_loss, target_test_domain_loss = cida_tet_jig.test(target_test)
source_val_label_accuracy, source_val_label_loss, source_val_domain_loss = cida_tet_jig.test(source_val)
target_val_label_accuracy, target_val_label_loss, target_val_domain_loss = cida_tet_jig.test(target_val)

history = cida_tet_jig.get_history()

total_epochs_trained = len(history["epoch_indices"])
total_experiment_time_secs = time.time() - start_time_secs

experiment = {
    "experiment_name": experiment_name,
    "parameters": parameters,
    "results": {
        "source_test_label_accuracy": source_test_label_accuracy,
        "source_test_label_loss": source_test_label_loss,
        "target_test_label_accuracy": target_test_label_accuracy,
        "target_test_label_loss": target_test_label_loss,
        "source_test_domain_loss": source_test_domain_loss,
        "target_test_domain_loss": target_test_domain_loss,
        "source_test_label_accuracy":source_test_label_accuracy,
        "source_test_label_loss":source_test_label_loss,
        "source_test_domain_loss":source_test_domain_loss,
        "target_test_label_accuracy":target_test_label_accuracy,
        "target_test_label_loss":target_test_label_loss,
        "target_test_domain_loss":target_test_domain_loss,
        "source_val_label_accuracy":source_val_label_accuracy,
        "source_val_label_loss":source_val_label_loss,
        "source_val_domain_loss":source_val_domain_loss,
        "target_val_label_accuracy":target_val_label_accuracy,
        "target_val_label_loss":target_val_label_loss,
        "target_val_domain_loss":target_val_domain_loss,
        "total_epochs_trained": total_epochs_trained,
        "total_experiment_time_secs": total_experiment_time_secs,
    },
    "history": history,
}

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

print("Source Val Label Accuracy:", source_val_label_accuracy, "Target Val Label Accuracy:", target_val_label_accuracy)
print("Source Test Label Accuracy:", source_test_label_accuracy, "Target Test Label Accuracy:", target_test_label_accuracy)


# We hijack the original loss curves diagram for our own nefarious purposes
plt.rcParams.update({'font.size': 15})
fig, axis = cida_tet_jig._do_diagram()

fig.suptitle("Experiment Summary")
fig.set_size_inches(30, 15)


# https://stackoverflow.com/questions/52480756/change-subplot-dimension-of-existing-subplots-in-matplotlib
#
# The original loss curves use indices [:4]
alpha_curve, train_label_loss_vs_train_domain_loss, source_val_label_loss_vs_target_val_label_loss, source_train_label_loss_vs_source_val_label_loss = fig.axes

gs = matplotlib.gridspec.GridSpec(2,3)

alpha_curve.set_position(gs[1].get_position(fig))
train_label_loss_vs_train_domain_loss.set_position(gs[2].get_position(fig))
source_val_label_loss_vs_target_val_label_loss.set_position(gs[4].get_position(fig))
source_train_label_loss_vs_source_val_label_loss.set_position(gs[5].get_position(fig))


ax = fig.add_subplot(gs[1,0])
ax.set_axis_off() 
ax.set_title("Results")
t = ax.table(
    [
        ["Source Val Label Accuracy", "{:.2f}".format(experiment["results"]["source_val_label_accuracy"])],
        ["Target Val Label Accuracy", "{:.2f}".format(experiment["results"]["target_val_label_accuracy"])],

        ["Source Test Label Accuracy", "{:.2f}".format(experiment["results"]["source_test_label_accuracy"])],
        ["Target Test Label Accuracy", "{:.2f}".format(experiment["results"]["target_test_label_accuracy"])],

        ["Total Epochs Trained", "{:.2f}".format(experiment["results"]["total_epochs_trained"])],
        ["Total Experiment Time", "{:.2f}".format(experiment["results"]["total_experiment_time_secs"])],

        ["Source Test Label Loss", "{:.2f}".format(experiment["results"]["source_test_label_loss"])],
        ["Target Test Label Loss", "{:.2f}".format(experiment["results"]["target_test_label_loss"])],
        ["Source Test Domain Loss", "{:.2f}".format(experiment["results"]["source_test_domain_loss"])],
        ["Target Test Domain Loss", "{:.2f}".format(experiment["results"]["target_test_domain_loss"])],

        ["Source Val Label Loss", "{:.2f}".format(experiment["results"]["source_val_label_loss"])],
        ["Target Val Label Loss", "{:.2f}".format(experiment["results"]["target_val_label_loss"])],
        ["Source Val Domain Loss", "{:.2f}".format(experiment["results"]["source_val_domain_loss"])],
        ["Target Val Domain Loss", "{:.2f}".format(experiment["results"]["target_val_domain_loss"])],

    ],
    loc="best",
)
t.auto_set_font_size(False)
t.set_fontsize(20)
t.scale(1.5, 2)

ax = fig.add_subplot(gs[0,0])
ax.set_axis_off() 
ax.set_title("Parameters")

t = ax.table(
    [
        ["Experiment Name", experiment_name],
        ["Learning Rate", lr],
        ["Num Epochs", n_epoch],
        ["Batch Size", batch_size],
        ["patience", patience],
        ["seed", seed],
        ["device", device],
        ["alpha", alpha],
        ["source_snrs", str(source_snrs)],
        ["target_snrs", str(target_snrs)],
        # ["x_net", str(x_net)],
        # ["u_net", str(u_net)],
        # ["merge_net", str(merge_net)],
        # ["class_net", str(class_net)],
        # ["domain_net", str(domain_net)],
    ],
    loc="best"
)
t.auto_set_font_size(False)
t.set_fontsize(20)
t.scale(1.5, 2)

if not (len(sys.argv) > 1 and sys.argv[1] == "-"):
    plt.show()
plt.savefig(LOSS_CURVE_PATH)