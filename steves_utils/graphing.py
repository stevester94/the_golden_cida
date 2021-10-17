import matplotlib.pyplot as plt
import numpy as np



def _do_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def _do_loss_curve(history):
    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='Training Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')


def plot_confusion_matrix(cm):
    _do_confusion_matrix(cm)
    plt.show()

def plot_loss_curve(history):
    _do_loss_curve(history)
    plt.show()


def save_confusion_matrix(cm, path="./confusion_matrix.png"):
    _do_confusion_matrix(cm)
    plt.savefig(path)

def save_loss_curve(history, path="./loss_curve.png"):
    _do_loss_curve(history)
    plt.savefig(path)