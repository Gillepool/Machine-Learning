from optparse import OptionParser

import matplotlib.pyplot as plt

from networks.NeuralNet import FullyConnectedNet
from networks.cnn import CNN
from data import mnist
import trainer

parser = OptionParser()
parser.add_option("-m", "--model", dest="model",
                  help="choose model", default=CNN)

args = parser.parse_args()
data = mnist.load_mnist_data()

''' CNN is super slow to train, using a naive implementation'''
#model = CNN()

model = FullyConnectedNet([300], dropout=0.2)



trainer = trainer.Trainer(model, data)
trainer.train()

plt.subplot(2, 1, 1)
plt.plot(trainer.get_losshistory(), 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(trainer.get_train_acc_history(), '-o')
plt.plot(trainer.get_val_acc_history(), '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()