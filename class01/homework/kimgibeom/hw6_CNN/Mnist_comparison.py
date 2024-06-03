import pickle
import numpy as np
import matplotlib.pyplot as plt
historyNoBatch = pickle.load(open('./historyMnist_withoutbatch', "rb"))
historyBatch = pickle.load(open('./historyMnist_withbatch', "rb"))
historylelu = pickle.load(open('./historyMnist_relu', "rb"))
historyAnn = pickle.load(open('./historyMnist_ann', "rb"))
val_accNB = historyNoBatch["val_accuracy"]
val_lossNB= historyNoBatch["val_loss"]
val_lossB = historyBatch["val_loss"]
val_accB = historyBatch["val_accuracy"]
val_lossL = historylelu["val_loss"]
val_accL = historylelu["val_accuracy"]
val_lossA = historylelu["val_loss"]
val_accA = historylelu["val_accuracy"]

plt.subplot(1,2,1)
plt.title('Validation Loss')
plt.plot(range(len(val_lossNB)),val_lossNB,label = "sigmoid (No Batch)")
plt.plot(range(len(val_lossB)),val_lossB,label = "sigmoid (Batch)")
plt.plot(range(len(val_lossL)),val_lossL,label = "relu")
plt.plot(range(len(val_lossA)),val_lossA,label = "Ann")
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.title('Validation Accuracy')
plt.plot(range(len(val_accNB)),val_accNB,label = "sigmoid (No Batch)")
plt.plot(range(len(val_accB)),val_accB,label = "sigmoid (Batch)")
plt.plot(range(len(val_accL)),val_accL,label = "relu")
plt.plot(range(len(val_accA)),val_accA,label = "Ann")
plt.grid()
plt.legend()
plt.show()
plt.savefig("Summary.png")