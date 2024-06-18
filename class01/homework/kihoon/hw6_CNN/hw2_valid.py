import pickle
import matplotlib.pyplot as plt

# 각 모델에 대한 학습 이력 파일의 경로
history_files = ['historyBatchReLu1.pickle', 'historyBatchReLu2.pickle', 'historyBatchReLu3.pickle']

# Validation Loss와 Validation Accuracy를 저장할 리스트
val_losses = []
val_accuracies = []

# 각 모델에 대해 학습 이력을 로드하고 Validation Loss와 Validation Accuracy를 추출
for history_file in history_files:
    with open(history_file, 'rb') as file:
        history = pickle.load(file)
        val_loss = history['val_loss']
        val_acc = history['val_accuracy']
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

# 그래프 그리기
epochs = range(1, len(val_losses[0]) + 1)

plt.figure(figsize=(12, 6))

# Validation Loss 그래프
plt.subplot(1, 2, 1)
for i, val_loss in enumerate(val_losses):
    plt.plot(epochs, val_loss, label='Model {}'.format(i+1))
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Validation Accuracy 그래프
plt.subplot(1, 2, 2)
for i, val_acc in enumerate(val_accuracies):
    plt.plot(epochs, val_acc, label='Model {}'.format(i+1))
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()