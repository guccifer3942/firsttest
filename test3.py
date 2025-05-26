
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# 参数配置
MAX_WORDS = 10000
MAX_LEN = 200
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 15

def load_imdb_data(data_path):
    texts, labels = [], []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(data_path, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, np.array(labels)

# 数据加载
train_texts, train_labels = load_imdb_data('D:/aclImdb_v1/aclImdb/train')
test_texts, test_labels = load_imdb_data('D:/aclImdb_v1/aclImdb/test')

# 文本预处理
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(train_sequences, maxlen=MAX_LEN)
X_test = pad_sequences(test_sequences, maxlen=MAX_LEN)
y_train, y_test = train_labels, test_labels

# 构建模型
model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练配置
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# 训练模型
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=callbacks
)

# 评估模型
model.load_weights('best_model.h5')
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# 可视化结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('training_metrics.png')
plt.show()
