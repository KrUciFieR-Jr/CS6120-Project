import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from config import config

MODEL_PATH = config['MODEL_PATH']


def lstm_model_without_dropout_architecture(train_texts,test_texts,train_labels,test_labels):
    """

    :param train_text:
    :param test_texts:
    :param train_labeltokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Pad the sequences
    max_sequence_length = max(len(seq) for seq in train_sequences)
    train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)s:
    :param test_labels:
    :return:
    """
    tokenizer = Tokenizer(num_words=5000)

    # Tokenize the text
    if MODEL_PATH != 'saved':
        # Define the LSTM model
        model = Sequential()
        model.add(Embedding(5000, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Set up early stopping and model checkpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        checkpoint = ModelCheckpoint('fake_news_detector_model.h5', save_best_only=True, save_weights_only=False)

        # Train the model
        # Train the model
        model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=3, batch_size=64, callbacks=[early_stopping, checkpoint])
        # Save the model
        model.save(f'{MODEL_PATH}/fake_news_detector_model.h5')
    else:
        model = load_model(f'{MODEL_PATH}/fake_news_detector_model.h5')

    y_pred = model.predict(test_data, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(test_labels, y_pred_bool))


def lstm_model_with_dropout_architecture(train_texts,test_texts,train_labels,test_labels):
    """

    :param train_text:
    :param test_texts:
    :param train_labels:
    :param test_labels:
    :return:
    """
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Pad the sequences
    max_sequence_length = max(len(seq) for seq in train_sequences)
    train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)
    # Tokenize the text
    if MODEL_PATH != 'saved':
        # Define the LSTM model
        model = Sequential()
        model.add(Embedding(5000, 128))
        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Set up early stopping and model checkpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        checkpoint = ModelCheckpoint('fake_news_detector_model_second_architecture.h5', save_best_only=True, save_weights_only=False)

        # Train the model
        model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=3, batch_size=64, callbacks=[early_stopping, checkpoint])

        # Save the model
        model.save(f'{MODEL_PATH}/fake_news_detector_model_second_architecture.h5')
    else:
        model = load_model(f'{MODEL_PATH}/fake_news_detector_model_second_architecture.h5')


    y_pred = model.predict(test_data, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(test_labels, y_pred_bool))


