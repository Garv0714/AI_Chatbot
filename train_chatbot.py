# -------------------------------
# Chatbot Training - Colab Ready
# -------------------------------
import re
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from convokit import Corpus, download
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# 1️⃣ Download and load the Cornell Movie Corpus
corpus = Corpus(download('movie-corpus'))

# 2️⃣ Prepare input-output pairs
pairs = []
for conv in corpus.iter_conversations():
    utts = list(conv.iter_utterances())  # ✅ Latest ConvoKit me ye sahi hai
    for i in range(len(utts)-1):
        input_line = utts[i].text.lower()
        input_line = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", input_line)
        target_line = utts[i+1].text.lower()
        target_line = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", target_line)
        pairs.append((input_line, target_line))

print(f"Total pairs: {len(pairs)}")

# 3️⃣ Tokenizer
tokenizer = Tokenizer()
all_text = [text for pair in pairs for text in pair]
tokenizer.fit_on_texts(all_text)

# Save tokenizer
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved as 'tokenizer.pickle'. Ready for model training!")

# 4️⃣ Prepare sequences
encoder_texts = [pair[0] for pair in pairs]
decoder_texts = [pair[1] for pair in pairs]
encoder_input = tokenizer.texts_to_sequences(encoder_texts)
decoder_input = tokenizer.texts_to_sequences(decoder_texts)

# Target is decoder input shifted
decoder_target = [seq[1:] + [0] for seq in decoder_input]

max_len = 20  # sequence max length
encoder_input = pad_sequences(encoder_input, maxlen=max_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_len, padding='post')

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
lstm_units = 256

# 5️⃣ Build model
encoder_inputs = Input(shape=(max_len,))
decoder_inputs = Input(shape=(max_len,))

encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)

# LSTM with recurrent_activation='sigmoid' for CPU/GPU compatibility
encoder_lstm = LSTM(lstm_units, return_sequences=True, recurrent_activation='sigmoid')
decoder_lstm = LSTM(lstm_units, return_sequences=True, recurrent_activation='sigmoid')

encoder_outputs = encoder_lstm(encoder_embedding)
decoder_outputs = decoder_lstm(decoder_embedding, initial_state=[encoder_outputs[:, -1, :], encoder_outputs[:, -1, :]])

decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# 6️⃣ Train model
model.fit([encoder_input, decoder_input], np.expand_dims(decoder_target, -1),
          batch_size=64, epochs=10, validation_split=0.1)
