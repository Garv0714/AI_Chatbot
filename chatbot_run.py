import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("chatbot_model_tf.keras")

# Load tokenizer
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 20  # same as training

def chatbot_response(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict([seq, seq])
    pred_word_index = pred.argmax(axis=-1)[0][0]
    for word, index in tokenizer.word_index.items():
        if index == pred_word_index:
            return word
    return "Sorry, I didn't understand."

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Bye!")
        break
    print("Chatbot:", chatbot_response(user_input))
