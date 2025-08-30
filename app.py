from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load model & tokenizer
model = load_model("chatbot_model_tf.keras")
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 20

def predict_response(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = np.pad(seq, ((0,0),(0,max_len-len(seq[0]))), mode='constant')
    pred = model.predict([seq, seq])
    # Simple way: convert predicted indices back to words
    words = []
    for idx in np.argmax(pred, axis=-1)[0]:
        for word, i in tokenizer.word_index.items():
            if i == idx:
                words.append(word)
                break
    return " ".join(words)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    msg = request.form["msg"]
    reply = predict_response(msg)
    return reply

if __name__ == "__main__":
    app.run(debug=True)
