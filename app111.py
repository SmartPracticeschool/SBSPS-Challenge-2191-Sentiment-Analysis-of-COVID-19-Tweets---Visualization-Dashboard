import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.python.framework import ops
graph = ops.get_default_graph()

app = Flask(__name__)
model = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    s = str(name)
    with graph.as_default():
        model = load_model('model.h5')
        tokenizer = Tokenizer(num_words=3000)
        x_test = pad_sequences(tokenizer.texts_to_sequences([s]), maxlen=10)
        # Predict
        score = model.predict([x_test])[0]
        # Decode sentiment
        if score <= 0.4:
            label = "NEGATIVE"
        elif score >= 0.7:
            label = "POSITIVE"
        else:
            label = "NEUTRAL"
    
    return render_template('index.html', prediction_text=label)

@app.route('/predict',methods=['POST'])
def predict(text):
    #start_at = time.time()
    # Tokenize text
    tokenizer = Tokenizer(num_words=3000)
    x_test = pad_sequences(tokenizer.texts_to_sequences([s]), maxlen=10)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    if score <= 0.4:
        label = "NEGATIVE"
    elif score >= 0.7:
        label = "POSITIVE"
    else:
        label = "NEUTRAL"
    
    return render_template('index.html', prediction_text=label)


    
if __name__ == "__main__":
    app.run(debug=True)
