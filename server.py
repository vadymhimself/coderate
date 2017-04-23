from flask import Flask
import json
import model
import numpy as np
from flask import request
from tflearn.data_utils import to_categorical, pad_sequences


app = Flask(__name__)

dnn = model.DNN()
dnn.load("model.tfl", weights_only=True)

def split_data(array):
    sub_arrays_num = max(1, len(array) // model.MAX_SEQ_LENGTH)
    data = np.array_split(array, sub_arrays_num)
    return pad_sequences(data, maxlen=model.MAX_SEQ_LENGTH, value=0.)

@app.route("/", methods=['POST'])
def hello():
    data = request.form["data"]
    input = split_data(bytearray(data, encoding='utf8'))
    result = np.array(dnn.predict(input))
    print result
    score = np.mean(result.T[0])

    response = {
        'score': score
    }
    return json.dumps(response)

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "X-Requested-With"
    return response

if __name__ == "__main__":
    app.run()