from flask import Flask
import json
import model
from flask import request
from tflearn.data_utils import to_categorical, pad_sequences


app = Flask(__name__)

dnn = model.DNN()
dnn.load("model.tfl", weights_only=True)

@app.route("/")
def hello():
    data = request.args.get('data')
    input = [bytearray(data, 'utf8')]
    input = pad_sequences(input, maxlen=model.MAX_SEQ_LENGTH, value=0.)

    response = {
        'score': dnn.predict(input)[0]
    }
    return json.dumps(response)

if __name__ == "__main__":
    app.run()