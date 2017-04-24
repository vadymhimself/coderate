import tflearn
import os
from sklearn.metrics import precision_recall_fscore_support

MAX_SEQ_LENGTH = 500

def DNN():
        # Network building
        net = tflearn.input_data([None, MAX_SEQ_LENGTH])
        net = tflearn.embedding(net, input_dim=256, output_dim=128)
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 2, activation='softmax')

        def f1(y_pred, y_true, x):
            return precision_recall_fscore_support(y_true, y_pred)

        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='categorical_crossentropy')

        model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='model.tfl.ckpt')

        return model