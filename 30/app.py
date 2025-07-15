import numpy as np
from keras.layers import Input, Dense, BatchNormalization, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
import h5py

def load_dataset(train_path, test_path):
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_path = './30/data/train_happy.h5'
test_path = './30/data/test_happy.h5'
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(train_path, test_path)

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T


def HappyModel(input_shape):
    X_input= Input(input_shape)

    X = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)  

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    return model

happyModel = HappyModel(X_train.shape[1:])
happyModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
happyModel.fit(x=X_train, y=Y_train, epochs=20, batch_size=50)

def predict_image(indx):
    x = X_test[indx]
    x = np.expand_dims(x, axis=0)                  

    prob = happyModel.predict(x)[0][0]               
    label = int(prob >= 0.5)
    return label


from flask import Flask, render_template, request, send_file
import cv2
import io

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.get("/results")
def get_results():
    photo_id = request.args.get("photo_id", type=int)

    try:
        data = {
        "ground_truth": int(Y_test[photo_id, 0]),
        "predicted": int(predict_image(photo_id))
        }
        return render_template("results.html", data=data, photo_id=photo_id)
    except IndexError:
        return  "Error, index out of range"

@app.get("/image_<int:photo_id>")
def show_img(photo_id: int):
    res_img = X_test_orig[photo_id]
    _, res_img = cv2.imencode(".jpg", res_img)
    return send_file(io.BytesIO(res_img.tobytes()), mimetype='image/jpg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


