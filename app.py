import os
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image

from flask import Flask, request, render_template
import pickle

# Create app object using Flask class.
app = Flask(__name__)

# Load the trained model.
model = pickle.load(open('models/model.pkl', 'rb'))


# Use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')


# POST : Used to send HTML form data to the server
# Redirect to /predict page with the output
@app.route('/predict',method=['POST'])
def predict():
    path = request.form.values()
    test_image = image.load_img(path, target_size=(180, 180))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    classes = os.listdir('dataset/training_set')

    index = int(np.argmax(result, axis=-1))

    return render_template('index_html', prediction_text=classes[index])


if __name__=="__main__":
    app.run()
