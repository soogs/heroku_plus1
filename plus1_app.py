# Code for the Web App using Flask

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# this creates the "Flask app"

from plus1_pickle import plus1_predict 
# this is so dumb, but I need to add this so that the function is run. But in this case, what is the point of pickling then?? no idea..

model = pickle.load(open('plus1_pickled.pkl', 'rb'))
# this loads the pickle saved model


# the following renders the webpage written by 'index.html'
# for some reason, this html file needs to be within the directory:
# "./template/index.html". 
# Otherwise it does not render
@app.route('/')
def home():
    return render_template('index.html')


# the following is a "post method"
# this provides some features to the model, lets users provide inputs

# because the "/predict" is here, that is connected with the "component scores??" button on the HTML!
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    # making the inputs be compiled into a list

    final_features = [np.array(float_features)]
    # making the features into np array

    component_scores = model.predict(final_features)
    # applying the plus1 prediction

    output = round(component_scores[0], 5)

    w_new_round = np.round(model.w_new,3)

    return render_template('index.html', prediction_text='According to the Plus1 model with weights {}, component scores should be: {}'.format(w_new_round, output))


# the main function "runs this entire flask"
if __name__ == "__main__":
    app.run(debug=True)

# after running with this python code, we get a local IP.
# copy-paste this local IP and we see that it's running! Cool