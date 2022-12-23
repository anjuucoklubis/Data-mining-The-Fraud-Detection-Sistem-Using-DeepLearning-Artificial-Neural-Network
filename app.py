from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', insurance_cost=2)

 
@app.route('/predict', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        
        with open('model.pkl','rb') as r:
            model = pickle.load(r)
            
        kdkc = float(request.form['kdkc'])
        dati2 = float(request.form['dati2'])
        typeppk = float(request.form['typeppk'])
        umur = float(request.form['umur'])
        jkpst = float(request.form['jkpst'])

        datas = np.array((kdkc,dati2,typeppk,umur,jkpst))
        datas = np.reshape(datas,(1,-1))
        
        isDiabetes = model.predict(datas)
        
        if isDiabetes == 1:
            output = "Fraud"
        else:
            output = "Non-Fraud"
        
        return render_template('index.html', finalData=output)
if __name__ == '__main__':
    app.run(debug=True)