from flask import Flask, render_template, url_for, request, Markup, jsonify
#from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle
import re
import os
import numpy as np

app = Flask(__name__)
pickle_in = open('model_fakenews.pickle','rb')
pac = pickle.load(pickle_in)
tfid = open('tfid.pickle','rb')
tfidf_vectorizer = pickle.load(tfid)

# Used by bow pickle file
def clean_article(article):
    art = re.sub("[^A-Za-z0-9' ]", '', str(article))
    art2 = re.sub("[( ' )(' )( ')]", ' ', str(art))
    art3 = re.sub("\s[A-Za-z]\s", ' ', str(art2))
    return art3.lower()


 


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['article']
         
 


        abc = comment
        input_data = [abc.rstrip()]
        # transforming input
        tfidf_test = tfidf_vectorizer.transform(input_data)
        # predicting the input
        y_pred = pac.predict(tfidf_test)
         
        x='FAKE'
        val=0
        print("ccccccccccccccccccccccccccccccccccccccc")
        print(y_pred[0])
        if y_pred[0] == 'FAKE':
           val=1
           print(val)
        else:
           val=0
           print(val)
       

    return render_template('result.html', prediction=val)




if __name__ == '__main__':
    app.run(debug=True)
    #bow = pickle.load(open("bow.pkl", "rb"))
    #model = pickle.load(open("model.pkl", "rb"))
