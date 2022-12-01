import re
from datetime import datetime
import json
from flask import Flask, jsonify
from LinearRegression import XGBoostHelper
app = Flask(__name__)

@app.route('/hello/')
def hello_there():
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
   
    clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content

@app.route('/hellotest/')
def hello_there_a():
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
   
    clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content

@app.route('/test', methods=['GET', 'POST'])
def testjson():
    lr = XGBoostHelper('C:\\Users\\Vasyl\\Desktop\\practice\\historical education students data.csv', 'Target', 0.9, 0.1)
    return  json.dumps(lr.startProcess())
                       
if __name__ == '__app__':
    app.run(debug=True)

