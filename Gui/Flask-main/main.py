from operator import index
from select import select
from flask import Flask, render_template, request, flash, redirect, url_for, Response
from prometheus_client import Counter
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from wtforms.validators import InputRequired
from wtforms import MultipleFileField
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import preprocessing
import predict
import pandas as pd
import numpy as np
import random
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# create Flask app
datatype = ""
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'


global counter
counter = 0


## Upload Form
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

class Submit(FlaskForm):
    submit = SubmitField("Predict")



@app.route('/', methods=['GET', "POST"])
#@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    datatype = ['Multi Int', 'Multi Float', 'HouseA', 'HouseB']
    model = ['LSTM', 'CNN', 'SPEED']
    
    if form.validate_on_submit():
        global type
        global model_type 
        type_dat = request.form["datatype"]
        model_chosen = request.form["model"]
        model_type = model_chosen
        file = form.file.data  # First grab the file
        if type_dat == "HouseA":
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                "Aras.txt"))  # Then save the file
            type = "HouseA"
        elif type_dat == "HouseB":
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                "Aras.txt"))  # Then save the file
            type = "HouseB"
        elif type_dat == "Multi Int":
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                "Multi_int.csv"))  # Then save the file
            type = "Multi"
        elif type_dat == "Multi Float":
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                "Multi_float.csv"))  # Then save the file
            type = "Multi"
        

       
        #return f"File has been uploaded data of {type_dat}"
    preprocess()
       
    return render_template('index.html', form=form, datatype=datatype, model = model, section = True)


@app.route("/predict")
def pred():
    print("",type)
    print("",model_type)
    return render_template('predict.html')

def select_dataset(type):
    if type == "HouseA":
            df = preprocessing.process_data_Aras("static/files/Aras.txt",type)
            print("",type)
            print("",model_type)
    elif type == "HouseB":
            df = preprocessing.process_data_Aras("static/files/Aras.txt",type)
            print("",type)
            print("",model_type)
    else:
            df = preprocessing.preprocess_multi("static/files/Multi_float.csv", "static/files/Multi_int.csv")
            print("",type)
            print("",model_type)
    
    return df
    
@app.route('/predict', methods=['GET', "POST"], endpoint='graph')
def preprocess():
    if request.form.get('Predict') == 'Predict':
        
        df = select_dataset(type)

        if model_type == "SPEED":
            pass

        else :
            X = predict.preprocessing_2(df)

            if model_type == "CNN":
                print(X)
                model_cnn = predict.init_model_cnn()
                print(model_cnn)
                out = predict.predict(model_cnn,X)
                print(out)
                y_pred_sensor= []
                y_pred_prob= []
                for i in range(0,len(out)): 
                    y_pred_sensor.append(np.argmax(out[i]))
                    y_pred_prob.append(np.max(out[i]))
                dic = {"sen": y_pred_sensor, "prob":y_pred_prob}
                df = pd.DataFrame(data = dic)
                df.to_csv('static/files/Output.csv') 

            else:
                model_lstm = predict.init_model_lstm()
                out = predict.predict(model_lstm,X)
                print(out)
                y_pred_sensor= []
                y_pred_prob= []
                for i in range(0,len(out)): 
                    y_pred_sensor.append(np.argmax(out[i]))
                    y_pred_prob.append(np.max(out[i]))
                dic = {"sen": y_pred_sensor, "prob":y_pred_prob}
                df = pd.DataFrame(data = dic)
                df.to_csv('static/files/Output.csv')
                data = pd.DataFrame(X)
                 
        return redirect(url_for('graph.html'))
    


@app.route("/graph", methods=['GET', 'POST'])
def chart():
    counter = 0
    border = counter + 20
    data = pd.read_csv("static/files/data.csv")
    prediction = pd.read_csv("static/files/Output.csv")
    add = prediction.iloc[counter]
    events = data.loc[counter:border, "event"]
    events = events.tolist()
    events.append(add.loc["sen"])
    print(events)
    label = [str(x) for x in range(1,21)]
    label.append("Pred")

    if request.method == 'POST':
        if request.form.get('action1') == 'Backward':
            if counter == 0:
                return render_template('predict.html')
            else:
                counter += -1
                return render_template('graph.html', title='Visualization', max=24, labels=labels, values=values)

            
        if  request.form.get('action2') == 'Forward':
            counter += 1
            return render_template('graph.html', title='Visualization', max=24, labels=labels, values=values)

    labels = label

    values = events
    
    return render_template('graph.html', title='Visualization', max=24, labels=labels, values=values)




def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
