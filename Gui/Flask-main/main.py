from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from wtforms import MultipleFileField
import preprocessing

# create Flask app
datatype = ""
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

## Upload Form
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

class Submit(FlaskForm):
    button = SubmitField("preprocess")


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
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

        print("up")
        return f"File has been uploaded data of {type_dat}"
    preprocess()
       
    return render_template('index.html', form=form, datatype=datatype, model = model, section = True)


def preprocess():
    
    if request.form.get('Preprocessing') == 'Preprocessing':
        print(type(type))
        return f"preprocess..."
    if type == "Aras":
        df = preprocessing.process_data_Aras("static/files/Aras.txt",type)
    else:
        df = preprocessing.preprocess_multi("static/file/Multi_float.csv", "static/file/Multi_int.csv")
    


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
