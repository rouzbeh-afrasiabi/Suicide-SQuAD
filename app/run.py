import sys
import os
sys.path.append('../')

#from models.train_classifier import *
import simplejson  as json
import pandas as pd


from flask import Flask
from flask import render_template, request, jsonify
from joblib import dump, load
import cloudpickle as cp
from io import BytesIO
import base64
import matplotlib.pyplot as plt

from suicide_squad.model import *
from suicide_squad.model_configs import *

app = Flask(__name__)

def get_image(ax):
    fig = ax.get_figure()
    buf = BytesIO()
    fig.savefig(buf, format='png',bbox_inches='tight')
    buf.seek(0)
    encoded_file = base64.b64encode(buf.read()).decode('ascii')
    return(encoded_file)
    


#model = build_model(estimator)
model_configs['cache_dir']=os.path.join('..//cache_dir')
model=QuestionAnswering(model_configs)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():


    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    question = request.args.get('question')
    text = request.args.get('text')
    answer=model.predict_pretrained(question,text)
    fig,ax=model.predict_pretrained_plot(question,text)
    scores_img=get_image(ax)
    
    return render_template(
        'go.html',
        question=question,text=text,answer=answer,scores_img=scores_img)


def main():

    app.run(debug=False)


if __name__ == '__main__':
    main()
