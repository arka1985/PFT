import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model_FVC = pickle.load(open('model1.pkl', 'rb'))
model_FVC_LLN = pickle.load(open('model2.pkl', 'rb'))
model_FEV1 = pickle.load(open('model3.pkl', 'rb'))                                
model_FEV1_LLN = pickle.load(open('model4.pkl', 'rb'))
                                 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    dummy_data = {'age':[18, 34, 42, 50, 23, 35, 32],
        'height':[150, 152, 156, 148, 160, 149, 148],
        'gender':['male','female','male','male','male','male','female'],
        'region':['n','s','e','w','c','ne','ns'],
        'smoking':['yes','no','no','no','no','no','yes']}
    dummy_df = pd.DataFrame(dummy_data)
    age = request.form.get('age')
    height = request.form.get('height')
    gender = request.form.get('gender')
    region = request.form.get('region')
    smoking = request.form.get('smoking')
    new_obj = {"age" : age, "height" : height, "gender" : gender, "region" : region, "smoking": smoking}
    dummy_df.loc[len(dummy_df.index)] = new_obj
    dummy_df = dummy_df.astype({'age':'float','height':'float'})
    dummy_df = pd.get_dummies(dummy_df)
    dummy_df = dummy_df.astype({'age':'float','height':'float','gender_female':'category','gender_male':'category','region_c':'category','region_e':'category','region_n':'category','region_ne':'category','region_ns':'category','region_s':'category','region_w':'category','smoking_no':'category','smoking_yes':'category'})
    new_data = dummy_df.iloc[-1, :].values.tolist()
    new = [round(x) for x in new_data]
    prediction_FVC = model1.predict([new])
    prediction_FVC_LLN = model2.predict([new])
    prediction_FEV1 = model3.predict([new])
    prediction_FEV1_LLN = model4.predict([new])

    output_FVC = round(prediction_FVC[0], 2)
    output_FVC_LLN = round(prediction_FVC_LLN[0], 2)
    output_FEV1 = round(prediction_FEV1[0], 2)
    output_FEV1_LLN = round(prediction_FEV1_LLN[0], 2)

    return render_template('index.html', 
                           prediction_text_FVC='Predicted FVC  {}'.format(output_FVC),
                           prediction_text_FVC_LLN='Predicted LLN FVC  {}'.format(output_FVC_LLN),
                           prediction_text_FEV1='Predicted FEV1  {}'.format(output_FEV1),
                           prediction_text_FEV1_LLN='Predicted LLN FEV1  {}'.format(output_FEV1_LLN))
    


if __name__ == "__main__":
    app.run(port=8080)
