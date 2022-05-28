from flask import Flask, request, jsonify,url_for
from flask import render_template
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pandas as pd


app = Flask(__name__) 
  


@app.route('/') 
def index():
    return render_template('index.html')   


@app.route('/result', methods=['POST'])
def result():
    age = request.form.get('age')
    income = request.form.get('income')
    exp = request.form.get('exp')
    zip = request.form.get('zip_code')
    family = request.form.get('family')
    no_of_loan = request.form.get('no_of_loan')
    mortgage = request.form.get('mortgage')
    cc_avg = request.form.get('cc_avg')
    edu = request.form.get('edu')
    online = request.form.get('Online')
    sec_account = request.form.get('sec_account')
    is_credit_card = request.form.get('is_credit_card')
    str_val_list = [age,exp,income,zip,family,cc_avg,
                         edu,mortgage,no_of_loan,sec_account,online,is_credit_card]
    person_details1 = [int(item) for item in str_val_list ] 
    sec_list= [1]
    sec_list.extend(person_details1)
    person_details = [sec_list]
    bank_df=pd.read_csv('UniversalBank.csv')
    X = bank_df.drop(columns=['Personal Loan'])
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    new_vector = scaler_x.transform(scaler_x.transform(person_details))
    ANN_model = load_model("m.hdf5")
    res= ANN_model.predict(new_vector)
    success_message = 'This Person will accept the loan.'
    error_message = 'This Person will not accept the loan.'
    message = success_message if res[0][0]==1.0 else error_message
    return render_template('index.html',name=message,list_m= person_details1)



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)