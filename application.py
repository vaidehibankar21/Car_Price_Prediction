from flask import Flask, render_template, request
import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car = pd.read_csv("Cleaned Car.csv")

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = {}
    for c in companies:
        models = sorted(car[car['company'] == c]['name'].unique())
        car_models[c] = models
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies = companies, car_models = car_models, years = year, fuel_types = fuel_type)


@app.route('/predict', methods = ['POST'])
def predict():
    company = request.form['company']
    car_model = request.form['car_model']
    year = int(request.form['year'])
    fuel = request.form['fuel_type']
    km = int(request.form['kilo_driven'])
    print(company, car_model, year, fuel, km)
    input_df = pd.DataFrame([[company, car_model, year, km, fuel]], columns=['company', 'name', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(input_df)
    return f"{prediction[0]:.2f}"
if __name__ == "__main__":
    app.run(debug=True)