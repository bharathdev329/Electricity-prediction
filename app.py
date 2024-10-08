from flask import Flask, request, render_template
import joblib
import numpy as np
import datetime as dt

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('electricity trained.pkl')

# Function to collect date-related inputs based on user input
def get_date_features(year, month, day):
    # Use the user-provided date
    user_date = dt.datetime(year, month, day)

    # Calculate PeriodOfDay based on the current time (this remains automatic)
    now = dt.datetime.now()
    period_of_day = get_period_of_day(now)

    return {
        'DayOfWeek': user_date.weekday(),
        'WeekOfYear': user_date.isocalendar()[1],
        'Day': user_date.day,
        'Month': user_date.month,
        'Year': user_date.year,
        'PeriodOfDay': period_of_day
    }

def get_period_of_day(now):
    # Combine the hour and minute to calculate the period (0-47)
    hour = now.hour
    minute = now.minute
    period_of_day = (hour * 2) + (minute // 30)  # 0-47 (30-minute periods)
    return period_of_day

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected holiday
    selected_holiday = request.form['Holiday']
    holiday_flag = 0 if selected_holiday == "None" else 1

    # Get user-provided date inputs
    year = int(request.form['Year'])
    month = int(request.form['Month'])
    day = int(request.form['Day'])

    # Collect date features based on user input
    date_features = get_date_features(year, month, day)

    # Get other manual inputs from the form
    input_features = [
        float(request.form['ForecastWindProduction']),
        float(request.form['SystemLoadEA']),
        float(request.form['SMPEA']),
        float(request.form['ORKTemperature']),
        float(request.form['ORKWindspeed']),
        float(request.form['CO2Intensity']),
        float(request.form['ActualWindProduction']),
        float(request.form['SystemLoadEP2'])
    ]

    # Combine manual inputs with automatic date features (and include HolidayFlag)
    features = np.array([
        date_features['DayOfWeek'],
        date_features['WeekOfYear'],
        date_features['Day'],
        date_features['Month'],
        date_features['Year'],
        date_features['PeriodOfDay'],
        holiday_flag,
        *input_features
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return prediction result
    return render_template('index.html', prediction_text=f'${prediction[0]:.2f}')


if __name__ == "__main__":
    app.run(debug=True)
