from django.shortcuts import render

# our home page view
def home(request):
    return render(request, 'index.html')

# custom method for generating predictions
def getPredictions(Age, Anaemia, Creatinine_phosphokinase, Diabetes, Ejection_fraction, High_blood_pressure, Platelets,
                   Serum_creatinine, Serum_sodium, Sex, Smoking):
    import pickle
    model = pickle.load(open("heart-prediction-rfc-model.sav", "rb"))
    scaled = pickle.load(open("s_scaler.sav", "rb"))
    prediction = model.predict(scaled.transform([[Age, Anaemia, Creatinine_phosphokinase, Diabetes, Ejection_fraction,
                                                    High_blood_pressure, Platelets, Serum_creatinine, Serum_sodium,
                                                    Sex, Smoking]]))

    if prediction == 0:
        return "Great! You don't have heart failure."
    elif prediction == 1:
        return "Oops! You have heart failure."
    else:
        return "error"

# our result page view
def result(request):
    Age = float(request.GET['age'])
    Anaemia = int(request.GET['anaemia'])
    Creatinine_phosphokinase = int(request.GET['creatinine_phosphokinase'])
    Diabetes = int(request.GET['diabetes'])
    Ejection_fraction = int(request.GET['ejection_fraction'])
    High_blood_pressure = int(request.GET['high_blood_pressure'])
    Platelets = float(request.GET['platelets'])
    Serum_creatinine = float(request.GET['serum_creatinine'])
    Serum_sodium = int(request.GET['serum_sodium'])
    Sex = int(request.GET['sex'])
    Smoking = int(request.GET['smoking'])

    result = getPredictions(Age, Anaemia, Creatinine_phosphokinase, Diabetes, Ejection_fraction, High_blood_pressure,
                            Platelets, Serum_creatinine, Serum_sodium, Sex, Smoking)

    return render(request, 'result.html', {'result': result})