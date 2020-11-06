from django.shortcuts import render
import pandas as pd

# our home page view
def home(request):
    return render(request, 'index.html')

# custom method for generating predictions
def getPredictions(Title, Author, Text):
    import pickle
    model = pickle.load(open("fake_news_model.sav", "rb"))
    vectorizer = pickle.load(open("fake_news_vectorizer.sav", "rb"))
    transformer = pickle.load(open("fake_news_transformer.sav", "rb"))
    df = pd.DataFrame([[Title, Author, Text]], columns=['title', 'author', 'text'])
    vectorized = vectorizer.transform(df)
    prediction = model.predict(transformer.transform(vectorized))

    if prediction == 0:
        return "TRUE"
    elif prediction == 1:
        return "FAKE"
    else:
        return "error"

# our result page view
def result(request):
    Title = str(request.GET['title'])
    Author = str(request.GET['author'])
    Text = str(request.GET['text'])

    result = getPredictions(Title, Author, Text)

    return render(request, 'result.html', {'result': result})