import sframe

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_dataset = [
    {'text': 'This product is excellent. I love it.', 'sentiment': 1},
    {'text': 'I hate this fucking product. Piece of shit.', 'sentiment': -1},
    {'text': 'I love this excellent product. Great great great',
        'sentiment': 1},
    {'text': 'Hate hate hate! Never again. Bad', 'sentiment': 1},
    {'text': 'Bad product. Really bad. I hate it.', 'sentiment': -1}
]

sf = sframe.SFrame(train_dataset)
data = sf.unpack('X1')

count_vect = CountVectorizer()
counts = count_vect.fit_transform(data['X1.text'])

regression = LogisticRegression()
model = regression.fit(counts, data['X1.sentiment'])

predict_data = ["I love this product.", "This is a really bad product."]
predict_count_vects = CountVectorizer(vocabulary=count_vect.vocabulary_)
predict_counts = predict_count_vects.fit_transform(predict_data)

predictions = model.predict(predict_counts)

for text, prediction in zip(predict_data, predictions):
    print "'{0}' seems {1}".format(
        text, 'positive' if prediction == 1 else 'negative')
