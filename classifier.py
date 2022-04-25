import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# import 20 news group dataset
from sklearn.datasets import fetch_20newsgroups

# the predefined categories
categories = ['rec.motorcycles', 'sci.electronics',
              'comp.graphics', 'sci.med']

# use sklearn subset data for training and testing
train_data = fetch_20newsgroups(subset='train',
                                categories=categories, shuffle=True, random_state=42)

# size of training dataset
print(train_data.target.size)

# a dictionary of features and transforms documents to feature vectors and convert text documents to a
# matrix of token counts (CountVectorizer)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data.data)

# transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
# tf-idf term frequency-inverse document frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# training our classifier ; train_data.target will be having numbers 
# assigned for each category in train data
clf = MultinomialNB().fit(X_train_tfidf, train_data.target)

# Input Data to predict their classes of the given categories
docs_new = ['Bentley is a car', 'Ducati is a bike']
# building up feature vector of our input
X_new_counts = count_vect.transform(docs_new)
# transform is called instead of fit_transform because it's already been fit
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicting the category of our input text: Will give out number for category
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train_data.target_names[category]))

# We can use Pipeline to add vectorizer -> transformer -> classifier all in a one compound classifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# Fitting train data to the pipeline
text_clf.fit(train_data.data, train_data.target)

# Test data 
test_data = fetch_20newsgroups(subset='test',
                               categories=categories, shuffle=True, random_state=42)
docs_test = test_data.data
# Predicting on test data
predicted = text_clf.predict(docs_test)
print('Accuracy of',np.mean(predicted == test_data.target)*100, '% over the test data.')
