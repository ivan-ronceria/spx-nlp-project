import pandas as pd
#converts articles to numbers
from sklearn.feature_extraction.text import TfidfVectorizer
#creates training and testing sets
from sklearn.model_selection import train_test_split
#Naive Bayes model with normal distibution assumption
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#imports wordcloud visualizatons
import matplotlib.pyplot as plt
#pip install wordcloud - does not come with Anaconda
from wordcloud import WordCloud

#logistic regression model imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from nltk.stem.snowball import SnowballStemmer

#reads in CSV of articles with predetermined sentiment scores
corpus = pd.read_csv(r'C:\Users\donal\OneDrive\Documents\College Documents\Cryptocurrency Club\QMAT Committee\Sentiment Analysis\scored_articles.csv', encoding='cp1252')
corpus

#PREPROCESS HERE.
nonum = corpus.Text.str.replace('\d+', '')

#initalizing our TF-IDF vectorizer
#sublinear - apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)
#max_df - ignore terms that have a document frequency strictly higher than the given threshhold
#ngram's allow us to store more context from our corpus
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english", ngram_range=(1,2))

#Visualization
#Separating corpus into distinct high and low volatility classifications
highVol = corpus[(corpus['Score']==1)]
lowVol = corpus[(corpus['Score']==0)]

#removing numbers from corpus
highVol = highVol.Text.replace('\d+', '') 
lowVol = lowVol.Text.str.replace('\d+', '') 

#converting series to string for use with WordCloud
highVol = highVol.to_string()
lowVol = lowVol.to_string()

#generating WordCloud for high volatility corpus
highVolWC = WordCloud(background_color="black", max_words=50, width=2000, height=1000, random_state=1).generate(highVol)

#to recolour the image
plt.imshow(highVolWC.recolor())

#generating WordCloud for high volatility corpus
highVolWC = WordCloud(background_color="black", max_words=50, width=2000, height=1000, random_state=1).generate(lowVol)

#to recolour the image
plt.imshow(highVolWC.recolor())

#model development
#initializes our training and testing sets
X_train, X_test, y_train, y_test = train_test_split(nonum, corpus.Score, test_size=0.25, random_state=10)

#model did not take our sparse matrix, so we coverted to an array
#calculates ill do this later
X_train = vect.fit_transform(X_train).toarray()
X_test = vect.transform(X_test).toarray()

#Reveals the features we will be working with
print(vect.get_feature_names())

#Gaussian Naive Bayes Model
#initializing our Gaussian Naive Bayes classifier
GaussNB = GaussianNB()
GaussNB.fit(X_train, y_train)

gauss_score_train = GaussNB.score(X_train, y_train)
gauss_score_test = GaussNB.score(X_test, y_test)

print(gauss_score_train)
print(gauss_score_test)

#Multinomial Naive Bayes Model
MultinomialNB = MultinomialNB.fit(X_train, y_train)

mn_score_train = MultinomialNB.score(X_train, y_train)
mn_score_test = MultinomialNB.score(X_test, y_test)

print(mn_score_train)
print(mn_score_test)

#Logistic Regression
logit = LogisticRegression()
logit.fit(X_train, y_train)
predicted_classes = logit.predict(X_train)
#accuracy = accuracy_score(y_train.flatten(), predicted_classes)
parameters = logit.coef_
