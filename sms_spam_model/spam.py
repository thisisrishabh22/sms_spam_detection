from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import sklearn
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

sms = pd.read_csv('spam.csv', sep='\t', names=['label', 'message'])
sms.drop_duplicates(inplace=True)
sms.reset_index(drop=True, inplace=True)

corpus = []
ps = PorterStemmer()

for i in range(0, sms.shape[0]):
    # Cleaning special character from the message
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms.message[i])
    message = message.lower()  # Converting the entire message into lower case
    words = message.split()  # Tokenizing the review by words
    words = [word for word in words if word not in set(
        stopwords.words('english'))]  # Removing the stop words
    words = [ps.stem(word) for word in words]  # Stemming the words
    message = ' '.join(words)  # Joining the stemmed words
    corpus.append(message)  # Building a corpus of messages

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(sms['label'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

acc_s = accuracy_score(y_test, y_pred)*100
print("Accuracy Score {} %".format(round(acc_s, 2)))


def predict_spam(sample_message):
    sample_message = re.sub(
        pattern='[^a-zA-Z]', repl=' ', string=sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(
        stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)


# Init
if __name__ == "__main__":
    result = ["The following SMS is a SPAM", "The following SMS is normal"]

    sample_sms = [
        "Hi! You are pre-qulified for Premium SBI Credit Card. Also get Rs.500 worth Amazon Gift Card*, 10X Rewards Point* & more. Click ",
        "[Update] Congratulations Nile Yogesh, You account is activated for investment in Stocks. Click to invest now",
        "Your Stock broker FALANA BROKING LIMITED reported your fund balance Rs.1500.5 & securities balance 0.0 as ",
        "We noticed some unusual activity on your bank card. Please reactivate your account here [link] Your Amazon account has been suspended.",
        "This is an urgent request to transfer INR 2000 to the Anti Corruption Organisation otherwise your account will be ceased. Click the following link to complete the payment orelse you will regret",
        "Our records show you overpaid for (a product or service). Kindly supply your bank routing and account number to receive your refund.",
        "Your niece has been arrested and needs $7,500. Kindly complete the payment to avoid further lawsuits"
    ]

    for msg in sample_sms:
        if predict_spam(msg):
            print(result[0])
        else:
            print(result[1])
