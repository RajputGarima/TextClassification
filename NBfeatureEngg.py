import json
import pandas as pd
from nltk import bigrams, trigrams
import math
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
import sys


stop_words = set(stopwords.words('english'))

file = sys.argv[1]
file_test = sys.argv[2]
mode = sys.argv[3]

# reading json file
data = []
with open(file) as f:
    for line in f:
        data.append(json.loads(line))

test_data = []
with open(file_test) as f:
    for line in f:
        test_data.append(json.loads(line))


# converting json list to data frame df --> contains entire json file
df = pd.DataFrame(data)
df_test = pd.DataFrame(test_data)

# dropping unwanted attributes
df.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)
df_test.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)


# prior probabilities
count = Counter(df['stars'])
term_freq = defaultdict(Counter)
doc_freq = Counter()
total_possible_outcomes = len(df)
size = len(df_test)


#x = math.ceil(total_possible_outcomes/10)
x = total_possible_outcomes
for i in range(x):
    wrd = word_tokenize(df['text'][i])
    #wrd = [w for w in word_tokens if w not in stop_words]
    #wrds = list(bigrams(wrd))
    wrds = list(bigrams(wrd))
    term_freq[df['stars'][i]].update(wrds)
    doc_freq.update(set(wrds))


wrd_cnt_tot = {cls: sum(ctr.values()) for cls, ctr in term_freq.items()}


vocab = set()
for ctr in term_freq.values():
    vocab |= set(ctr.keys())

vocab_size = len(vocab)


def predict_stars(text):
    p = count.keys()
    prob_all_class = np.zeros(len(p))
    for i in p:
        prior = count[i]/total_possible_outcomes
        total_words_in_class = wrd_cnt_tot[i]
        prob_all_class[int(i)-1] = math.log(prior)
        for j in list(bigrams(word_tokenize(text))):
            temp1 = term_freq[i].get(j, 0)
            temp2 = (temp1 + 1)/(total_words_in_class + vocab_size)
            prob_all_class[int(i)-1] = prob_all_class[int(i)-1] + math.log(temp2)
    return np.argmax(prob_all_class) + 1


size = len(df_test)
prediction = np.zeros(size)
correct_count = 0
print("Accuracy on test set using bi-grams: ")
for i in range(size):
    prediction[i] = predict_stars(df_test['text'][i])
    if prediction[i] == df_test['stars'][i]:
        correct_count = correct_count + 1
print((correct_count/size) * 100)


print("F-score: ")
f_score = f1_score(df_test['stars'], prediction, average=None, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
print(f_score)

f_score_avg = f1_score(df_test['stars'], prediction, labels=[1.0, 2.0, 3.0, 4.0, 5.0], average='macro')
print(f_score_avg)

if mode == 'e':
    print("")
    print("------------------------------------------------------------------------------------")
    print("")

    # prior probabilities
    count = Counter(df['stars'])
    term_freq = defaultdict(Counter)
    doc_freq = Counter()
    total_possible_outcomes = len(df)
    size = len(df_test)

    for r, c in zip(df['text'], df['stars']):
        wrds = r.split()
        term_freq[c].update(wrds)
        doc_freq.update(set(wrds))


    # Convert raw term frequencies into TF-IDF scores
    total_docs = total_possible_outcomes
    for cls, ctr in term_freq.items():
        for wrd in ctr:
            ctr[wrd] = ctr[wrd] * math.log(total_docs / doc_freq[wrd])


    def predict_stars_tf(text):
        p = count.keys()
        prob_all_class = np.zeros(len(p))
        for i in p:
            prior = count[i]/total_possible_outcomes
            total_words_in_class = wrd_cnt_tot[i]
            prob_all_class[int(i)-1] = math.log(prior)
            for j in text.split():
                temp1 = term_freq[i].get(j, 0)
                temp2 = (temp1 + 1)/(total_words_in_class + vocab_size)
                prob_all_class[int(i)-1] = prob_all_class[int(i) - 1] + math.log(temp2)
        return np.argmax(prob_all_class) + 1


    prediction = np.zeros(size)
    correct_count = 0
    print("Accuracy on test set using TF-IDF: ")
    for i in range(size):
        prediction[i] = predict_stars_tf(df_test['text'][i])
        if prediction[i] == df_test['stars'][i]:
            correct_count = correct_count + 1
    print((correct_count/size) * 100)

    print("F-score: ")
    f_score = f1_score(df_test['stars'], prediction, average=None, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
    print(f_score)

    f_score_avg = f1_score(df_test['stars'], prediction, labels=[1.0, 2.0, 3.0, 4.0, 5.0], average='macro')
    print(f_score_avg)
