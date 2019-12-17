from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import pandas as pd
import math
import sys
from nltk import word_tokenize
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import f1_score

# nltk.download('stopwords')

file = sys.argv[1]
file_test = sys.argv[2]

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
total_possible_outcomes = len(df)
size = len(df_test)


def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)


def getStemmedDocuments(docs, return_tokens=True):
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, p_stemmer, en_stop, return_tokens)


#x = math.ceil(total_possible_outcomes/10)
x = total_possible_outcomes
for i in range(x):
    result = getStemmedDocuments(df['text'][i])
    term_freq[df['stars'][i]].update(result)


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
        for j in getStemmedDocuments(text):
            temp1 = term_freq[i].get(j, 0)
            temp2 = (temp1 + 1)/(total_words_in_class + vocab_size)
            prob_all_class[int(i)-1] = prob_all_class[int(i)-1] + math.log(temp2)
    return np.argmax(prob_all_class) + 1


size = len(df_test)
prediction = np.zeros(size)
correct_count = 0
print("Accuracy on test data: ")
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
