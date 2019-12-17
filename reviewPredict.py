import json
import pandas as pd
import math
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import sys

# from sklearn.utils.multiclass import unique_labels
# nltk.download('punkt')

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
c = 5

# dropping unwanted attributes
df.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)
df_test.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

# prior probabilities
count = Counter(df['stars'])
total_possible_outcomes = len(df)
term_freq = defaultdict(Counter)


for r, c in zip(df['text'], df['stars']):
        wrds = r.split()
        term_freq[c].update(wrds)


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
        for j in text.split():
            temp1 = term_freq[i].get(j, 0)
            temp2 = (temp1 + 1)/(total_words_in_class + vocab_size)
            prob_all_class[int(i)-1] = prob_all_class[int(i) - 1] + math.log(temp2)
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


# -- if( $3 == (a)) then only training set accuracy is required
# print("Accuracy on train data: ")
# correct_count = 0
# for i in range(total_possible_outcomes):
#     prediction_val = predict_stars(df['text'][i])
#     if prediction_val == df['stars'][i]:
#         correct_count = correct_count + 1
# print((correct_count/total_possible_outcomes) * 100)


def plot_confusion_matrix(y_actual, y_predicted,  classes, title,  cmap=plt.cm.Blues):
    cm = confusion_matrix(y_actual, y_predicted)
    # classes = classes[unique_labels(y_actual, y_predicted)]
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


class_names = [1.0, 2.0, 3.0, 4.0, 5.0]
if mode == 'c':
    plot_confusion_matrix(df_test['stars'], prediction, classes=class_names, title='Confusion matrix')
    plt.show()

print("F-score: ")
f_score = f1_score(df_test['stars'], prediction, average=None, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
print(f_score)

f_score_avg = f1_score(df_test['stars'], prediction, labels=[1.0, 2.0, 3.0, 4.0, 5.0], average='macro')
print(f_score_avg)
