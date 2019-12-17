# TextClassification
## Abstract: 
Given a user review(in text), predict the stars(1-5) given by the reviewer.

[Yelp Dataset](https://www.yelp.com/dataset)
The dataset is available as JSON files, where each json object is a collection different attributes like user_review, review_id, user, user_id, stars etc. <br />

Implemented **Naive Bayes** algorithm from scratch to build the classifier. Plotted confusion matrix to get a better idea of results of this classifier.<br />
**Accuracy** over test data: 61.87% <br />

Since the accuracy wasn't great, I tried a bunch of different things along with pure Naive Bayes like Stemming, Stopword removal. <br />
**Accuracy** after above transformations: 60.058% <br />

Next, I tried some **feature engineering** like constructing *bi-grams, tri-grams, TF-IDF* etc. but no major improvement in accuracy was seen.
