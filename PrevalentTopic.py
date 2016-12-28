#Author -Priyanka Rao
# 1. Find top 10 prevalent topic of interest in patent data
from __future__ import print_function
from time import time
import lda
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt

n_samples = 100
n_features = 1000
n_topics = 10
n_top_words = 10
summary = []
titles=[]
t0=time()
#open input file and copy each document into d
with open ('uspto.json','rb') as f:
    d=[json.loads(line) for line in f ]

# copy summary from  patent data into dictionary summary
for i in range(len(d)):
    summary.append((d[i]['object']['summary']))
    titles.append((d[i]['object']['title']))

data_samples = summary
# print top 10 topics and and 10 words in that dictionary
print("done in %0.3fs." % (time() - t0))
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# creating TF-IDF scores that indicate how common or rare a word in a document is, with respect to the entire corpus
print("Extracting tf-idf features ")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
#print(dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_)))
print("shape: {}".format(tfidf.shape))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print("shape: {}".format(tf.shape))


print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
model=lda.LDA(n_topics=n_topics, n_iter=200, random_state=1)
model.fit(tf)


print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(model, tf_feature_names, n_top_words)

topic_word = model.topic_word_
doc_topic = model.doc_topic_

print("type(doc_topic): {}".format(topic_word.shape))
print("shape: {}".format(doc_topic.shape))

for n in range(10):
    sum_pr = sum(doc_topic[n,:])
    print("document: {} sum: {}".format(n, sum_pr))


for n in range(10):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}\n{}...".format(n,
                                            topic_most_pr,
                                            titles[n][:50]))
#Visualization
f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([0, 3, 5, 6, 9]):
    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(-50,1100)
    ax[i].set_ylim(0, 0.08)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("topic {}".format(k))

ax[4].set_xlabel("word")

plt.tight_layout()
plt.show()
#plt.close()

f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([1, 3, 4, 8, 9]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 11)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))

ax[4].set_xlabel("Topic")

plt.tight_layout()
plt.show()
