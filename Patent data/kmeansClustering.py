#Author Priyanka
# Find natural segments(clusters) in patent data
from __future__ import print_function
from time import time
import lda
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.plotting import show
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

n_samples = 100
n_features = 1000
n_topics = 10
n_top_words = 10
summary = []
t0=time()

with open ('uspto.json','rb') as f:
    d=[json.loads(line) for line in f ]

    #summary.append()
for i in range(len(d)):
    summary.append((d[i]['object']['publicationDate']))

data_samples = summary
print("done in %0.3fs." % (time() - t0))

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print (tfidf.shape)
# Truncated SVD to reduce dimension of vector
svd = TruncatedSVD(n_components=50, random_state=0)
svd_tfidf = svd.fit_transform(tfidf[:50])
print(svd_tfidf.shape)

#TSNE to further reduce dimensin from 50 to 2
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)

#Kmeans Clustering
num_clusters = 6
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
                         init_size=10, batch_size=1000, verbose=False, max_iter=100)
kmeans = kmeans_model.fit(tfidf)
kmeans_clusters = kmeans.predict(tfidf)
kmeans_distances = kmeans.transform(tfidf)

sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i, end='')
    for j in sorted_centroids[i, :10]:
        print(' %s' % terms[j], end='')
    print()

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))


print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
model=lda.LDA(n_topics=n_topics, n_iter=100, random_state=1)
model.fit(tf)

topic_word = model.topic_word_

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(model, tf_feature_names, n_top_words)

colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
])

plot_kmeans = bp.figure(plot_width=900, plot_height=700, title="Patent data k-means clustering on Publication date",
                        x_axis_type=None, y_axis_type=None, min_border=1)

tsne_kmeans = tsne_model.fit_transform(kmeans_distances[:1000])

plot_kmeans.scatter(x=tsne_kmeans[:, 0], y=tsne_kmeans[:, 1],
                    color=colormap[kmeans_clusters][:1000],
                    source=bp.ColumnDataSource({"cluster": kmeans_clusters[:1000]
}))

hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"cluster": "@cluster"}
show(plot_kmeans)


#silhouette  analysis of clusters
from sklearn.cluster import KMeans
range_n_clusters = [2, 4, 6, 10, 15 ]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, 10 + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(tfidf)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(tfidf, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

