import numpy as np
import pandas as pd

import unicodedata, re, string
#import nltk
#nltk.download() #use this for to download nltk (popular packages)

# import seaborn as sns
# sns.set(color_codes=True)


# import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# import unicodedata, re, string

from nltk.corpus import stopwords


from collections import Counter

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from scipy.misc import comb
from itertools import combinations
from sklearn.metrics import silhouette_samples, silhouette_score

import statistics 

#k-means model fit and results
def kmeans(X, vectorizer, k):

    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1) #algorithm
    model.fit(X) #training

    #print clusters
    # print("Top terms per cluster:")
    # order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    # for i in range(k):
    #     print("---------")
    #     print("Cluster %d:" % i)
    #     for ind in order_centroids[i, :20]:
    #         print(' %s' % terms[ind])

    return model



def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def main():

    dataset = pd.read_csv("labelled_text.txt", delimiter="\t")

    print("----Dataset information-----")
    print(dataset.info())
    print("----------------------------")

    #print(dataset.head() )
    # print(dataset['Class'])

    #Lowercase
    dataset= dataset.applymap(lambda s:s.lower() if type(s) == str else s)

    #Recovery
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'\'s', '')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'.', '')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r',', '')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'i\'ve', 'i have')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'i\'m', 'i am')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'didn\'t', 'did not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'don\'t', 'do not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'does n\'t', 'does not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'is n\'t', 'is not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'were n\'t', 'were not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'are n\'t', 'are not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'had n\'t', 'had not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'have n\'t', 'have not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'would n\'t', 'would not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'ca n\'t', 'can not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'could n\'t', 'could not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'must n\'t', 'must not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'should n\'t', 'should not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'wo n\'t', 'will not')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'n\'t', 'not')

    

    #Defining stop words
    stop = set(stopwords.words('english'))
    #Add some neccessary words.
    stop.add("") #add empty word into stop set.
    stop.add("-")
    stop.add("would")
    stop.add("could")    


    #Create CountVectorizer for deleting stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer(stop_words=stop, max_features=10000)
    cvec.fit(dataset.Sentence)


    #Count words as positive or negative after deleting stop words.(TERM FREQUENCY) 
    #Get negative sentences -> 0
    neg_document_matrix_nostop = cvec.transform(dataset.Sentence[dataset['Class'] == 0])
    
    #Get negative words into array.
    neg_batches = np.linspace(0,156061,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(neg_document_matrix_nostop[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        #print(neg_batches[i+1],"entries' term frequency calculated")
        i += 1


    #Get positive sentences-> 1
    pos_document_matrix_nostop = cvec.transform(dataset.Sentence[dataset['Class'] == 1])

    #Get positive words into array.
    pos_batches = np.linspace(0,156061,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(pos_document_matrix_nostop[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        #print(pos_batches[i+1],"entries' term frequency calculated")
        i += 1


    #Create dataframe
    #Columns from neg_tf and pos_tf array
    neg = np.sum(neg_tf,axis=0)
    pos = np.sum(pos_tf,axis=0)
    #Creates pandas dataframe
    term_freq_data = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
    #Write column names
    term_freq_data.columns = ['negative', 'positive']
    #Create 'total' column and write its value.
    term_freq_data['total'] = term_freq_data['negative'] + term_freq_data['positive']
    #Print term frequencies dataframe by order. (Top 10)
    print (term_freq_data.sort_values(by='total', ascending=False).iloc[:30])



    #Plotting most used words(negative)
    y_pos = np.arange(30)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, term_freq_data.sort_values(by='negative', ascending=False)['negative'][:30], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_data.sort_values(by='negative', ascending=False)['negative'][:30].index,rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 30 negative words')
    plt.title('Top 30 tokens in negative sentiments')
    plt.show()

    #Plotting most used words(positive)
    y_pos = np.arange(30)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, term_freq_data.sort_values(by='positive', ascending=False)['positive'][:30], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_data.sort_values(by='positive', ascending=False)['positive'][:30].index,rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 30 positive tokens')
    plt.title('Top 30 tokens in positive sentiments')
    plt.show()


    #Scatter plot matrix
    import seaborn as sns
    plt.figure(figsize=(8,6))
    ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_data)
    plt.ylabel('Positive Frequency')
    plt.xlabel('Negative Frequency')
    plt.title('Negative Frequency vs Positive Frequency')
    plt.show()


    #term frequency - inverse document frequency calculating
    vectorizer = TfidfVectorizer(stop_words = stop)
    X = vectorizer.fit_transform(dataset.Sentence)
    Y = dataset['Class']

    true_k = [2,3,4,5] #cluster numbers
    for i in range(len(true_k)):
        print("K Value is:", true_k[i])
        model = kmeans(X, vectorizer, true_k[i])
        total = 0
        for j in range(true_k[i]):
            print("%d. cluster size: %d " %(j, list(model.labels_).count(j)))
            total = list(model.labels_).count(j) + total

        print("Overall average in clusters:", total/true_k[i])
        #convert to a list
        predicted_Y = list(model.labels_)
        print("STD:", np.std(predicted_Y))
        #calculate sse (true_y - predicted_y)**2
        squared_errors = (Y - predicted_Y)**2
        sum_of_squared_errors = sum(squared_errors)
        print("SSE:", sum_of_squared_errors)
        nmi = normalized_mutual_info_score(Y, predicted_Y )
        print("NMI: ",nmi)
        print("Silhouette Value:",silhouette_score(X, predicted_Y))
        print("RI:", rand_index_score (Y, predicted_Y))
        
        
        # Here we get the proportions
        nb_samples = [sum(model.labels_ == m) for m in range(true_k[i])]

        # On the next line the order is RANDOM. I do NOT know which cluster represents what.
        # The first label should represent samples in cluster 0, and so on
        if true_k[i] == 2:
            labels = 0 , 1
            colors = [ 'green', 'red']  # Same size as labels
        elif true_k[i] == 3:
            labels = 0, 1 , 2
            colors = [ 'green', 'red', 'lightblue']
        elif true_k[i] == 4:
            labels = 0, 1 , 2, 3
            colors = [ 'green', 'red', 'lightblue', 'grey']
        elif true_k[i] == 5:
            labels = 0, 1 , 2, 3, 4
            colors = [ 'green', 'red', 'lightblue', 'grey', 'pink']

        # Pie chart
        plt.pie(nb_samples, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.axis('equal')
        plt.show()




main()
