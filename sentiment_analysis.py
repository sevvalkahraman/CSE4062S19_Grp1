import numpy as np
import pandas as pd

import unicodedata, re, string
#import nltk

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



main()
