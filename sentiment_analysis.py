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



from wordcloud import WordCloud,STOPWORDS

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if not word.startswith(',')
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                    background_color=color,
                    width=2500,
                    height=2000
                    ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()



def main():

    dataset = pd.read_csv("labelled_text.txt", delimiter="\t")

    print("----Dataset information-----")
    print(dataset.info())
    print("----------------------------")


    #print(dataset.head() )
    # print(dataset['Class'])

    #Recovery
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'\'s', '')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r'.', '')
    dataset['Sentence'] = dataset['Sentence'].str.replace(r',', '')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'does n\'t', 'does not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'is n\'t', 'is not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'were n\'t', 'were not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'are n\'t', 'are not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'had n\'t', 'had not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'have n\'t', 'have not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'would n\'t', 'would not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'ca n\'t', 'can not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'could n\'t', 'could not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'must n\'t', 'must not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'should n\'t', 'should not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'wo n\'t', 'will not')
    # dataset['Sentence'] = dataset['Sentence'].str.replace(r'n\'t', 'not')



    #print(dataset['Sentence'])

    #Split words and turn to lower case, then count words in all sentences.
    words = Counter(" ".join(dataset['Sentence'].str.lower().values.tolist()).split(" ")).items()

    #dict_items to the list
    list_words = list(words)

    #Sort the list according to the second elements.
    sorted_list_words = sorted(list_words, key=lambda l:l[1], reverse=True)

   
    #Defining stop words
    stop = set(stopwords.words('english'))
    #Add some neccessary words.
    stop.add("") #add empty word into stop set.
    stop.add("-") 
    stop.add("i'm") 
    stop.add("i've") 
    #print (stop)


    # #Removing stop words
    # removed_stop_words = []
    # for i in range(len(sorted_list_words)):
    #     if sorted_list_words[i][0] not in stop :
    #         removed_stop_words.append([]) #2nd dimension
    #         length = len(removed_stop_words) #current length of array
    #         removed_stop_words[length-1].append(sorted_list_words[i][0])
    #         removed_stop_words[length-1].append(sorted_list_words[i][1])

    # text_file = open("latex_table_format.txt", "w")
    # #print most used 50 words
    # for i in range(50):
    #     print("{}. word is '{}' : {}" .format(i, removed_stop_words[i][0], removed_stop_words[i][1] ))
    #     text_file.write("\hline\n%s & %d r'\\'\n" %(removed_stop_words[i][0],removed_stop_words[i][1]))
            
    #print("Before deleting stop words size is : {}, After deleting stop words size is: {} " .format(len(sorted_list_words), len(removed_stop_words)))


    # print(dataset['Sentence'])
    # data = data.rename(columns={'Sentence ': 'Sentence'})
    # data = data.rename(columns={'Class ': 'Class'})

    #Class labels
    y_train = dataset['Class']

    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(binary=True)
    cv.fit(dataset.Sentence)
    X = cv.transform(dataset.Sentence)
    #X_test = cv.transform(dataset.Sentence)

    #neg = 0
    #pos = 1
    #dataset_pos = dataset[dataset.Class.isin([pos])] #positive sentences-class
    #dataset_neg = dataset[dataset.Class.isin([neg])] #negative sentences-class
    #print(dataset_pos)
    #print(dataset_neg)

    # print("Positive words")
    # wordcloud_draw(dataset_pos,'white')
    # print("Negative words")
    # wordcloud_draw(dataset_neg)



    #Count words as positive or negative.(TERM FREQUENCY)
    neg_doc_mat = dataset.Sentence[dataset['Class'] == 0]
    neg_document_matrix = cv.transform(dataset.Sentence[dataset['Class'] == 0])

    pos_doc_mat = dataset.Sentence[dataset['Class'] == 1]
    pos_document_matrix = cv.transform(pos_doc_mat)
    
    neg_batches = np.linspace(0,156061,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(neg_document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neg_batches)-2):
            print(neg_batches[i+1],"entries' term frequency calculated")
        i += 1

    pos_batches = np.linspace(0,156061,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(pos_document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(pos_batches)-2):
            print(pos_batches[i+1],"entries' term frequency calculated")
        i += 1



    #Create dataframe
    neg = np.sum(neg_tf,axis=0)
    pos = np.sum(pos_tf,axis=0)
    term_freq_df = pd.DataFrame([neg,pos],columns=cv.get_feature_names()).transpose()
    term_freq_df.head()


    term_freq_df.columns = ['negative','positive']
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
    print (term_freq_df.sort_values(by='total', ascending=False).iloc[:10])


    #Create CountVectorizer for deleting stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer(stop_words='english',max_features=10000)
    cvec.fit(dataset.Sentence)


    #Count words as positive or negative after deleting stop words.(TERM FREQUENCY) 
    neg_document_matrix_nostop = cvec.transform(dataset.Sentence[dataset['Class'] == 0])
    
    neg_batches = np.linspace(0,156061,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(neg_document_matrix_nostop[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        print(neg_batches[i+1],"entries' term frequency calculated")
        i += 1

    
    pos_document_matrix_nostop = cvec.transform(dataset.Sentence[dataset['Class'] == 1])

    pos_batches = np.linspace(0,156061,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(pos_document_matrix_nostop[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        print(pos_batches[i+1],"entries' term frequency calculated")
        i += 1

    #Create dataframe
    neg = np.sum(neg_tf,axis=0)
    pos = np.sum(pos_tf,axis=0)
    term_freq_df2 = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
    term_freq_df2.columns = ['negative', 'positive']
    term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
    print (term_freq_df2.sort_values(by='total', ascending=False).iloc[:10])



    #Plotting most used words(negative)
    y_pos = np.arange(50)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 negative words')
    plt.title('Top 50 tokens in negative sentiments')
    plt.show()

    #Plotting most used words(positive)
    y_pos = np.arange(50)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 positive tokens')
    plt.title('Top 50 tokens in positive sentiments')
    plt.show()


    #Scatter plot matrix
    import seaborn as sns
    plt.figure(figsize=(8,6))
    ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df2)
    plt.ylabel('Positive Frequency')
    plt.xlabel('Negative Frequency')
    plt.title('Negative Frequency vs Positive Frequency')
    plt.show()



main()
