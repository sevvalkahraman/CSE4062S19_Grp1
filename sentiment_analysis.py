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

def main():

    dataset = pd.read_csv("imdb_labelled_next.txt", delimiter="\t")

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
    #print (stop)


    #Removing stop words
    removed_stop_words = []
    for i in range(len(sorted_list_words)):
        if sorted_list_words[i][0] not in stop :
            removed_stop_words.append([]) #2nd dimension
            length = len(removed_stop_words) #current length of array
            removed_stop_words[length-1].append(sorted_list_words[i][0])
            removed_stop_words[length-1].append(sorted_list_words[i][1])

    #print most used 10 words
    for i in range(10):
        print("{}. word is '{}' : {}" .format(i, removed_stop_words[i][0], removed_stop_words[i][1] ))

            
    print("Before deleting stop words size is : {}, After deleting stop words size is: {} " .format(len(sorted_list_words), len(removed_stop_words)))

main()
