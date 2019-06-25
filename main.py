import os
import csv
import telugu_sa, english_sa

from langdetect import detect
from string import punctuation
from collections import Counter

"""
    Variables for tweaking the analysis to run are to be mentioned down here.
    Always place the file to run analysis in the corpus folder.
"""
filename=""
sequence_length=100



class preprocessing_reading:
    '''
        An integrator class for automating the process of data cleaning and passing the data for training or testing for both
        code-mixed telugu and english languages.

        "task" variable in this class defines the task needed to be done by the neural network
            1: prediction
            2: accuracy_check
            0: training
    '''
    def __init__(self, task):
        self.task = task
        #A dictionary list for storing all dictionaries. It contains all the acutal data from the csv file
        self.dictlist=[]
        #List for storing actual reviews in it
        self.reviews=[]
        #List for storing reviews after tokenising it
        self.reviews_int=[]
        #List for storing the labels of each review
        self.labels=[]

        #Labels for sorting telugu and english features and labels
        self.eng_features = []
        self.eng_labels = []
        self.tel_features=[]
        self.tel_labels=[]

    def cleaning(self):
        if self.task==0:
            '''
                Data for training is cleaned here.
                Step 1: Data is converted to lower case
                Step 2: Punctuation are remove from the data    
            '''
            for each_review in self.dictlist:
                try:
                    each_review['emoji_clean'] = each_review['emoji_clean'].lower()
                    temp_string = each_review['emoji_clean']
                except:
                    each_review['emoji_clean'] = str(each_review['emoji_clean'])
                    each_review['emoji_clean'] = each_review['emoji_clean'].lower()
                for each in punctuation:
                    temp_string.replace(each, '')
                each_review['emoji_clean'] = temp_string
                self.reviews.append(temp_string)
            '''
                Converting the words in all the reviews to numbers using counter function for best results. Words 
                appearing more are given smaller index
            '''
            all_text2 = ' '.join(self.reviews)

            #Create a list of words
            words = all_text2.split()

            #Count all the words using counter method
            count_words = Counter(words)

            total_words = len(words)
            sorted_words = count_words.most_common(total_words)

            '''
                Used i+1 below to give keys to the dictionary from 1 as it would be convenient while using nn as 0 is left
                for padding.
                Converted all the reviews into respective number format for easy processing
            '''
            vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
            for review in self.reviews:
                r = [vocab_to_int[w] for w in review.split()]
                self.reviews_int.append(r)

            #Converted sentiment of tweets to numbers, 0 :Neutral, 1:Positive, -1: Negative
            for each_review in self.dictlist:
                if each_review['sentiment']=="Positive":
                    self.labels.append(1)
                elif each_review['sentiment']=="Neutral":
                    self.labels.append(0)
                elif each_review['sentiment']=="Negative":
                    self.labels.append(-1)

            '''
                Reviews are truncated to a particular length to train the neural network of a particular number of nodes.
                Reviews whose length is longer are truncated or completely removed from the dataset.
                Reviews whose length is shorter is padded with 0 as the review integer. This is the reason 0 is excluded
                from word to integer mapping.
            '''
            count=0
            truncate=[]
            for each_review in self.reviews_int:
                if len(each_review)<sequence_length:
                    #Need to adding padding to the comment
                    pad_required = sequence_length-len(each_review)
                    for i in range(pad_required):
                        each_review.append(0)
                elif len(each_review)>sequence_length:
                    #Need to truncate the review
                    truncate.append(count)
                elif len(each_review)==100:
                    pass
                count+=1
            for each in truncate:
                self.reviews_int.pop(each)
                self.labels.pop(each)
                self.reviews.pop(each)

            #Checking the language of each feature andd sorting features and labels accordingly in the specified lists
            count=0
            for review in self.reviews:
                self.lang_detect(review, count)
                count+=1

            #Passing all the parameters of this class to the senti_analysis class for training the neural network
            print(len(self.eng_features))
            kwargs = {"features": self.eng_features, "encoded_labels": self.eng_labels, "vocab_to_int": vocab_to_int}
            english = english_sa.senti_analysis(**kwargs)
            english.training()
            kwargs = {"features": self.tel_features, "encoded_labels": self.tel_labels, "vocab_to_int": vocab_to_int}
            telugu = telugu_sa.senti_analysis(**kwargs)
            telugu.training()

        elif self.task==1:
            #Pass the data for the rnn for prediction
            """
                Data for prediction is cleaned and then passesd on to the neural network
                Step 1: Data is converted to lower case
                Step 2: Punctuation are remove from the data
            """['emoji_clean']
            for each_review in self.dictlist:
                try:
                    each_review['emoji_clean'] = each_review['emoji_clean'].lower()
                    temp_string = each_review['emoji_clean']
                except:
                    each_review['emoji_clean'] = str(each_review['emoji_clean'])
                    each_review['emoji_clean'] = each_review['emoji_clean'].lower()
                for each in punctuation:
                    temp_string.replace(each, '')
                each_review['emoji_clean'] = temp_string
                self.reviews.append(temp_string)

            '''
                Converting the words in all the reviews to numbers using counter function for best results. Words 
                appearing more are given smaller index
            '''
            all_text2 = ' '.join(self.reviews)

            #Create a list of words
            words = all_text2.split()

            #Count all the words using counter method
            count_words = Counter(words)

            total_words = len(words)
            sorted_words = count_words.most_common(total_words)

            '''
                Used i+1 below to give keys to the dictionary from 1 as it would be convenient while using nn as 0 is left
                for padding.
                Converted all the reviews into respective number format for easy processing
            '''
            vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
            # print(vocab_to_int)
            vocabulary_len = len(vocab_to_int)
            # print(vocabulary_len)
            exit(0)
            for review in self.reviews:
                r = [vocab_to_int[w] for w in review.split()]
                self.reviews_int.append(r)

            '''
                Reviews are truncated to a particular length to train the neural network of a particular number of nodes.
                Reviews whose length is longer are truncated or completely removed from the dataset.
                Reviews whose length is shorter is padded with 0 as the review integer. This is the reason 0 is excluded
                from word to integer mapping.
            '''
            count = 0
            truncate = []
            for each_review in self.reviews_int:
                if len(each_review) < 100:
                    # Need to adding padding to the comment
                    pad_required = 100 - len(each_review)
                    for i in range(pad_required):
                        each_review.append(0)
                elif len(each_review) > 100:
                    # Need to truncate the review
                    truncate.append(count)
                elif len(each_review) == 100:
                    pass
                count += 1
            for each in truncate:
                self.reviews_int.pop(each)
                self.reviews.pop(each)

            # Checking the language of each feature andd sorting features and labels accordingly in the specified lists
            count = 0
            for review in self.reviews:
                print()
                self.lang_detect(review, count)
                count+=1

            # Passing all the parameters of this class to the senti_analysis class for training the neural network
            english = english_sa.senti_analysis(self.eng_features, vocab_to_int)
            telugu = telugu_sa.senti_analysis(self.tel_features, vocab_to_int)


    def read(self):
        # file = os.listdir("./corpus/")
        corpus_path = "corpus/118tweets_SA.csv"
        csvreader = csv.DictReader(open(corpus_path, 'r'))
        for eachline in csvreader:
            self.dictlist.append(eachline)
        self.cleaning()

    def lang_detect(self, data, count):
        lang=detect(data)
        if lang=='te':
            self.tel_features.append(self.reviews_int[count])
            self.tel_labels.append(self.labels[count])
        elif lang=='en':
            self.eng_features.append(self.reviews_int[count])
            self.eng_labels.append(self.labels[count])


if __name__=='__main__':
    task = int(input("Enter 0 for training and 1 for testing:"))
    integrator=preprocessing_reading(task)
    integrator.read()
