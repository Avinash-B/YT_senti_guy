import os,sys
import csv
import telugu_sa, english_sa

from langdetect import detect
from string import punctuation
from collections import Counter


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
        self.dictlist=[]
        self.read()
        self.reviews=[]
        self.labels=[]
        pass
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
                for padding
            '''
            vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
        elif self.task==1:
            #Pass the data for the rnn for prediction
        elif self.task==2:
            #Pass the data for the rnn for accuracy check
        #This line need to be deleted at last, just added for syntax warning supression
        cleaned_data=""
        self.lang_detect(self,data=cleaned_data)
        pass
    def read(self):
        file = os.listdir("./corpus/")
        corpus_path = "corpus/" + file[0]
        csvreader = csv.DictReader(open(corpus_path, 'rb'))
        for eachline in csvreader:
            self.dictlist.append(eachline)
        self.cleaning()
        pass
    def lang_detect(self, data):
        lang=detect("")
        if lang=='te':
            telugu_sa.senti_analysis()
        elif lang=='en':
            english_sa.senti_analysis()


if __name__=='__main__':
    task = input("Enter 0 for training and 1 for testing:")
    integrator=preprocessing_reading(task)
