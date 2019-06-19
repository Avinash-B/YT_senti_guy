import os,sys
import torch

from torch.utils.data import DataLoader, TensorDataset
from langdetect import detect
import telugu_sa, english_sa


class preprocessing_reading:
    def __init__(self):
        self.read()
        pass
    def cleaning(self):

        #This line need to be deleted at last, just added for syntax warning supression
        cleaned_data=""
        self.lang_detect(self,data=cleaned_data)
        pass
    def read(self):
        file = os.listdir("./corpus/")
        corpus_path = "corpus/" + file[0]
        with open(corpus_path) as csvinputfile:
            for eachline in csvinputfile:
                self.cleaning(eachline)
        self.cleaning()
        pass
    def lang_detect(self, data):
        lang=detect("")
        if lang=='te':
            telugu_sa.senti_analysis()
        elif lang=='en':
            english_sa.senti_analysis()


if __name__=='__main__':
    integrator=preprocessing_reading()
