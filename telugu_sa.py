import main
import torch

from torch.utils.data import DataLoader, TensorDataset


class senti_analysis(main.preprocessing_reading):
    def director(self,task):
        if task==0:
            self.training()
        elif task==1:
            self.testing()
        elif task==2:
            self.accuracy_check()
    def training(self):
        pass
    def testing(self):
        pass
    def accuracy_check(self):
        pass