import configparser
import os

class Config:
    def __init__(self, config_file):
        self.conf = configparser.RawConfigParser()
        self.conf.read(config_file, encoding='utf-8')

        # Read from INI
        self.k = self.conf.getint("Model_Setup", "k")
        self.learning_rate = self.conf.getfloat("Model_Setup", "learning_rate")
        self.epochs = self.conf.getint("Model_Setup", "epochs")
        self.hidden_dim = self.conf.getint("Model_Setup", "hidden_dim")
        self.dropout = self.conf.getfloat("Model_Setup", "dropout")
        
        # 👇 ADD THESE TWO LINES 👇
        self.fdim = self.conf.getint("Model_Setup", "fdim")
        self.radius = self.conf.getint("Model_Setup", "radius")
        
        self.lr = self.conf.getfloat("Model_Setup", "lr")
        self.alpha = self.conf.getfloat("Model_Setup", "alpha")
        self.beta = self.conf.getfloat("Model_Setup", "beta")
        self.gamma = self.conf.getfloat("Model_Setup", "gamma")
        self.nhid1 = self.conf.getint("Model_Setup", "nhid1")
        self.nhid2 = self.conf.getint("Model_Setup", "nhid2")
        self.weight_decay = self.conf.getfloat("Model_Setup", "weight_decay")

        self.data_name = self.conf.get("Data_Setup", "data_name")
        self.data_path = self.conf.get("Data_Setup", "data_path")
        self.save_path = self.conf.get("Data_Setup", "save_path")
        
        self.device = self.conf.get("Train_Setup", "device")
        self.seed = self.conf.getint("Train_Setup", "seed")
        