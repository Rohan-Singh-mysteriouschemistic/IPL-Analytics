import pandas as pd

def load_data():
    ipl = pd.read_csv("data/IPL_2008_2022.csv")
    balls = pd.read_csv("data/IPL_Balls.csv")
    return ipl, balls