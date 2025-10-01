import pandas as pd

train = pd.read_csv("../data/imad/BrushlessMotor/train/attributes_normal_source_train.csv")
test = pd.read_csv("../data/imad/BrushlessMotor/test/attributes_normal_source_test.csv")

print(train.shape, test.shape)
print(train.head())
print (train.isna().sum())
