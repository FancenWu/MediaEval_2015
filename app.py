import pandas as pd

train_data = pd.read_csv("assignment-comp3222-comp6246-mediaeval2015-dataset/mediaeval-2015-trainingset.txt", sep="	")
test_data = pd.read_csv("assignment-comp3222-comp6246-mediaeval2015-dataset/mediaeval-2015-testset.txt", sep="	")

# train_data.to_csv("train_data.csv", index=None)
print(train_data.info())
