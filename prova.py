from datasets import load_dataset
import pandas as pd

dataset=load_dataset("lmms-lab/OK-VQA")
dataframe=pd.DataFrame(dataset)
dataframe_temp = dataframe.iloc[5036:]
i=1
for index, row in dataframe_temp.iterrows():
    image = "images/img{}.jpg".format(i)
    input_text = row[0]['question']
    i=i+1
    print(image)