import pandas as pd
import re

data = pd.read_json('dev.json')

end_token = ['.', '!', '?' ]
# tmp = data.iloc[0]


documents = []
for doc, asp, gold in zip(data['document'], data['aspect'], data['summary']):
    document = []
    sentence = []
    
    for word in doc.split(' '):
        if word in end_token:
            sentence = sentence + [word]
            document.append(sentence)
            sentence = []
        else:
            sentence = sentence + [word]
    
    num_lines = len(document)
    label = [0] * num_lines
    
    for num_line, line in enumerate(document):
        cnt = 0
        for lab_word in gold.split(' '):
            if len(lab_word) >= 2:
                if lab_word in line:
                    cnt += 1
        if cnt >= 1:
            label[num_line] = 1
        
        gold 

    documents.append({'document':doc, 'aspect':asp, 'summary':gold, 'label':label})


df = pd.DataFrame(documents)
print(df['label'][0])
print(df['summary'][0])
print(df['document'][0])
# df.to_json()