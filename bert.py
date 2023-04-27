import torch
from pytorch_pretrained_bert import BertTokenizer
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # uncased means with no capital letters



def get_bert_vocab():
    with open('bert_vocab.txt', 'w', encoding='UTF-8') as f:
        for token in tokenizer.vocab.keys():
            f.write(token+'\n')




if __name__=='__main__':
    df = pd.read_csv('./data/colca_public/raw/in_domain_train.tsv', delimiter='\t', header=None, names=['sentence_source','label','label_notes','sentence'])
    print(f'Number of training examples') 


