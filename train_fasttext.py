from commoncrawl import get_cc_docs

import os
import tarfile
import codecs
from tqdm import tqdm
import fasttext

def preprocess_for_fasttext(x):
    return x.replace('\n', ' ')


num_owt = 0

def limit(x, num):
    try:
        for _ in range(num):
            yield next(x)
    except StopIteration:
        return

with open('fasttext_training.txt', 'w') as fout:
    print('Processing OWT data')
    for subset in tqdm(os.listdir('openwebtext')):
        tar = tarfile.open('openwebtext/' + subset, encoding='utf-8')
        utf8reader = codecs.getreader('utf-8')
        for name in tar.getmembers():
            fp = utf8reader(tar.extractfile(name))
            contents = fp.read()
            fout.write('__label__owt ' + preprocess_for_fasttext(contents) + '\n')
            num_owt += 1

    print('Processed', num_owt, 'documents from OWT')
    
    print('Processing CC data')
    for doc in tqdm(limit(get_cc_docs(), num_owt), total=num_owt):
        fout.write('__label__cc ' + preprocess_for_fasttext(doc) + '\n')

print('Training fasttext')
model = fasttext.train_supervised(input="fasttext_training.txt")
model.save_model('fasttext_filter.bin')
