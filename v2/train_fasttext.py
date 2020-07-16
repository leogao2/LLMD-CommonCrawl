from commoncrawl import get_cc_docs, get_seg_urls, preprocess_for_fasttext

import os
import tarfile
import codecs
from tqdm import tqdm
import fasttext
from lm_dataformat import Reader


def limit(x, num):
    try:
        for _ in range(num):
            yield next(x)
    except StopIteration:
        return


doc_ct = 100000

if __name__ == '__main__':
    num_owt = 0

    with open('fasttext_training.txt', 'w') as fout:
        print('Processing OWT data')
        wt_rdr = Reader('../openwebtext')
        for doc in tqdm(limit(wt_rdr.stream_data(), doc_ct)):
            fout.write('__label__owt ' + preprocess_for_fasttext(doc) + '\n')
            num_owt += 1

        print('Processed', num_owt, 'documents from OWT')
        
        print('Processing CC data')
        for doc in tqdm(limit(get_cc_docs(get_seg_urls(0)), doc_ct), total=doc_ct):
            fout.write('__label__cc ' + preprocess_for_fasttext(doc) + '\n')

    print('Training fasttext')
    model = fasttext.train_supervised(input="fasttext_training.txt")
    model.save_model('fasttext_filter.bin')
