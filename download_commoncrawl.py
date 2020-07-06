from commoncrawl import get_cc_docs
import fasttext
from train_fasttext import preprocess_for_fasttext

import numpy as np
from tqdm import tqdm
import json
import zstd
import os
import sys
import multiprocessing as mp


OUTPUT_DIR = "cc_data_" + sys.argv[1]
alpha = 9

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(14165)

model = fasttext.load_model('fasttext_filter.bin')

class Archive:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.data = []
        self.i = 0
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            self.i = max(map(lambda x: int(x.split('_')[1].split('.')[0]), os.listdir(out_dir))) + 1
    
    def add_data(self, data):
        self.data.append(data)
    
    def commit(self):
        compression_level = 3
        cdata = zstd.compress(json.dumps(self.data).encode('UTF-8'), compression_level)
        with open(self.out_dir + '/cc_' + str(self.i) + '.json.zst', 'wb') as fh:
            fh.write(cdata)

        self.i += 1
        self.data = []
        

def get_doc_score(doc):
    pred = model.predict(preprocess_for_fasttext(doc))
    document_score = pred[1][0]
    if pred[0][0] == '__label__cc':
        document_score = 1 - document_score
    
    return document_score


# target: about 79x
processed = 0
kept = 0
i = 0
ratio = None
archive = Archive(OUTPUT_DIR)
lastchunknum = 0

resume_file = f'resume_{sys.argv[1]}.dat'

skip = 0
if resume_file and os.path.exists(resume_file):
    skip = int(open(resume_file).read().strip())


def fn(x):
    doc, chunkname = x
    return doc, chunkname, get_doc_score(doc)

def get_cc_and_score(index_loc, pool, dl_pool, skip):
    return pool.imap(fn, get_cc_docs(index_loc, dl_pool, skip))

if __name__ == '__main__':
    pool = mp.Pool(4)
    dl_pool = None
    for doc, chunknum, document_score in get_cc_and_score(sys.argv[1], pool, dl_pool, skip):
        if lastchunknum > chunknum:
            lastchunknum = chunknum

            if chunknum % 100 == 0:
                if ratio is None: ratio = processed / kept
                ratio = ratio * 0.99 + (processed / kept) * 0.01
                tqdm.write(f'Filter ratio: {ratio:.4f}')
                archive.commit()

                with open('resume.dat', 'w') as fh:
                    fh.write(str(chunknum))
        
        processed += len(doc)

        # from https://arxiv.org/pdf/2005.14165.pdf P.43
        if np.random.pareto(alpha) > 1 - document_score:
            archive.add_data(doc)
            if kept == 0: kept = len(doc)
            kept += len(doc)
        
        i += 1

    archive.commit()