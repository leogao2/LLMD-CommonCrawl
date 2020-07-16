import requests
import zlib
from tqdm import tqdm
import os
import json
from warcio.archiveiterator import ArchiveIterator
import justext
import ftfy
import multiprocessing as mp
import numpy as np


def preprocess_for_fasttext(x):
    return x.replace('\n', ' ')[:2000].lower()


def get_seg_urls(seg):
    with open('warc_blocks/urls_' + str(seg).rjust(4, '0')) as fh:
        yield from map(lambda x: x.strip(), fh)


def urls(index_path):
    if os.path.exists(index_path + '_warc_urls.txt'):
        with open(index_path + '_warc_urls.txt') as fh:
            yield from fh

        return

    ret = []
    with open(index_path) as ind:
        for url in tqdm(ind):
            response = requests.get(url.strip(), stream=True)
            
            data = zlib.decompress(response.content, zlib.MAX_WBITS|32)
            for warc in data.decode('utf-8').split('\n'):
                ret.append(warc)
                yield warc
    
    with open(index_path + '_warc_urls.txt', 'w') as fh:
        fh.write('\n'.join(ret))


import fasttext
model = fasttext.load_model('fasttext_filter.bin')

def get_doc_score(doc):
    pred = model.predict(preprocess_for_fasttext(doc))
    document_score = pred[1][0]
    if pred[0][0] == '__label__cc':
        document_score = 1 - document_score
    
    return document_score


def process_pipeline(content):
    alpha = 9
    try:
        content = content.decode('utf-8')
        content = '\n\n'.join([para.text for para in justext.justext(content, justext.get_stoplist("English")) if not para.is_boilerplate])
        #content = ftfy.fix_text(content)
        # from https://arxiv.org/pdf/2005.14165.pdf P.43
        document_score = get_doc_score(preprocess_for_fasttext(content))
#        print(document_score)
        if np.random.pareto(alpha) < 1 - document_score:
            ...
        else:
            return

    except:
        #import traceback
        #traceback.print_exc()
        #print(content)
        return
    return content


def get_warc_contents(warc_url):
    response = requests.get(warc_url.strip(), stream=True)
    for record in ArchiveIterator(response.raw, arc2warc=True):
        if record.rec_type == 'response':
            content = record.content_stream().read()
            yield content


def dl_warc(pool, warc_url):
    return filter(lambda x:x, pool.imap(process_pipeline, get_warc_contents(warc_url), chunksize=8))



def get_cc_docs(warc_urls, dl_pool=None, skip=0):
    pool = mp.Pool(8)
    warc_urls = list(map(lambda x: "https://commoncrawl.s3.amazonaws.com/" + x, warc_urls))

    for warc_url in list(warc_urls):
        print(warc_url)
        for doc in dl_warc(pool, warc_url):
            yield doc