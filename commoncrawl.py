import requests
import zlib
from tqdm import tqdm
import os
import json
from warcio.archiveiterator import ArchiveIterator

def urls():
    if os.path.exists('wet_urls.json'):
        with open('wet_urls.json') as fh:
            yield from json.load(fh)

        return

    ret = []
    with open('indexes.txt') as ind:
        for url in ind:
            response = requests.get(url.strip(), stream=True)
            
            data = zlib.decompress(response.content, zlib.MAX_WBITS|32)
            for wet in data.decode('utf-8').split('\n'):
                ret.append(wet)
                yield wet
    
    with open('wet_urls.json', 'w') as fh:
        json.dump(ret, fh)


def dl_wet(wet_url):
    response = requests.get(wet_url.strip(), stream=True)

    for record in ArchiveIterator(response.raw, arc2warc=True):
        if record.rec_type == 'conversion':
            content = record.content_stream().read().decode('utf-8')
            yield content


def dl_wet_list(wet_url):
    response = requests.get(wet_url.strip(), stream=True)
    ret = []
    for record in ArchiveIterator(response.raw, arc2warc=True):
        if record.rec_type == 'conversion':
            content = record.content_stream().read().decode('utf-8')
            ret.append(content)
    return ret


def get_cc_docs(dl_pool=None, skip=0):
    wet_urls = list(urls())[skip:]
    wet_urls = list(map(lambda x: "https://commoncrawl.s3.amazonaws.com/" + x, wet_urls))

    if dl_pool is not None:
        for i, docs in enumerate(tqdm(list(dl_pool.imap(dl_wet_list, wet_urls)))):
            for doc in docs: yield (doc, i)
    else:
        for i, wet_url in enumerate(tqdm(list(wet_urls))):
            for doc in dl_wet(wet_url):
                yield (doc, i)