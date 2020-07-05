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


def dl_wet(fh, wet_url):
    response = requests.get(wet_url.strip(), stream=True)

    for record in ArchiveIterator(response.raw, arc2warc=True):
        if record.rec_type == 'conversion':
            content = record.content_stream().read().decode('utf-8')
            yield content


def get_cc_docs(resume_file=None):
    skip = 0
    if resume_file and os.path.exists(resume_file):
        skip = int(open(resume_file).read().strip())

    wet_urls = list(urls())[skip:]
    for i, wet_url in enumerate(wet_urls):
        if resume_file:
            with open(resume_file, 'w') as fh:
                fh.write(i)

        yield from dl_wet(None, "https://commoncrawl.s3.amazonaws.com/" + wet_url)