from commoncrawl import get_cc_docs, get_seg_urls, preprocess_for_fasttext
import fasttext

import numpy as np
from tqdm import tqdm
import json
import zstd
import os
import sys
import multiprocessing as mp
from lm_dataformat import Archive
import pybloomfilter
import collections


class DownloadSchedule:
    def __init__(self, fpath):
        self.schedule = []
        # segments are INCLUSIVE!
        self.fpath = fpath
        with open(fpath) as fh:
            for line in fh:
                segs = list(map(int, line.split('-')))
                if len(segs) == 1:
                    segs = [segs[0], segs[0]]
                assert len(segs) == 2

                self.schedule.append(segs)
    
    def segments(self):
        for seg in self.schedule:
            for i in range(seg[0], seg[1] + 1):
                yield i
    
    def mark_as_done(self, num):
        for i, (start, end) in enumerate(self.schedule):
            if start == end and start == num:
                del self.schedule[i]
            elif start == num:
                self.schedule[i][0] += 1
            elif end == num:
                self.schedule[i][0] -= 1
            else:
                raise Exception('Not yet implemented!')
        
        with open(self.fpath, 'w') as fh:
            for start, end in self.schedule:
                if start == end:
                    fh.write(str(start) + '\n')
                else:
                    fh.write(str(start) + '-' + str(end) + '\n')


class RollingBloomFilter:
    def __init__(self, each_capacity=1000000, each_error_rate=0.0001, num_filters=10):
        self.filters = collections.deque()
        self.each_capacity = each_capacity
        self.each_error_rate = each_error_rate
        self.num_filters = num_filters
        self.seen = 0
    
    def roll(self):
        if len(self.filters) == self.num_filters:
            self.filters.pop()
        self.filters.appendleft(pybloomfilter.BloomFilter(self.each_capacity, self.each_error_rate))

    def add(self, x):
        if self.seen % self.each_capacity == 0:
            self.roll()

        self.filters[0].add(x)

        self.seen += 1
    
    def __contains__(self, item):
        for filter in self.filters:
            if item in filter:
                return True
        return False


sched = DownloadSchedule('download_schedule.txt')
ar = Archive('data_output')
fh = open('cc_warc_justext_multistop.txt', 'w')
for seg in sched.segments():
    print(seg)
    urls = list(get_seg_urls(seg))
    for doc in get_cc_docs(urls):
        #print(doc)
        if doc.strip():
            ar.add_data(doc.strip())
            fh.write(doc + "\n========================================================================\n")
    ar.commit(archive_name="chunk_" + str(seg))