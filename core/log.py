import os
import time
import csv

class text_logger(object):
    def __init__(self, dir, file_name):
        self.dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.file_name = file_name

    def flush(self, text, mode='a+', print_out=False):
        if print_out:
            print(text)
        with open(os.path.join(self.dir, self.file_name+'.txt'), mode) as f:
            f.write(text+'\n')


class csv_logger(object):
    def __init__(self, dir, file_name, first_row=''):
        self.dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.file_name = file_name
        self.add_row(first_row, 'w')


    def add_row(self, row, mode='a+'):
        with open(os.path.join(self.dir, self.file_name+'.csv'), mode) as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(row)