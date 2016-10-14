# -*- coding: utf-8 -*-

'''
python prepare_word2vec_train_dataset.py input_file output_file
'''

import os
import sys
import logging
import multiprocessing
import time
import json

if __name__ == '__main__':
    start_time = time.time()
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    input_file, output_file = sys.argv[1:3]

    output_file_handler = open(output_file, 'w')
    for line in open(input_file, 'r'):
    	new_line = ''
    	words = line.strip().split()
    	for word in words:
    		word = word.strip('[ ')
    		end_index = word.find(']')
    		if end_index >= 0:
    			word = word[0:end_index]
    		word, tag = word.split('/')
    		new_line = new_line + word + ' '
    	output_file_handler.write(new_line.strip() + '\n')
    	output_file_handler.flush()
    output_file_handler.close()

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))