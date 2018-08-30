from __future__ import print_function
import os
import yaml
import sys
import re
import glob
import pandas as pd
import subprocess
import pickle
config_file = sys.argv[1]

def get_number_(s):
	m = re.search("\d+(.)\d+",s)
	return m.group(0)

def get_rouge(results_path):
	f = open(results_path,'r')
	s = ''
	for i in f.readlines():
		s = s+ i
	w = s.split('\n')
	r1 = float(get_number_(w[2])) * 100
	r2 = float(get_number_(w[7])) * 100
	rl = float(get_number_(w[12])) * 100

	return r1,r2,rl

def get_result_file_name(config,dataset,epoch_num):

	dirname = os.path.join(config['log_root'], config['exp_name'],"decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, config['max_enc_steps'], config['beam_size'], config['min_dec_steps'], config['max_dec_steps']))
	ckpt_name = "epoch-" + str(epoch_num)
	dirname += "_%s" % ckpt_name

	return os.path.join(dirname,'ROUGE_results.txt')




with open(config_file) as f:
	config = yaml.load(f)

other_f = open(config['exp_name']+'.txt', 'w')
#copy graph 
train_path = os.path.join(config['log_root'], config['exp_name'], "train")
epoch_path = os.path.join(config['log_root'], config['exp_name'], "epoch")

train_projector = os.path.join(train_path, 'projector_config.pbtxt')
train_graph	= os.path.join(train_path, 'graph.pbtxt')

epoch_projector = os.path.join(epoch_path, 'projector_config.pbtxt')
epoch_graph	= os.path.join(epoch_path, 'graph.pbtxt')

os.system('cp ' + train_graph + ' ' + epoch_graph)
os.system('cp ' + train_projector + ' ' + epoch_projector)

file_list = glob.glob(epoch_path+'/epoch_*.index')
count = []
for i in file_list:
	m = re.search('\d+(.)index',i)
        s = m.group(0)
        m = re.search('\d+',s)
	count.append(int(m.group(0)))
best_rouge_2 = 0.0
rouge_1 = []
rouge_2 = []
rouge_l = []
best_rouge_epoch = 1000
print(file_list)
print(count)
'''
for k,c in enumerate(count): 
	p = subprocess.call('python run_summarization.py --mode=decode --use_val_as_test=True --test_by_epoch=True --epoch_num ' + str(c) + ' --config_file ' + str(config_file),shell=True)
        print(p)
	r1,r2,rl = get_rouge(get_result_file_name(config,'val',c))
	rouge_1.append(r1)
	rouge_2.append(r2)
	rouge_l.append(rl)
	if r2 > best_rouge_2:
		best_rouge_2 = r2 
		best_rouge_epoch = c

	d = {'epoch':count[:k+1],'rouge_1':rouge_1, 'rouge_2':rouge_2, 'rouge_l': rouge_l}
	df = pd.DataFrame(d)
	df.to_csv(config['exp_name']+'.csv',index=False)
'''
best_rouge_epoch = 33
p = subprocess.call('python run_summarization.py --mode=decode --test_by_epoch=True --epoch_num ' + str(best_rouge_epoch) + ' --config_file ' + str(config_file), shell=True)
r1,r2,rl = get_rouge(get_result_file_name(config,'test',best_rouge_epoch))

print("%s\t%f\t%f\t%f\t"%(config['exp_name'],r1,r2,rl))
print("%s\t%f\t%f\t%f\t"%(config['exp_name'],r1,r2,rl),file=other_f)

