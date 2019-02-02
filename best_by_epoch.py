from __future__ import print_function
import os
import yaml
import sys
import re
import glob
import pandas as pd
import subprocess
import pickle
import rouge 
import bleu
global config

from multiprocessing import Pool

config_file = sys.argv[1]

def get_number_(s):
	m = re.search("\d+(.)\d+",s)
	return m.group(0)

def get_result_dir_name(dataset,epoch_num):

	dirname = os.path.join(config['log_root'], config['exp_name'],"decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, config['max_enc_steps'], 1, config['min_dec_steps'], config['max_dec_steps']))
	ckpt_name = "ckpt-" + str(epoch_num)
	dirname += "_%s" % ckpt_name

	return dirname

def get_metrics(f1,f2):
	ref = []
	decoded = []
	count = 0
	for i, j in zip(sorted(glob.glob(f1)),sorted(glob.glob(f2))):
		ref_tex = ''
		dec_tex = ''
		for k in open(i).readlines():
				dec_tex = dec_tex + k.strip()
		for l in open(j).readlines():
				ref_tex = ref_tex + l.strip()
		ref.append(ref_tex)
		decoded.append(dec_tex)
		count = count + 1
	
	x = rouge.rouge(decoded,ref)

	bl = bleu.moses_multi_bleu(decoded,ref) #replace by pycoco bleu

	return x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100, bl



def run_test(c):
	os.system('python run_summarization.py --mode=decode --use_val_as_test=True --test_by_epoch=True --epoch_num ' + str(c) + ' --config_file ' + str(config_file))

def multi_test(epoch):
	pool = Pool(processes=5)
	res1 = pool.apply_async(run_test,[epoch[0]])
	res2 = pool.apply_async(run_test,[epoch[1]])
	res3 = pool.apply_async(run_test,[epoch[2]])
	res4 = pool.apply_async(run_test,[epoch[3]])
	res5 = pool.apply_async(run_test,[epoch[4]])
	pool.close()
	pool.join()


with open(config_file) as f:
	config = yaml.load(f)

train_path = os.path.join(config['log_root'], config['exp_name'], "train")


file_list = glob.glob(train_path+'/*.index')
print(file_list)
count = []
for i in sorted(file_list):
	m = re.search('\d+(.)index',i)
	s = m.group(0)
	m = re.search('\d+',s)
	count.append(int(m.group(0)))

k = len(file_list)
p = 10
#print(count)
for i in range(15,k,5):
	multi_test(count[p:i])
	p = i

best_bleu = 0.0
rouge_1 = []
rouge_2 = []
rouge_l = []
bleu_4 = []
best_rouge_epoch = 1000
#print(file_list)
#print(count)

for k,c in enumerate(count): 
	w =  get_result_dir_name('val',c)
	r1,r2,rl,bl = get_metrics(os.path.join(w,'decoded','*.txt'), os.path.join(w,'reference','*.txt'))
	rouge_1.append(r1)
	rouge_2.append(r2)
	rouge_l.append(rl)
	bleu_4.append(bl)
	if bl > best_bleu:
		best_bleu = bl 
		best_rouge_epoch = c

	d = {'epoch':count[:k+1],'rouge_1':rouge_1, 'rouge_2':rouge_2, 'rouge_l': rouge_l,'bleu_4':bleu_4}
	df = pd.DataFrame(d)
	df.to_csv(config['exp_name']+'.csv',index=False)
	print(k)

print(best_rouge_epoch)

os.system('python run_summarization.py --mode=decode --test_by_epoch=True --epoch_num ' + str(best_rouge_epoch) + ' --config_file ' + str(config_file))


'''
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


def get_result_dir_name(dataset,epoch_num):

	dirname = os.path.join(config['log_root'], config['exp_name'],"decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, config['max_enc_steps'], config['beam_size'], config['min_dec_steps'], config['max_dec_steps']))
	ckpt_name = "ckpt-" + str(epoch_num)
	dirname += "_%s" % ckpt_name

	return dirname


def get_metrics(f1,f2):
		ref = []
		decoded = []
		count = 0
		for i, j in zip(sorted(glob.glob(f1)),sorted(glob.glob(f2))):
				ref_tex = ''
				dec_tex = ''
				for k in open(i).readlines():
						dec_tex = dec_tex + k.strip()
				for l in open(j).readlines():
						ref_tex = ref_tex + l.strip()
				ref.append(ref_tex)
				decoded.append(dec_tex)
				count = count + 1
		x = rouge.rouge(decoded,ref)
	bl = bleu.moses_multi_bleu(decoded,ref)

		return x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100, bl


with open(config_file) as f:
	config = yaml.load(f)

other_f = open(config['exp_name']+'.txt', 'w')
#copy graph 
train_path = os.path.join(config['log_root'], config['exp_name'], "train")
#epoch_path = os.path.join(config['log_root'], config['exp_name'], "epoch")

train_projector = os.path.join(train_path, 'projector_config.pbtxt')
train_graph	= os.path.join(train_path, 'graph.pbtxt')

#epoch_projector = os.path.join(epoch_path, 'projector_config.pbtxt')
#epoch_graph	= os.path.join(epoch_path, 'graph.pbtxt')

#os.system('cp ' + train_graph + ' ' + epoch_graph)
#os.system('cp ' + train_projector + ' ' + epoch_projector)

file_list = glob.glob(train_path+'/*.index')
count = []
for i in sorted(file_list):
	m = re.search('\d+(.)index',i)
		s = m.group(0)
		m = re.search('\d+',s)
	count.append(int(m.group(0)))
best_bleu = 0.0
rouge_1 = []
rouge_2 = []
rouge_l = []
bleu_4 = []
best_rouge_epoch = 1000
print(file_list)
print(count)

for k,c in enumerate(count): 
	print(c)
	p = os.system('python run_summarization.py --mode=decode --use_val_as_test=True --test_by_epoch=True --epoch_num ' + str(c) + ' --config_file ' + str(config_file))
		w =  get_result_dir_name('val',c)
	r1,r2,rl,bl = get_metrics(os.path.join(w,'decoded','*.txt'), os.path.join(w,'reference','*.txt'))
	rouge_1.append(r1)
	rouge_2.append(r2)
	rouge_l.append(rl)
	bleu_4.append(bl)
	if bl > best_bleu:
		best_bleu = bl 
		best_rouge_epoch = c

	d = {'epoch':count[:k+1],'rouge_1':rouge_1, 'rouge_2':rouge_2, 'rouge_l': rouge_l}
	df = pd.DataFrame(d)
	df.to_csv(config['exp_name']+'.csv',index=False)
	print(k)

print(best_rouge_epoch)
best_rouge_epoch = 33
r1,r2,rl = get_rouge(get_result_file_name(config,'test',best_rouge_epoch))

print("%s\t%f\t%f\t%f\t"%(config['exp_name'],r1,r2,rl))
print("%s\t%f\t%f\t%f\t"%(config['exp_name'],r1,r2,rl),file=other_f)

'''

