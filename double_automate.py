import os
import sys
import yaml
import subprocess

from multiprocessing import Pool

file_name_1 = sys.argv[1]
file_name_2 = sys.argv[2]
#gpu_val_id = sys.argv[3]
def run_train(file_name,out_file_name):
	train_command = 'python run_summarization.py --mode=train --config_file ' + str(file_name)
	os.system(train_command)
#	with open(out_file_name, "w") as outfile:
#		subprocess.call(train_command, stdout=outfile)
	
def run_eval(file_name,out_file_name):
	eval_command = 'python  run_summarization.py --mode=eval --config_file ' + str(file_name) 
	eval_command = 'sleep 100; ' + eval_command
	os.system(eval_command)
	#with open(out_file_name, "w") as outfile:
	#	subprocess.call(eval_command, stdout=outfile)

def run_test(file_name,out_file_name):
	test_command = 'python -u run_summarization.py --mode=decode --config_file '+str(file_name) 
	#with open(out_file_name, "w") as outfile:
	#	subprocess.call(test_command, stdout=outfile)
	os.system(test_command)

def run_eval_test(file_name,out_file_name):
	test_command = 'python -u run_summarization.py --mode=decode --use_val_as_test=True --config_file ' + str(file_name)
	#with open(out_file_name, "w") as outfile:
	#	subprocess.call(test_command, stdout=outfile)
	os.system(test_command)	

with open(file_name_1) as f_1:
	doc_1 = yaml.load(f_1)


with open(file_name_2) as f_2:
	doc_2 = yaml.load(f_2)

pool = Pool(processes=4)
res1 = pool.apply_async(run_train,[file_name_1,doc_1['exp_name']])
res2 = pool.apply_async(run_eval,[file_name_1,doc_1['exp_name']])
res3 = pool.apply_async(run_train,[file_name_2,doc_2['exp_name']])
res4 = pool.apply_async(run_eval,[file_name_2,doc_2['exp_name']])
pool.close()
pool.join()


restore_best_model_command_1 = 'python run_summarization.py --mode=restore_best_model --config_file ' + str(file_name_1)
os.system(restore_best_model_command_1)

restore_best_model_command_2 = 'python run_summarization.py --mode=restore_best_model --config_file ' + str(file_name_2)
os.system(restore_best_model_command_2)

pool = Pool(processes=4)
res1 = pool.apply_async(run_test,[file_name_1,doc_1['exp_name']])
res2 = pool.apply_async(run_eval_test,[file_name_1,doc_1['exp_name']])
res3 = pool.apply_async(run_test,[file_name_2,doc_2['exp_name']])
res4 = pool.apply_async(run_eval_test,[file_name_2,doc_2['exp_name']])
pool.close()
pool.join()



print('Completed!')
