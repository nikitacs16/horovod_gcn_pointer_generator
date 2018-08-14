import os
import sys
import yaml
file_name = sys.argv[1]
folds = sys.argv[2]
parent_path = sys.argv[3]

for i in range(1,folds+1):
	with open(file_name) as f:
        doc = yaml.load(f)
    doc['train_path']= os.path.join(parent_path,str(i),'train*.pkl')
	doc['dev_path']= os.path.join(parent_path,str(i),'val*.pkl')
	doc['test_path']= os.path.join(parent_path,str(i),'test*.pkl')
	doc['vocab_path'] =  os.path.join(parent_path,str(i),'finished_files/vocab')
	doc['exp_name'] = doc['base_name'] + '_' + str(i)
	with open(file_name, 'w') as f:
        yaml.dump(doc, f)	

	os.system('python -u run_summarization.py --mode=train --config_file ' + str(file_name) + ' &> out_'+str(doc['exp_name']) +' ; sleep 100; python -u run_summarization.py --mode=eval --config_file ' + str(file_name) + ' &> out_v_'+str(doc['exp_name']))
	os.system('python -u run_summarization.py --mode=restore_best_model --config_file ' + str(file_name) +' &> out_s_'+str(doc['exp_name']))
	os.system('python -u run_summarization.py --mode=decode --config_file '+str(file_name) +  ' &> out_t'+str(doc['exp_name']) + ' & python -u run_summarization.py --mode=decode --use_val_as_test=True --config_file '+str(file_name) + ' &> out_v_'+str(doc['exp_name']))
	doc['use_stop_after'] = True
	doc['stop_steps'] = 400
	with open(file_name, 'w') as f:
        yaml.dump(doc, f)	
	os.system('python -u run_summarization.py --mode=train --convert_to_coverage_model=True --coverage=True --config_file '+str(file_name) + ' &> out_c'+str(i)+str(doc['exp_name']))
	#os.system('python -u run_summarization.py --mode=train --coverage=True --config_file '+str(file_name) + ' &> out')
	#os.system('python -u run_summarization.py --mode=decode --coverage=True --config_file '+str(file_name) +  ' &> out_t' +' & python -u run_summarization.py --coverage=True --use_val_as_test=True --mode=decode --config_file '+str(file_name) + ' &> out_v')
	print(i)

print('Completed!')
