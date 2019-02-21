from __future__ import print_function
import sys
import glob
import os
import pandas as pd
from nlgeval import NLGEval
metric_to_omit = ['METEOR','CIDEr','SkipThoughtCS','GreedyMatchingScore','EmbeddingAverageCosineSimilairty','VectorExtremaCosineSimilarity']
nlgeval = NLGEval(metrics_to_omit=metric_to_omit)
input_path = sys.argv[1]
#print(input_path.split('/')[-2])
try:
        val_file = pd.read_csv(input_path.split('/')[-2] + '.csv')
        bleu_val = max(val_file['bleu_4'])
except:
        bleu_val = 0.0
print(bleu_val)
#ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
#refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
  
#dec = open(os.path.join(input_path, 'decoded_n.txt'),'w')
#ref = open(os.path.join(input_path,  'reference_n.txt'), 'w')
dec = []
ref = []
dec_files = glob.glob(os.path.join(input_path, 'decoded/*.txt'))
ref_files = glob.glob(os.path.join(input_path, 'reference/*.txt'))
steps = int(input_path.split('-')[-1])
#print(len(dec_files))
#print(len(ref_files))
for i in sorted(dec_files):
        f = open(i,'r')
        s = ''
        for k in f.readlines():
                s = s + k.strip().replace('\n','')
        #print(s)
        #print(s,file=dec)
        dec.append(s)

for i in sorted(ref_files):
        f = open(i,'r')
        s = ''
        for k in f.readlines():
                s = s + k.strip().replace('\n','')
        #s = s + ' <eos>'
        #print(s,file=ref)
        ref.append(s)

metrics_dict = nlgeval.compute_metrics([ref],dec)

#dec.close()
#ref.close()
print("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%(steps, bleu_val, metrics_dict['Bleu_1']*100, metrics_dict['Bleu_2']*100, metrics_dict['Bleu_3']*100, metrics_dict['Bleu_4']*100))

