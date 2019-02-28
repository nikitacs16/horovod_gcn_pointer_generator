from __future__ import print_function
import sys
import glob
import rouge
import bleu
import re
import os
import pandas as pd
from nlgeval import NLGEval
metric_to_omit = ['METEOR','CIDEr','SkipThoughtCS','GreedyMatchingScore','EmbeddingAverageCosineSimilairty','VectorExtremaCosineSimilarity']
nlgeval = NLGEval(metrics_to_omit=metric_to_omit)

def get_metrics(f1,f2,steps, bleu_val):
        ref = []
        decoded = []
        count = 0
        for i, j in zip(sorted(glob.glob(f1)),sorted(glob.glob(f2))):
                ref_tex = ''
                dec_tex = ''
                for k in open(i).readlines():
                        dec_tex = dec_tex +  k.strip().replace('\n','')
                for l in open(j).readlines():
                        ref_tex = ref_tex + l.strip().replace('\n','')
                ref.append(ref_tex)
                decoded.append(dec_tex)
                count = count + 1

        metrics_dict = nlgeval.compute_metrics([ref],decoded)
        s = "%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%(steps, bleu_val, metrics_dict['Bleu_1']*100, metrics_dict['Bleu_2']*100, metrics_dict['Bleu_3']*100, metrics_dict['Bleu_4']*100,metrics_dict["Answerability"]*100 )
        print(count)
        return s


input_path = sys.argv[1]

for k in sorted(glob.glob(input_path+'/*skip_entity*run_2*')):
        print(k)
        try:
                x = glob.glob(os.path.join(k,'decode_test*'))
        except IndexError:

                continue
        x.sort()

        s = ''
        for i in x:
                s = ''
                steps = int(i.split('-')[-1])
                try:
                        #print(i.split('/')[-2] + '.csv')
                        val_file = pd.read_csv(i.split('/')[-2] + '.csv')
                        bleu_val = max(val_file['bleu_4'])
                except:
                        bleu_val = 0.0

                f1 = os.path.join(i,'decoded','*.txt')
                f2 = os.path.join(i,'reference','*.txt')
                s = s + get_metrics(f1,f2,steps,bleu_val)

        print(s)

