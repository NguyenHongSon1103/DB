import os
import datetime

print('RUNING EVALUATE')
with open('logEval.txt', 'a') as f:
    f.write(str(datetime.datetime.now())+'\n')
for name in ['aeon', 'bigc', 'coopmart', 'lotte', 'mega', 'mega_new', 'satra', 'tgs', 'vinmart', '711']:
    print('evaluate for %s\n'%name)
    for it in range(50, 55, 5):
        cmd = 'python evaluate.py -l ../Receipt_dataset/test/%s/ -i ../Receipt_dataset/test/%s/ -m trained_models/model_t20220517/savedmodel -it %.2f'%(name, name, it/100)
#     os.system(cmd)
        log = os.popen(cmd).read()
        with open('logEval.txt', 'a') as f:
            f.write('evaluate for %s with %.2f\n'%(name, it/100))
            f.write(log)
    print(name, it)
f.close()