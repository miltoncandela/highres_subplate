from random import sample, seed
import os

subjects_dir = 'upsamp/done'
subjects = os.listdir(subjects_dir)

n_folds = 5
p_valid = 0.1
n_valid = round(p_valid * len(subjects))

print('Training:', len(subjects) - n_valid, 'Validation:', n_valid)

for n_seed in range(n_folds):
    seed(n_seed)
    sub_sample = sample(subjects, n_valid)
    
    # t = 1
    # while len(set(['FCB128', 'FCB046']).intersection(set(sub_sample))) > 0:
    #     seed(n_seed + 10 + t)
    #     sub_sample = sample(subjects, n_valid)
    #     t +=1
    
    with open('data/set40/fold_{}.txt'.format(n_seed), "w") as file:
        file.writelines("%s\n" % x for x in sub_sample)
    
    print(sub_sample)
