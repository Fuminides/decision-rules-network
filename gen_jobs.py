import os
import pandas as pd


def load_NIPS_results():
    return pd.read_csv('NIPS_crips_additive_rules_0999_8_1.csv', index_col=0, sep=';')


def good_NIPS_datasets():
    df = load_NIPS_results()

    return list(df[df['Accuracy'] == df['Accuracy']].index)

def gen_job_files_DRNET():
    dataset_folder = '../keel_datasets-master/'
    datasets_names = os.listdir(dataset_folder)
    good_datasets_local = good_NIPS_datasets()
    script_folder = 'drnet_scripts'
    if not os.path.exists(script_folder):
        os.makedirs(script_folder)
    
    template = '''#!/bin/bash\n#$ -cwd\n#$ -j y\n#$ -S /bin/bash\n#$ -q gpu.q\n#$ -l gpu=1\nsource /usr/local/gpuallocation.sh\nconda activate gpuenv'''
    
    for dataset in datasets_names:
        init = False
        script = template
        if dataset in good_datasets_local:
                                                script += '\npython mainkeel.py ' + dataset
                                                init = True
        if init:
            with open(script_folder + '/run_' + dataset + '.sh', 'w') as f:
                f.write(script)

if __name__ == '__main__':
    gen_job_files_DRNET()