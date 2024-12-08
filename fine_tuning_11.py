import os

script = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_train.sh'
test_script = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_test.sh'

script2 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_train_prototype.sh'
test_script2 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_test_prototype.sh'

dataset_dir = '/home/harim/ai_finalproject/CoOp/configs/datasets'

os.chdir('/home/harim/ai_finalproject/CoOp/scripts/cocoop')

for dataset in os.listdir(dataset_dir):
    dataset = dataset.split('.')[0]
    try:
        os.system(f"bash {script} {dataset} 1")
        os.system(f"bash {test_script} {dataset} 1")
    except:
        print(f"Err,,,,,{script} {dataset}")

for dataset in os.listdir(dataset_dir):
    dataset = dataset.split('.')[0]
    try:
        os.system(f"bash {script2} {dataset} 1")
        os.system(f"bash {test_script2} {dataset} 1")
    except:
        print(f"Err,,,,,{script2} {dataset}")