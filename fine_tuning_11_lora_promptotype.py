import os
# epoch 50(overfitting 의심)

script1 = '/home/harim/ai_finalproject/CoOp/scripts/lora_promptotype/all_train_lora_promptotype.sh'
script2 = '/home/harim/ai_finalproject/CoOp/scripts/lora_promptotype/base2new_train_lora_promptotype.sh'

dataset_dir = '/home/harim/ai_finalproject/CoOp/configs/datasets'

os.chdir('/home/harim/ai_finalproject/CoOp/scripts/cocoop')

epoch = 10

datset_list = {"caltech101",	"dtd",	"eurosat",	"fgvc_aircraft",	"food101",	"imagenet",	"oxford_flowers",	"oxford_pets"	,"stanford_cars"	,"sun397",	"ucf101"}
for dataset in os.listdir(dataset_dir):
    dataset = dataset.split('.')[0]
    if dataset in datset_list:
        try:
            os.system(f"bash {script1} {dataset} 1 {epoch}")
        except:
            print(f"Err,,,,,{script1} {dataset}")

for dataset in os.listdir(dataset_dir):
    dataset = dataset.split('.')[0]
    if dataset in datset_list:
        try:
            os.system(f"bash {script2} {dataset} 1 {epoch}")
        except:
            print(f"Err,,,,,{script2} {dataset}")