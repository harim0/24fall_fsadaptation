import os
# epoch 50(overfitting 의심)

# test_script1 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_test.sh'
# test_script2 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_test_prompt2.sh'
# test_script3 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_test_prototype.sh'
# test_script4 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/base2new_test_promptotype.sh'

test_script3 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/all/all_test_prompt2.sh'
test_script4 = '/home/harim/ai_finalproject/CoOp/scripts/cocoop/all/all_test.sh'

dataset_dir = '/home/harim/ai_finalproject/CoOp/configs/datasets'

os.chdir('/home/harim/ai_finalproject/CoOp/scripts/cocoop')

epoch = 10

datset_list = {"caltech101",	"dtd",	"eurosat",	"fgvc_aircraft",	"food101",	"imagenet",	"oxford_flowers",	"oxford_pets"	,"stanford_cars"	,"sun397",	"ucf101"}
for dataset in os.listdir(dataset_dir):
    dataset = dataset.split('.')[0]
    if dataset in datset_list:
        # try:
        #     os.system(f"bash {test_script1} {dataset} 1 {epoch}")
        # except:
        #     print(f"Err,,,,, {dataset}")
        
        # try:
        #     os.system(f"bash {test_script2} {dataset} 1 {epoch}")
        # except:
        #     print(f"Err,,,,, {dataset}")
            
        try:
            os.system(f"bash {test_script3} {dataset} 1 {epoch}")
        except:
            print(f"Err,,,,, {dataset}")
            
        try:
            os.system(f"bash {test_script4} {dataset} 1 {epoch}")
        except:
            print(f"Err,,,,, {dataset}")

