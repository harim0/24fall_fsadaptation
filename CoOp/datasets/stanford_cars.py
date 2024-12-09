import os
import pickle
from scipy.io import loadmat

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class StanfordCars(DatasetBase):

    SUPERCLASS_MAPPING = {
        "2000 AM General Hummer SUV": "SUV",
    "2007 Dodge Durango SUV": "SUV",
    "2009 Chrysler Aspen SUV": "SUV",
    "2012 BMW X6 SUV": "SUV",
    "2012 Cadillac SRX SUV": "SUV",
    "2012 Buick Enclave SUV": "SUV",
    "2007 Buick Rainier SUV": "SUV",
    "2012 Dodge Journey SUV": "SUV",
    "2012 Chevrolet Traverse SUV": "SUV",
    "2012 Chevrolet Tahoe Hybrid SUV": "SUV",
    "2012 Dodge Durango SUV": "SUV",
    "2012 BMW X3 SUV": "SUV",
    "2007 BMW X5 SUV": "SUV",
    "2012 Acura RL Sedan": "sedan",
    "2012 Acura TSX Sedan": "sedan",
    "2007 Bentley Continental Flying Spur Sedan": "sedan",
    "2009 Bentley Arnage Sedan": "sedan",
    "2011 Bentley Mulsanne Sedan": "sedan",
    "2012 Buick Verano Sedan": "sedan",
    "2012 BMW 3 Series Sedan": "sedan",
    "2012 BMW ActiveHybrid 5 Sedan": "sedan",
    "2012 Cadillac CTS-V Sedan": "sedan",
    "2007 Chevrolet Impala Sedan": "sedan",
    "2012 Chevrolet Sonic Sedan": "sedan",
    "2010 Chrysler 300 SRT-8": "sedan",
    "1994 Audi V8 Sedan": "sedan",
    "1994 Audi 100 Sedan": "sedan",
    "2011 Audi S6 Sedan": "sedan",
    "2012 Audi S4 Sedan": "sedan",
    "2007 Audi S4 Sedan": "sedan",
    "2010 Chevrolet Malibu Hybrid Sedan": "sedan",
    "2012 Audi A5 Coupe": "coupe",
    "2012 BMW 1 Series Coupe": "coupe",
    "2012 Bentley Continental GT Coupe": "coupe",
    "2007 Bentley Continental GT Coupe": "coupe",
    "2012 Audi S5 Coupe": "coupe",
    "2012 Audi TT RS Coupe": "coupe",
    "2009 Bugatti Veyron 16.4 Coupe": "coupe",
    "2012 Aston Martin Virage Coupe": "coupe",
    "2012 Bentley Continental Supersports Conv. Convertible": "convertible",
    "2012 Chevrolet Camaro Convertible": "convertible",
    "2009 Bugatti Veyron 16.4 Convertible": "convertible",
    "2012 Aston Martin V8 Vantage Convertible": "convertible",
    "2012 Audi S5 Convertible": "convertible",
    "2012 BMW Z4 Convertible": "convertible",
    "2008 Chrysler Crossfire Convertible": "convertible",
    "2010 Chrysler Sebring Convertible": "convertible",
    "2008 Chrysler PT Cruiser Convertible": "convertible",
    "2008 Audi RS 4 Convertible": "convertible",
    "2007 BMW 6 Series Convertible": "convertible",
    "2007 Chevrolet Silverado 1500 Classic Extended Cab": "pickup",
    "2012 Chevrolet Silverado 1500 Regular Cab": "pickup",
    "2012 Chevrolet Silverado 1500 Extended Cab": "pickup",
    "2012 Chevrolet Silverado 1500 Hybrid Crew Cab": "pickup",
    "2012 Chevrolet Silverado 2500HD Regular Cab": "pickup",
    "2012 Chevrolet Avalanche Crew Cab": "pickup",
    "2007 Cadillac Escalade EXT Crew Cab": "pickup",
    "2010 Dodge Ram Pickup 3500 Crew Cab": "pickup",
    "2009 Dodge Ram Pickup 3500 Quad Cab": "pickup",
    "2007 Dodge Dakota Club Cab": "pickup",
    "2010 Dodge Dakota Crew Cab": "pickup",
    "2011 Audi TT Hatchback": "hatchback",
    "1998 Eagle Talon Hatchback": "hatchback",
    "2012 BMW 3 Series Wagon": "wagon",
    "1994 Audi 100 Wagon": "wagon",
    "2002 Daewoo Nubira Wagon": "wagon",
    "2007 Dodge Caliber Wagon": "wagon",
    "2012 Dodge Caliber Wagon": "wagon",
    "2008 Dodge Magnum Wagon": "wagon",
    "1997 Dodge Caravan Minivan": "minivan",
    "2012 Chrysler Town and Country Minivan": "minivan",
    "2007 Chevrolet Express Cargo Van": "van",
    "2007 Chevrolet Express Van": "van",
    "2009 Dodge Sprinter Cargo Van": "van",
    "2009 Dodge Charger SRT-8": "sports car",
    "2011 Dodge Challenger SRT8": "sports car",
    "2007 Chevrolet Monte Carlo Coupe": "coupe",
    "2012 Audi TTS Coupe": "coupe",
    "2012 Aston Martin V8 Vantage Coupe": "sports car",
    "2012 BMW M3 Coupe": "coupe",
    "2012 Audi R8 Coupe": "sports car",
    "2012 Acura TL Sedan": "sedan",
    "2010 BMW M5 Sedan": "sedan",
    "2012 Dodge Charger Sedan": "sedan",
    "2007 Chevrolet Malibu Sedan": "sedan",
    "2012 Buick Regal GS": "sedan",
    "2010 BMW M6 Convertible": "convertible",
    "2012 Aston Martin Virage Convertible": "sports car",
    "2012 Chevrolet Corvette Convertible": "sports car",
    "2012 BMW 1 Series Convertible": "convertible",
    "2001 Acura Integra Type R": "sports car",
    "2012 Chevrolet Corvette ZR1": "sports car",
    "2007 Chevrolet Corvette Ron Fellows Edition Z06": "sports car",
    "2012 Acura ZDX Hatchback": "SUV",
    "2008 Acura TL Type-S": "sedan",
    "2009 Chevrolet TrailBlazer SS": "SUV",
    "2010 Chevrolet HHR SS": "hatchback",
    "2010 Chevrolet Cobalt SS": "sports car",
    "2012 Infiniti G Coupe IPL": "coupe",
    "2012 Hyundai Elantra Touring Hatchback": "hatchback",
    "2007 Suzuki Aerio Sedan": "sedan",
    "2012 GMC Yukon Hybrid SUV": "SUV",
    "2012 Suzuki SX4 Sedan": "sedan",
    "2012 Ferrari 458 Italia Coupe": "sports car",
    "2012 MINI Cooper Roadster Convertible": "convertible",
    "2012 Toyota 4Runner SUV": "SUV",
    "2012 Maybach Landaulet Convertible": "convertible",
    "2012 smart fortwo Convertible": "convertible",
    "2012 Mercedes-Benz S-Class Sedan": "sedan",
    "2012 Nissan NV Passenger Van": "van",
    "1999 Plymouth Neon Coupe": "coupe",
    "2012 Ford F-450 Super Duty Crew Cab": "pickup",
    "2012 Jeep Patriot SUV": "SUV",
    "2012 Ram C/V Cargo Van Minivan": "minivan",
    "2012 Ferrari 458 Italia Convertible": "sports car",
    "2012 Hyundai Tucson SUV": "SUV",
    "2007 Hyundai Elantra Sedan": "sedan",
    "2012 Hyundai Santa Fe SUV": "SUV",
    "2012 Ferrari FF Coupe": "sports car",
    "1998 Nissan 240SX Coupe": "coupe",
    "1991 Volkswagen Golf Hatchback": "hatchback",
    "2012 Volkswagen Beetle Hatchback": "hatchback",
    "2012 Jeep Grand Cherokee SUV": "SUV",
    "2012 Hyundai Sonata Sedan": "sedan",
    "2012 Volvo C30 Hatchback": "hatchback",
    "2012 Hyundai Accent Sedan": "sedan",
    "2012 FIAT 500 Convertible": "convertible",
    "1993 Geo Metro Convertible": "convertible",
    "2012 Hyundai Genesis Sedan": "sedan",
    "2012 Rolls-Royce Phantom Drophead Coupe Convertible": "convertible",
    "2012 Hyundai Veracruz SUV": "SUV",
    "2006 Ford GT Coupe": "sports car",
    "2012 McLaren MP4-12C Coupe": "sports car",
    "2012 Mercedes-Benz E-Class Sedan": "sedan",
    "2012 Tesla Model S Sedan": "sedan",
    "2011 Lincoln Town Car Sedan": "sedan",
    "2012 Suzuki Kizashi Sedan": "sedan",
    "2012 FIAT 500 Abarth": "hatchback",
    "1993 Volvo 240 Sedan": "sedan",
    "2012 Rolls-Royce Phantom Sedan": "sedan",
    "2012 Nissan Leaf Hatchback": "hatchback",
    "2011 Mazda Tribute SUV": "SUV",
    "2012 GMC Canyon Extended Cab": "pickup",
    "2012 Mitsubishi Lancer Sedan": "sedan",
    "2012 Porsche Panamera Sedan": "sedan",
    "2012 GMC Savana Van": "van",
    "2012 Ford F-150 Regular Cab": "pickup",
    "2012 Ford Edge SUV": "SUV",
    "2001 Lamborghini Diablo Coupe": "sports car",
    "2012 Land Rover LR2 SUV": "SUV",
    "2007 Ford Focus Sedan": "sedan",
    "2012 Jeep Liberty SUV": "SUV",
    "2010 HUMMER H3T Crew Cab": "pickup",
    "2012 Mercedes-Benz Sprinter Van": "van",
    "2011 Ford Ranger SuperCab": "pickup",
    "2009 Spyker C8 Coupe": "sports car",
    "2012 Toyota Corolla Sedan": "sedan",
    "2012 Scion xD Hatchback": "hatchback",
    "2012 Hyundai Azera Sedan": "sedan",
    "2012 Toyota Camry Sedan": "sedan",
    "2012 Jeep Wrangler SUV": "SUV",
    "2012 Ferrari California Convertible": "sports car",
    "2009 Spyker C8 Convertible": "sports car",
    "2012 Nissan Juke Hatchback": "hatchback",
    "2007 Volvo XC90 SUV": "SUV",
    "2011 Infiniti QX56 SUV": "SUV",
    "2007 Honda Odyssey Minivan": "minivan",
    "2012 Hyundai Veloster Hatchback": "hatchback",
    "2008 Lamborghini Reventon Coupe": "sports car",
    "2007 Ford Freestar Minivan": "minivan",
    "2012 Mercedes-Benz C-Class Sedan": "sedan",
    "2012 Toyota Sequoia SUV": "SUV",
    "2009 Mercedes-Benz SL-Class Coupe": "sports car",
    "2007 Ford Mustang Convertible": "convertible",
    "2012 Lamborghini Aventador Coupe": "sports car",
    "1993 Mercedes-Benz 300-Class Convertible": "convertible",
    "2012 Lamborghini Gallardo LP 570-4 Superleggera": "sports car",
    "2012 Honda Accord Sedan": "sedan",
    "2012 Rolls-Royce Ghost Sedan": "sedan",
    "2012 Fisker Karma Sedan": "sedan",
    "2012 GMC Acadia SUV": "SUV",
    "2012 Jaguar XK XKR": "sports car",
    "2012 Hyundai Sonata Hybrid Sedan": "sedan",
    "2012 Honda Odyssey Minivan": "minivan",
    "2012 Honda Accord Coupe": "coupe",
    "2012 Land Rover Range Rover SUV": "SUV",
    "2012 Volkswagen Golf Hatchback": "hatchback",
    "2012 Ford Fiesta Sedan": "sedan",
    "2012 Jeep Compass SUV": "SUV",
    "2012 GMC Terrain SUV": "SUV",
    "2009 HUMMER H2 SUT Crew Cab": "pickup",
    "2008 Isuzu Ascender SUV": "SUV",
    "2009 Ford Expedition EL SUV": "SUV",
    "2012 Suzuki SX4 Hatchback": "hatchback",
    "2012 Ford E-Series Wagon Van": "van",
    "2007 Ford F-150 Regular Cab": "pickup"
    }

    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        superclass = self.SUPERCLASS_MAPPING.get(item.classname, "car")
        return item._replace(superclass=superclass)    
    
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items
