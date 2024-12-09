import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class FGVCAircraft(DatasetBase):

    SUPERCLASS_MAPPING = {
    # International/large commercial aircraft
    "A321": "long-haul",
    "A320": "long-haul",
    "A319": "long-haul",
    "A318": "long-haul",
    "A330-200": "long-haul",
    "A330-300": "long-haul",
    "A340-200": "long-haul",
    "A340-300": "long-haul",
    "A340-500": "long-haul",
    "A340-600": "long-haul",
    "A380": "long-haul",
    "757-200": "long-haul",
    "757-300": "long-haul",
    "767-200": "long-haul",
    "767-300": "long-haul",
    "767-400": "long-haul",
    "747-100": "long-haul",
    "747-200": "long-haul",
    "747-300": "long-haul",
    "747-400": "long-haul",
    "777-200": "long-haul",
    "777-300": "long-haul",
    "707-320": "long-haul",
    "An-12": "long-haul",
    "C-130": "long-haul",
    "C-47": "long-haul",

    # Short-haul/medium commercial aircraft
    "727-200": "short-haul",
    "737-200": "short-haul",
    "737-300": "short-haul",
    "737-400": "short-haul",
    "737-500": "short-haul",
    "737-600": "short-haul",
    "737-700": "short-haul",
    "737-800": "short-haul",
    "737-900": "short-haul",
    "ATR-72": "short-haul",
    "ATR-42": "short-haul",
    "A300B4": "short-haul", 
    "A310": "short-haul", 

    # Regional/small commercial aircraft
    "CRJ-200": "regional",
    "CRJ-700": "regional",
    "CRJ-900": "regional",
    "BAE 146-200": "regional",
    "BAE 146-300": "regional",
    "Boeing 717": "regional",

    # Civil/private aircraft
    "Cessna 172": "civil",
    "Beechcraft 1900": "civil",
    "Cessna 525": "civil",
    "Cessna 208": "civil",
    "BAE-125": "civil",
    
    # Long-Haul
    "L-1011": "long-haul",
    "MD-11": "long-haul",
    "Tu-134": "long-haul",
    "Tu-154": "long-haul",
    "Il-76": "long-haul",
    "DC-10": "long-haul",
    "DC-8": "long-haul",

    # Short-Haul
    "MD-80": "short-haul",
    "MD-87": "short-haul",
    "MD-90": "short-haul",
    "DC-9-30": "short-haul",
    "Fokker 100": "short-haul",
    "Fokker 70": "short-haul",
    "Saab 2000": "short-haul",
    "E-190": "short-haul",
    "E-195": "short-haul",

    # Regional
    "DHC-8-100": "regional",
    "DHC-8-300": "regional",
    "DHC-6": "regional",
    "Saab 340": "regional",
    "ERJ 145": "regional",
    "ERJ 135": "regional",
    "E-170": "regional",
    "Dornier 328": "regional",
    "EMB-120": "regional",
    "Metroliner": "regional",
    "Fokker 50": "regional",

    # Civil
    "PA-28": "civil",
    "DR-400": "civil",
    "Falcon 900": "civil",
    "Cessna 560": "civil",
    "Challenger 600": "civil",
    "Global Express": "civil",
    "Gulfstream IV": "civil",
    "Gulfstream V": "civil",
    "Model B200": "civil",
    "Falcon 2000": "civil",
    "Cessna 208": "civil",
    "Embraer Legacy 600": "civil", 
    "SR-20": "civil",

    # Military
    "Hawk T1": "military",
    "Tornado": "military",
    "F/A-18": "military",
    "F-16A/B": "military",
    "Eurofighter Typhoon": "military",
    "Spitfire": "military",

    # Other
    "DHC-1": "other",
    "DH-82": "other",
    "DC-3": "other",
    "DC-6": "other",
    "Yak-42": "other",
}

    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

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
        superclass = self.SUPERCLASS_MAPPING.get(item.classname, "aircraft")
        return item._replace(superclass=superclass)
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
