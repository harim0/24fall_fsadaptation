import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]


NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}



@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):
    SUPERCLASS_MAPPING = {
    "ant": "animal",
    "beaver": "animal",
    "cougar_body": "animal",
    "cougar_face": "animal",
    "crab": "animal",
    "crayfish": "animal",
    "crocodile": "animal",
    "crocodile_head": "animal",
    "dalmatian": "animal",
    "dolphin": "animal",
    "elephant": "animal",
    "gerenuk": "animal",
    "hedgehog": "animal",
    "leopard": "animal",
    "hawksbill": "animal",
    "garfield": "animal",
    "bonsai": "plant",
    "face": "animal",
    "brain": "animal",
    "brontosaurus": "animal",
    
    
    "airplane": "vehicle",
    "motorbike": "vehicle",
    "car_side": "vehicle",
    "ferry": "vehicle",
    "helicopter": "vehicle",
    
    "anchor": "object",
    "barrel": "object",
    "binocular": "object",
    "chair": "object",
    "cup": "object",
    "dollar_bill": "object",
    "cannon": "object",
    "ceiling_fan": "object",
    
    "camera": "device",
    "cellphone": "device",
    "gramophone": "device",
    "headphone": "device",
    
    "accordion": "instrument",
    "bass": "instrument",
    "electric_guitar": "instrument",
    "euphonium": "instrument",
    "grand_piano": "instrument",
    
    "emu": "bird",
    "flamingo": "bird",
    "flamingo_head": "bird",
    
    "butterfly": "insect",
    "dragonfly": "insect",
    
    "ewer": "sculpture",
    "chandelier": "sculpture",
    "buddha": "sculpture",
    
    "rhino": "animal",
    "rooster": "animal",
    "scorpion": "animal",
    "panda": "animal",
    "octopus": "animal",
    "platypus": "animal",
    "llama": "animal",
    "kangaroo": "animal",
    "wild_cat": "animal",
    "okapi": "animal",
    "sea_horse": "animal",
    "starfish": "animal",
    "stegosaurus": "animal",
    "lobster": "animal",
    "trilobite": "animal",

    # Bird
    "ibis": "bird",
    "pigeon": "bird",

    # Plant
    "water_lilly": "plant",
    "sunflower": "plant",
    "lotus": "plant",
    "joshua_tree": "plant",
    "strawberry": "plant",

    # Insect
    "mayfly": "insect",
    "tick": "insect",

    # Vehicle
    "ketch": "vehicle",
    "schooner": "vehicle",
    "wheelchair": "vehicle",
    "inline_skate": "vehicle",

    # Object
    "stapler": "object",
    "menorah": "object",
    "stop_sign": "object",
    "pizza": "object",
    "umbrella": "object",
    "wrench": "object",
    "revolver": "object",
    "scissors": "object",
    "pagoda": "object",
    "pyramid": "object",
    "lamp": "object",
    "soccer_ball": "object",
    "windsor_chair": "object",

    # Device
    "laptop": "device",
    "watch": "device",

    # Instrument
    "mandolin": "instrument",
    "saxophone": "instrument",
    "metronome": "instrument",

    # Sculpture
    "yin_yang": "sculpture",
    "nautilus": "sculpture",
    "snoopy": "sculpture",
    "minaret": "sculpture"
    
    }
    
    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

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
        superclass = self.SUPERCLASS_MAPPING.get(item.classname, "object")
        return item._replace(superclass=superclass)