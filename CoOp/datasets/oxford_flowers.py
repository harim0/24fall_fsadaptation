import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class OxfordFlowers(DatasetBase):
    SUPERCLASS_MAPPING = {
        "pink primrose": "wildflower",
        "canterbury bells": "wildflower",
        "corn poppy": "wildflower",
        "common dandelion": "wildflower",
        "mexican aster": "wildflower",
        "tiger lily": "ornamental",
        "purple coneflower": "ornamental",
        "peruvian lily": "ornamental",
        "giant white arum lily": "ornamental",
        "yellow iris": "aquatic",
        "red ginger": "tropical",
        "artichoke": "medicinal",
        "balloon flower": "ornamental",
        "pincushion flower": "ornamental",
        "cape flower": "ornamental",
        "poinsettia": "ornamental",
        "buttercup": "wildflower",
        "spear thistle": "wildflower",
        "love in the mist": "wildflower",
        "globe-flower": "wildflower",
        "globe thistle": "wildflower",
        "oxeye daisy": "wildflower",
        "alpine sea holly": "wildflower",
        "stemless gentian": "wildflower",
        "wallflower": "wildflower",
        "fritillary": "wildflower",
        "great masterwort": "wildflower",
        "moon orchid": "ornamental",
        "king protea": "ornamental",
        "sweet william": "ornamental",
        "sweet pea": "ornamental",
        "garden phlox": "ornamental",
        "barbeton daisy": "ornamental",
        "marigold": "ornamental",
        "english marigold": "ornamental",
        "grape hyacinth": "ornamental",
        "sword lily": "ornamental",
        "daffodil": "ornamental",
        "petunia": "ornamental",
        "fire lily": "ornamental",
        "bolero deep blue": "ornamental",
        "lenten rose": "ornamental",
        "snapdragon": "ornamental",
        "carnation": "ornamental",
        "prince of wales feathers": "ornamental",
        "siam tulip": "tropical",
        "bird of paradise": "tropical",
        "ruby-lipped cattleya": "tropical",
        "hard-leaved pocket orchid": "tropical",
        "monkshood": "medicinal",
        "colt's foot": "medicinal",
    "lotus": "aquatic",
    "californian poppy": "wildflower",
    "camellia": "ornamental",
    "windflower": "wildflower",
    "osteospermum": "ornamental",
    "japanese anemone": "ornamental",
    "hibiscus": "ornamental",
    "ball moss": "ornamental",
    "pelargonium": "ornamental",
    "tree mallow": "wildflower",
    "columbine": "wildflower",
    "toad lily": "wildflower",
    "desert-rose": "ornamental",
    "geranium": "ornamental",
    "trumpet creeper": "ornamental",
    "primula": "ornamental",
    "frangipani": "tropical",
    "bougainvillea": "ornamental",
    "bishop of llandaff": "ornamental",
    "mallow": "wildflower",
    "azalea": "ornamental",
    "foxglove": "medicinal",
    "magnolia": "ornamental",
    "rose": "ornamental",
    "orange dahlia": "ornamental",
    "cautleya spicata": "tropical",
    "canna lily": "ornamental",
    "watercress": "aquatic",
    "gazania": "ornamental",
    "clematis": "ornamental",
    "anthurium": "tropical",
    "gaura": "wildflower",
    "passion flower": "tropical",
    "spring crocus": "ornamental",
    "blackberry lily": "ornamental",
    "bee balm": "wildflower",
    "thorn apple": "medicinal",
    "bearded iris": "ornamental",
    "tree poppy": "wildflower",
    "silverbush": "ornamental",
    "pink-yellow dahlia": "ornamental",
    "water lily": "aquatic",
    "sunflower": "wildflower",
    "cyclamen": "ornamental",
    "mexican petunia": "ornamental",
    "hippeastrum": "ornamental",
    "wild pansy": "wildflower",
    "bromelia": "tropical",
    "black-eyed susan": "wildflower",
    "blanket flower": "wildflower",
    "morning glory": "wildflower"
    }

    dataset_dir = "oxford_flowers"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
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
        superclass = self.SUPERCLASS_MAPPING.get(item.classname, "flower")
        return item._replace(superclass=superclass)

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test
