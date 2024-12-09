import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class Food101(DatasetBase):
    SUPERCLASS_MAPPING = {
        "apple_pie": "dessert",
        "baklava": "dessert",
        "beignets": "dessert",
        "bread_pudding": "dessert",
        "carrot_cake": "dessert",
        "cheesecake": "dessert",
        "chocolate_cake": "dessert",
        "chocolate_mousse": "dessert",
        "churros": "dessert",
        "creme_brulee": "dessert",
        "cup_cakes": "dessert",
        "donuts": "dessert",
        "frozen_yogurt": "dessert",
        "cannoli": "dessert",
        
        "baby_back_ribs": "main_course",
        "bibimbap": "main_course",
        "breakfast_burrito": "main_course",
        "chicken_curry": "main_course",
        "chicken_quesadilla": "main_course",
        "croque_madame": "main_course",
        "filet_mignon": "main_course",
        "fish_and_chips": "main_course",
        "gnocchi": "main_course",
        "grilled_salmon": "main_course",
        "fried_rice": "main_course",
        
        "beef_carpaccio": "appetizer",
        "beef_tartare": "appetizer",
        "bruschetta": "appetizer",
        "ceviche": "appetizer",
        "cheese_plate": "appetizer",
        "crab_cakes": "appetizer",
        "deviled_eggs": "appetizer",
        "dumplings": "appetizer",
        "edamame": "appetizer",
        "escargots": "appetizer",
        "foie_gras": "appetizer",
        "fried_calamari": "appetizer",
        "clam_chowder": "appetizer",
        "french_onion_soup": "appetizer",
        
        "chicken_wings": "snack",
        "club_sandwich": "snack",
        "falafel": "snack",
        "grilled_cheese_sandwich": "snack",
        "french_fries": "snack",
        "garlic_bread": "snack",
        "eggs_benedict": "snack",
        "french_toast": "snack",
        
        "beet_salad": "salad",
        "caesar_salad": "salad",
        "caprese_salad": "salad",
        "greek_salad": "salad",
        
        # Dessert
    "panna_cotta": "dessert",
    "tiramisu": "dessert",
    "strawberry_shortcake": "dessert",
    "red_velvet_cake": "dessert",
    "macarons": "dessert",
    "ice_cream": "dessert",

    # Main Course
    "risotto": "main_course",
    "prime_rib": "main_course",
    "paella": "main_course",
    "pad_thai": "main_course",
    "peking_duck": "main_course",
    "hamburger": "main_course",
    "pork_chop": "main_course",
    "lasagna": "main_course",
    "pizza": "main_course",
    "steak": "main_course",
    "macaroni_and_cheese": "main_course",
    "pho": "main_course",
    "omelette": "main_course",
    "ramen": "main_course",
    "shrimp_and_grits": "main_course",
    "spaghetti_bolognese": "main_course",
    "spaghetti_carbonara": "main_course",
    "sushi": "main_course",
    "tacos": "main_course",

    # Appetizer
    "spring_rolls": "appetizer",
    "samosa": "appetizer",
    "takoyaki": "appetizer",
    "lobster_bisque": "appetizer",
    "sashimi": "appetizer",
    "gyoza": "appetizer",
    "tuna_tartare": "appetizer",
    "hot_and_sour_soup": "appetizer",
    "miso_soup": "appetizer",
    "scallops": "appetizer",
    "mussels": "appetizer",
    "oysters": "appetizer",
    "guacamole": "appetizer",

    # Snack
    "onion_rings": "snack",
    "pulled_pork_sandwich": "snack",
    "hot_dog": "snack",
    "lobster_roll_sandwich": "snack",
    "nachos": "snack",
    "hummus": "snack",
    "waffles": "snack",

    # Salad
    "seaweed_salad": "salad",
    "huevos_rancheros": "main_course",  # 주로 아침 식사로 제공
    "pancakes": "dessert",  # 디저트 또는 아침으로 제공
    "poutine": "snack",  # 간식으로 주로 분류됨
    "ravioli": "main_course",  # 메인 코스 파스타
    }

    dataset_dir = "food-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
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
        superclass = self.SUPERCLASS_MAPPING.get(item.classname, "food")
        return item._replace(superclass=superclass)
