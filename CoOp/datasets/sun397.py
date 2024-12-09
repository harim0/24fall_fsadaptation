import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class SUN397(DatasetBase):
    SUPERCLASS_MAPPING = {
        "bedroom": "residential",
    "childs_room": "residential",
    "dorm_room": "residential",
    "home_office": "residential",
    "hotel_room": "residential",
    "house": "residential",
    "airport_terminal": "commercial",
    "bar": "commercial",
    "bookstore": "commercial",
    "clothing_store": "commercial",
    "delicatessen": "commercial",
    "food_court": "commercial",
    "gift_shop": "commercial",
    "ice_cream_parlor": "commercial",
    "amphitheater": "recreational",
    "ballroom": "recreational",
    "baseball_field": "recreational",
    "bowling_alley": "recreational",
    "boxing_ring": "recreational",
    "discotheque": "recreational",
    "golf_course": "recreational",
    "badlands": "natural",
    "bamboo_forest": "natural",
    "bayou": "natural",
    "beach": "natural",
    "canyon": "natural",
    "creek": "natural",
    "forest_path": "natural",
    "forest_road": "natural",
    "hot_spring": "natural",
    "ice_floe": "natural",
    "bridge": "transportational",
    "bus_interior": "transportational",
    "dock": "transportational",
    "gas_station": "transportational",
    "harbor": "transportational",
    "highway": "transportational",
    "assembly_line": "industrial",
    "construction_site": "industrial",
    "engine_room": "industrial",
    "fire_station": "industrial",
    "abbey": "religious/cultural",
    "basilica": "religious/cultural",
    "art_gallery": "religious/cultural",
    "arch": "other",
    "alley": "other",
    "courtyard": "other",
    "fountain": "other",
    # Residential
    "attic": "residential",
    "basement": "residential",
    "bedroom": "residential",
    "home dinette": "residential",
    "dining_room": "residential",
    "closet": "residential",
    "backseat car_interior": "residential",
    "frontseat car_interior": "residential",

    # Commercial
    "dentists_office": "commercial",
    "drugstore": "commercial",
    "candy_store": "commercial",
    "coffee_shop": "commercial",
    "butchers_shop": "commercial",
    "indoor diner": "commercial",
    "indoor bistro": "commercial",
    "fastfood_restaurant": "commercial",
    "indoor florist_shop": "commercial",
    "indoor general_store": "commercial",
    "indoor ice_skating_rink": "commercial",
    "hospital": "commercial",
    "beauty_salon": "commercial",
    "cafeteria": "commercial",

    # Recreational
    "amusement_arcade": "recreational",
    "ball_pit": "recreational",
    "batters_box": "recreational",
    "indoor badminton_court": "recreational",
    "game_room": "recreational",
    "golf_course": "recreational",
    "indoor gymnasium": "recreational",
    "indoor casino": "recreational",
    "indoor firing_range": "recreational",
    "amusement_park": "recreational",
    "outdoor basketball_court": "recreational",
    "outdoor athletic_field": "recreational",
    "outdoor driving_range": "recreational",

    # Natural
    "wild field": "natural",
    "corn_field": "natural",
    "herb_garden": "natural",
    "vegetation desert": "natural",
    "broadleaf forest": "natural",
    "needleleaf forest": "natural",
    "natural canal": "natural",
    "crevasse": "natural",
    "hill": "natural",
    "butte": "natural",
    "coast": "natural",
    "cliff": "natural",
    "sand desert": "natural",
    "ice_shelf": "natural",
    "aquarium": "natural",

    # Transportational
    "airplane_cabin": "transportational",
    "elevator_shaft": "transportational",
    "door elevator": "transportational",
    "control_room": "transportational",
    "outdoor control_tower": "transportational",
    "baggage_claim": "transportational",
    "driveway": "transportational",
    "exterior covered_bridge": "transportational",
    "corridor": "transportational",
    "crosswalk": "transportational",
    "bridge": "transportational",
    "campsite": "transportational",

    # Industrial
    "auto_factory": "industrial",
    "cheese_factory": "industrial",
    "clean_room": "industrial",
    "indoor factory": "industrial",
    "indoor brewery": "industrial",
    "assembly_line": "industrial",

    # Religious/Cultural
    "indoor church": "religious/cultural",
    "indoor cathedral": "religious/cultural",
    "indoor cloister": "religious/cultural",
    "outdoor cathedral": "religious/cultural",
    "indoor apse": "religious/cultural",
    "indoor bow_window": "religious/cultural",
    "indoor greenhouse": "religious/cultural",
    "formal_garden": "religious/cultural",
    "outdoor greenhouse": "religious/cultural",
    "indoor hangar": "religious/cultural",
    "burial_chamber": "religious/cultural",

    # Other
    "barn": "other",
    "aqueduct": "other",
    "chalet": "other",
    "exterior gazebo": "other",
    "public atrium": "other",
    "castle": "other",
    "conference_room": "other",
    "biology_laboratory": "other",
    "chemistry_lab": "other",
    "archive": "other",
    "art_school": "other",
    "art_studio": "other",
    "anechoic_chamber": "other",
    "urban canal": "other",
    "dam": "other",
    "barndoor": "other",
    "fire_escape": "other",
    "outdoor cabin": "other",
    "outdoor hotel": "other",
    "outdoor apartment_building": "other",
    "exterior balcony": "residential",

    # Commercial
    "indoor chicken_coop": "commercial",
    "outdoor general_store": "commercial",
    "outdoor bazaar": "commercial",
    "galley": "commercial",

    # Transportational
    "indoor escalator": "transportational",

    # Industrial
    "outdoor chicken_coop": "industrial",

    # Other
    "indoor cavern": "other",
    "computer_room": "other",
    "indoor garage": "residential",
    "bathroom": "residential",
    "interior balcony": "residential",

    # Commercial
    "shop bakery": "commercial",
    "indoor bazaar": "commercial",
    "office cubicle": "commercial",
    "boardwalk": "commercial",
    "courthouse": "commercial",
    "hospital_room": "commercial",
    "classroom": "commercial",
    "vehicle dinette": "commercial",
    "outdoor diner": "commercial",

    # Recreational
    "banquet_hall": "recreational",
    "auditorium": "recreational",
    "carrousel": "recreational",
    "bullring": "recreational",
    "fairway": "recreational",
    "cottage_garden": "recreational",
    "campus": "recreational",
    "dining_car": "recreational",
    "outdoor hot_tub": "recreational",
    "boathouse": "recreational",

    # Natural
    "hayfield": "natural",
    "cemetery": "natural",
    "botanical_garden": "natural",
    "corral": "natural",
    "fishpond": "natural",
    "cultivated field": "natural",

    # Transportational
    "outdoor doorway": "transportational",
    "outdoor arrival_gate": "transportational",
    "boat_deck": "transportational",
    "interior elevator": "transportational",
    "outdoor bow_window": "transportational",
    "heliport": "transportational",

    # Industrial
    "outdoor hangar": "industrial",
    "electrical_substation": "industrial",
    "garbage_dump": "industrial",
    "building_facade": "industrial",
    "excavation": "industrial",

    # Religious/Cultural
    "catacomb": "religious/cultural",
    "outdoor church": "religious/cultural",
    "outdoor hunting_lodge": "religious/cultural",
    "courtroom": "religious/cultural",  # 법적 및 문화적 장소로 분류

    # Commercial
    "indoor booth": "commercial",  # 상업적 공간 내 부스를 나타냄
    # Other
    "cockpit": "other",
    "berth": "other",
    "conference_center": "other",
    "shopfront": "commercial",
    "wind_farm": "industrial",
    "indoor jail": "other",
    "reception": "commercial",
    "indoor synagogue": "religious/cultural",
    "raft": "transportational",
    "videostore": "commercial",
    "pasture": "natural",
    "indoor movie_theater": "recreational",
    "restaurant": "commercial",
    "outdoor track": "recreational",
    "rope_bridge": "transportational",
    "indoor podium": "recreational",
    "indoor parking_garage": "transportational",
    "indoor shopping_mall": "commercial",
    "skatepark": "recreational",
    "restaurant_kitchen": "commercial",
    "outdoor observatory": "recreational",
    "jewelry_shop": "commercial",
    "kitchen": "residential",
    "plaza": "other",
    "indoor mosque": "religious/cultural",
    "igloo": "residential",
    "coral_reef underwater": "natural",
    "indoor pub": "commercial",
    "sky": "natural",
    "ski_slope": "recreational",
    "natural lake": "natural",
    "television_studio": "other",
    "lecture_room": "commercial",
    "music_store": "commercial",
    "ruin": "natural",
    "throne_room": "religious/cultural",
    "nursery": "residential",
    "indoor tennis_court": "recreational",
    "laundromat": "commercial",
    "lighthouse": "transportational",
    "lobby": "commercial",
    "outdoor lido_deck": "recreational",
    "schoolhouse": "commercial",
    "trench": "natural",
    "fan waterfall": "natural",
    "ocean": "natural",
    "outdoor power_plant": "industrial",
    "skyscraper": "commercial",
    "pharmacy": "commercial",
    "landfill": "industrial",
    "indoor stage": "recreational",
    "supermarket": "commercial",
    "slum": "residential",
    "indoor swimming_pool": "recreational",
    "block waterfall": "natural",
    "outdoor oil_refinery": "industrial",
    "veranda": "residential",
    "platform train_station": "transportational",
    "waiting_room": "transportational",
    "parking_lot": "transportational",
    "indoor warehouse": "industrial",
    "outdoor inn": "residential",
    "phone_booth": "other",
    "promenade_deck": "recreational",
    "water moat": "natural",
    "platform subway_station": "transportational",
    "vegetable_garden": "natural",
    "bottle_storage wine_cellar": "industrial",
    "sandbox": "recreational",
    "swamp": "natural",
    "shower": "residential",
    "industrial_area": "industrial",
    "snowfield": "natural",
    "server_room": "industrial",
    "youth_hostel": "residential",
    "indoor museum": "religious/cultural",
    "restaurant_patio": "commercial",
    "outdoor outhouse": "residential",
    "wheat_field": "natural",
    "ski_resort": "recreational",
    "valley": "natural",
    "recreation_room": "recreational",
    "barrel_storage wine_cellar": "industrial",
    "pulpit": "religious/cultural",
    "wet_bar": "commercial",
    "kitchenette": "residential",
    "sea_cliff": "natural",
    "staircase": "residential",
    "indoor_procenium theater": "recreational",
    "indoor volleyball_court": "recreational",
    "iceberg": "natural",
    "outdoor swimming_pool": "recreational",
    "pantry": "residential",
    "vineyard": "natural",
    "watering_hole": "natural",
    "pond": "natural",
    "topiary_garden": "natural",
    "tower": "transportational",
    "village": "residential",
    "sauna": "recreational",
    "squash_court": "recreational",
    "mausoleum": "religious/cultural",
    "train_railway": "transportational",
    "tree_farm": "natural",
    "outdoor mosque": "religious/cultural",
    "veterinarians_office": "commercial",
    "parlor": "residential",
    "subway_interior": "transportational",
    "south_asia temple": "religious/cultural",
    "sandbar": "natural",
    "utility_room": "industrial",
    "outdoor tennis_court": "recreational",
    "kindergarden_classroom": "commercial",
    "outdoor market": "commercial",
    "picnic_area": "recreational",
    "rainforest": "natural",
    "pavilion": "recreational",
    "ski_lodge": "recreational",
    "runway": "transportational",
    "outdoor parking_garage": "transportational",
    "indoor wrestling_ring": "recreational",
    "rice_paddy": "natural",
    "outdoor monastery": "religious/cultural",
    "jail_cell": "other",
    "locker_room": "recreational",
    "water_tower": "industrial",
    "martial_arts_gym": "recreational",
    "street": "transportational",
    "orchard": "natural",
    "physics_laboratory": "other",
    "kasbah": "religious/cultural",
    "indoor_seats theater": "recreational",
    "music_studio": "other",
    "mountain_snowy": "natural",
    "playground": "recreational",
    "outdoor volleyball_court": "recreational",
    "ticket_booth": "transportational",
    "limousine_interior": "transportational",
    "yard": "residential",
    "thriftshop": "commercial",
    "residential_neighborhood": "residential",
    "tree_house": "residential",
    "indoor jacuzzi": "recreational",
    "railroad_track": "transportational",
    "outdoor podium": "recreational",
    "playroom": "residential",
    "wave": "natural",
    "indoor pilothouse": "transportational",
    "putting_green": "recreational",
    "toll_plaza": "transportational",
    "river": "natural",
    "palace": "religious/cultural",
    "raceway": "recreational",
    "stable": "industrial",
    "oast_house": "residential",
    "football stadium": "recreational",
    "manufactured_home": "residential",
    "home poolroom": "residential",
    "indoor market": "commercial",
    "baseball stadium": "recreational",
    "establishment poolroom": "recreational",
    "operating_room": "industrial",
    "east_asia temple": "religious/cultural",
    "medina": "religious/cultural",
    "outdoor library": "religious/cultural",
    "pagoda": "religious/cultural",
    "office_building": "commercial",
    "outdoor kennel": "industrial",
    "plunge waterfall": "natural",
    "office": "commercial",
    "rock_arch": "natural",
    "islet": "natural",
    "viaduct": "transportational",
    "landing_deck": "transportational",
    "mountain": "natural",
    "indoor library": "religious/cultural",
    "shoe_shop": "commercial",
    "sushi_bar": "commercial",
    "park": "recreational",
    "lock_chamber": "transportational",
    "outdoor synagogue": "religious/cultural",
    "toyshop": "commercial",
    "windmill": "industrial",
    "outdoor labyrinth": "recreational",
    "outdoor tent": "recreational",
    "living_room": "residential",
    "mansion": "residential",
    "motel": "residential",
    "racecourse": "recreational",
    "oilrig": "industrial",
    "van_interior": "transportational",
    "riding_arena": "recreational",
    "outdoor nuclear_power_plant": "industrial",
    "marsh": "natural",
    "shed": "residential",
    "indoor kennel": "industrial",
    "outdoor ice_skating_rink": "recreational",
    "volcano": "natural",
    "lift_bridge": "transportational",
    "outdoor planetarium": "recreational",
    "patio": "residential",
    
    }

    dataset_dir = "sun397"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            classnames = []
            with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()[1:]  # remove /
                    classnames.append(line)
            cname2lab = {c: i for i, c in enumerate(classnames)}
            trainval = self.read_data(cname2lab, "Training_01.txt")
            test = self.read_data(cname2lab, "Testing_01.txt")
            train, val = OxfordPets.split_trainval(trainval)
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
        superclass = self.SUPERCLASS_MAPPING.get(item.classname, "scene")
        return item._replace(superclass=superclass)
    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)

                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
