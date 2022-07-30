from .objects import *

CORRESPONDENCE_DICT = {}

AI2Thor_NO_RELATION = {
    'Fork': Fork,
    'Mirror': Mirror,
    'CounterTop': CounterTop,
    'Desk': Desk,
    'PaperTowelRoll': PaperTowelRoll,
    'Lettuce': Lettuce,
    'SoapBar': SoapBar,
    'Bowl': Bowl,
    'Bed': Bed,
    'WateringCan': WateringCan,
    'Dumbbell': Dumbbell,
    'Curtains': Curtains,
    'Sofa': Sofa,
    'Poster': Poster,
    'Cabinet': Cabinet,
    'SideTable': SideTable,
    'Cup': Cup,
    'CoffeeTable': CoffeeTable,
    'Drawer': Drawer,
    'BaseballBat': BaseballBat,
    'AlarmClock': Clock,
    'AluminumFoil': AluminumFoil,
    'ShowerCurtain': ShowerCurtain,
    'Stool': Stool,
    'Spatula': Spatula,
    'Sink': Sink,
    'GarbageBag': GarbageBag,
    'Footstool': Footstool,
    'CreditCard': CreditCard,
    'ShowerGlass': ShowerGlass,
    'Shelf': Shelf,
    'TableTopDecor': TableTopDecor,
    'KeyChain': KeyChain,
    'TVStand': TVStand,
    'SaltShaker': SaltShaker,
    'PepperShaker': PepperShaker,
    'DogBed': DogBed,
    'Painting': Painting,
    'ShowerHead': ShowerHead,
    'LaundryHamper': LaundryHamper,
    'CD': CD,
    'SinkBasin': SinkBasin,
    'Plate': Plate,
    'HousePlant': HousePlant,
    'ShelvingUnit': ShelvingUnit,
    'Pan': Pan,
    'BasketBall': BasketBall,
    'Statue': Statue,
    'TeddyBear': TeddyBear,
    'Bathtub': Bathtub,
    'BathtubBasin': BathtubBasin,
    'Mug': Mug,
    'Desktop': Desktop,
    'TissueBox': TissueBox,
    'DiningTable': DiningTable,
    'Spoon': Spoon,
    'Dresser': Dresser,
    'DishSponge': DishSponge,
    'Bottle': Bottle,
    'ArmChair': FoldingChair,
    # 'ShowerDoor': ShowerDoor,
    'ShowerDoor': DoorsetNoTrigger,
    'Pot': Pot,
    'Watch': Watch,
    'Box': Box,
    'RoomDecor': RoomDecor,
    'TennisRacket': TennisRacket,
    'Cloth': Cloth,
    'Window': Window,
    'Pillow': Pillow,
    'Ottoman': Ottoman,
    'Chair': FoldingChair,
    'SoapBottle': SoapBottle,
    'Kettle': Kettle,
    'Vase': Vase,
    'WineBottle': WineBottle,
    'Ladle': Ladle,
    'Boots': Boots,
}

for key, value in AI2Thor_NO_RELATION.items():
    CORRESPONDENCE_DICT[key] = value

AI2Thor_NO_RELATION_KEY = list(AI2Thor_NO_RELATION.keys())

AI2Thor_SELF_RELATION = {
    'CoffeeMachine': CoffeeMachine,
    'Faucet': Faucet,
    'GarbageCan': TrashCan,
    'Safe': Safe,
    'Candle': Candle,
    'Microwave': Microwave,
    'Toaster': Toaster,
    'DeskLamp': DeskLamp,
    'SprayBottle': Dispenser,
    'WashingMachine': WashingMachine,
    'Laptop': Laptop,
}

for key, value in AI2Thor_SELF_RELATION.items():
    CORRESPONDENCE_DICT[key] = value

AI2Thor_SELF_RELATION_KEY = list(AI2Thor_SELF_RELATION.keys())

AI2Thor_BINARY_RELATION_1_TO_ALL = [
    ('Microwave', Microwave, 'Bread', Bread),
    ('Microwave', Microwave, 'Potato', Potato),
    ('Microwave', Microwave, 'Egg', Egg),
    ('Microwave', Microwave, 'Tomato', Tomato),
    ('Microwave', Microwave, 'Apple', Apple),
    ('Fridge', Refrigerator, 'Bread', Bread),
    ('Fridge', Refrigerator, 'Potato', Potato),
    ('Fridge', Refrigerator, 'Egg', Egg),
    ('Fridge', Refrigerator, 'Tomato', Tomato),
    ('Fridge', Refrigerator, 'Apple', Apple),
    ('Knife', Knife, 'Bread', Bread),
    ('Knife', Knife, 'Potato', Potato),
    ('Knife', Knife, 'Apple', Apple),
    ('Knife', Knife, 'Tomato', Tomato),
    ('ButterKnife', Knife, 'Bread', Bread),
    ('ButterKnife', Knife, 'Potato', Potato),
    ('ButterKnife', Knife, 'Bread', Bread),
    ('ButterKnife', Knife, 'Potato', Potato),
    ('Pen', Pen, 'Newspaper', Newspaper),
    ('Pencil', Pencil, 'Newspaper', Newspaper),
    ('Pen', Pen, 'Book', Book),
    ('Pencil', Pencil, 'Book', Book),
    ('VacuumCleaner', VacuumCleaner, 'Floor', Floor),
    ('ScrubBrush', ScrubBrush, 'Toilet', Toilet),
    ('Plunger', Plunger, 'Toilet', Toilet),
    ('ToiletPaperHanger', ToiletPaperHanger, 'ToiletPaper', ToiletPaper),
    ('HandTowelHolder', HandTowelHolder, 'HandTowel', HandTowel),
    ('TowelHolder', TowelHolder, 'Towel', Towel),
    ('CellPhone', Phone, 'FloorLamp', FloorLamp),
    # ('Bathtub', Bathtub, 'BathtubBasin', BathtubBasin),
]

for SRC_NAME, SRC, DST_NAME, DST in AI2Thor_BINARY_RELATION_1_TO_ALL:
    CORRESPONDENCE_DICT[SRC_NAME] = SRC
    CORRESPONDENCE_DICT[DST_NAME] = DST


AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_KEY = []
AI2Thor_BINARY_RELATION_1_TO_ALL_DST_KEY = []
AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_DICT = {}
for SRC, _, DST, _ in AI2Thor_BINARY_RELATION_1_TO_ALL:
    if SRC not in AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_KEY:
        AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_KEY.append(SRC)
    if DST not in AI2Thor_BINARY_RELATION_1_TO_ALL_DST_KEY:
        AI2Thor_BINARY_RELATION_1_TO_ALL_DST_KEY.append(DST)
    if SRC not in AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_DICT:
        AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_DICT[SRC] = [DST]
    else:
        AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_DICT[SRC].append(DST)


AI2Thor_BINARY_RELATION_1_TO_1 = [
    ('RemoteControl', Remote, 'Television', Display),
    ('LightSwitch', PrismaticSwitch, 'CeilLamp', CeilLamp),
    ('StoveKnob', StoveKnob, 'StoveBurner', StoveBurner),
]

for SRC_NAME, SRC, DST_NAME, DST in AI2Thor_BINARY_RELATION_1_TO_1:
    CORRESPONDENCE_DICT[SRC_NAME] = SRC
    CORRESPONDENCE_DICT[DST_NAME] = DST


AI2Thor_BINARY_RELATION_1_TO_1_SRC_KEY = []
AI2Thor_BINARY_RELATION_1_TO_1_DST_KEY = []
AI2Thor_BINARY_RELATION_1_TO_1_SRC_DICT = {}
for SRC, _, DST, _ in AI2Thor_BINARY_RELATION_1_TO_1:
    if SRC not in AI2Thor_BINARY_RELATION_1_TO_1_SRC_KEY:
        AI2Thor_BINARY_RELATION_1_TO_1_SRC_KEY.append(SRC)
    if DST not in AI2Thor_BINARY_RELATION_1_TO_1_DST_KEY:
        AI2Thor_BINARY_RELATION_1_TO_1_DST_KEY.append(DST)
    AI2Thor_BINARY_RELATION_1_TO_1_SRC_DICT[SRC] = DST

AI2Thor_BINARY_RELATION_1_TO_1_MANUAL = [
    ('FanSwtich', RevoluteSwitch, 'Fan', Fan),
    ('ToiletSwtich', RevoluteSwitch, 'Toilet', Toilet),
    ('FloorLampSwtich', PrismaticSwitch, 'FloorLamp', FloorLamp),
    ('BlindSwtich', PrismaticSwitch, 'Blinds', Blinds),
    ('Mouse', Mouse, 'Laptop', Laptop),
]

for SRC_NAME, SRC, DST_NAME, DST in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL:
    CORRESPONDENCE_DICT[SRC_NAME] = SRC
    CORRESPONDENCE_DICT[DST_NAME] = DST


AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_KEY = []
AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_KEY = []
AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_DICT = {}
AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_DICT = {}
for SRC, _, DST, _ in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL:
    if SRC not in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_KEY:
        AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_KEY.append(SRC)
    if DST not in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_KEY:
        AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_KEY.append(DST)
    AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_DICT[SRC] = DST
    AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_DICT[DST] = SRC

ALL_KEY = AI2Thor_NO_RELATION_KEY + AI2Thor_SELF_RELATION_KEY + AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_KEY + AI2Thor_BINARY_RELATION_1_TO_ALL_DST_KEY + AI2Thor_BINARY_RELATION_1_TO_1_SRC_KEY + AI2Thor_BINARY_RELATION_1_TO_1_DST_KEY + AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_KEY + AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_KEY
