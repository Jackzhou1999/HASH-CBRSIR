import os
from random import shuffle
import shutil

To_Dataset = '/home/jackzhou/PycharmProjects/CBRSIR_hash/Dataset'
From_Dataset = '/home/jackzhou/Downloads/UCMerced_LandUse-2/Images'


train_write = os.path.join(To_Dataset, 'train')
os.mkdir(train_write)
test_write = os.path.join(To_Dataset, 'test')
os.mkdir(test_write)
category = [i for i in os.listdir(From_Dataset) if not i.startswith('.')]
print(category)
print("类别数:", len(category))
print(category)

ratio = 0.8

for classname in category:
    train_class_dir = os.path.join(train_write, classname)
    os.mkdir(train_class_dir)
    test_class_dir = os.path.join(test_write, classname)
    os.mkdir(test_class_dir)

    dir_path = os.path.join(From_Dataset, classname)
    images = [i for i in os.listdir(dir_path) if not i.startswith('.')]
    train_num = int(len(images)*ratio)
    shuffle(images)

    train_image = images[:train_num]
    test_image = images[train_num:]
    for image in train_image:
        from_filepath = os.path.join(dir_path, image)
        to_filepath = os.path.join(train_class_dir, image)
        shutil.copyfile(src=from_filepath, dst=to_filepath)

    for image in test_image:
        from_filepath = os.path.join(dir_path, image)
        to_filepath = os.path.join(test_class_dir, image)
        shutil.copyfile(src=from_filepath, dst=to_filepath)



