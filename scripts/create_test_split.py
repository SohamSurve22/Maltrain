import os
import shutil
import random

SOURCE_DIR = "../dataset/train"
TEST_DIR = "../dataset/test"

TEST_RATIO = 0.1   # 10% for test

os.makedirs(TEST_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_name)
    test_class_path = os.path.join(TEST_DIR, class_name)

    os.makedirs(test_class_path, exist_ok=True)

    images = os.listdir(class_path)

    test_count = int(len(images) * TEST_RATIO)

    test_images = random.sample(images, test_count)

    for img in test_images:

        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_path, img)

        shutil.move(src, dst)

print("Test dataset created successfully.")