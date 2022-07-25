from sklearn.model_selection import train_test_split
import os
import argparse
import json

parser = argparse.ArgumentParser(description='Split Shoe Dataset into train-dev-test set')
parser.add_argument('--image_folder', default='./Dataset', type=str, 
                    help='Image Folder Path')
parser.add_argument('--shk_folder', type=str, default='./Dataset/shapeKeys', 
                    help='Shape Key Folder Path')
parser.add_argument('--dump_folder', type=str, default='./Dataset', help='Path to save splits json')
parser.add_argument('--test_size', default=0.1, type=float, 
                    help='Test Set split size in fraction (eg. 0.1 for 10%)')
parser.add_argument('--dev_size', default=0.1, type=float, 
                    help='Dev set split size in fraction (eg. 0.1 for 10%)')
parser.add_argument('--seed', type=int, default=2021, help='Randomization seed')

opt = parser.parse_args()

image_folder = opt.image_folder
json_path = opt.shk_folder

def get_complete_path(img_folder, sk_folder):
    img_paths = []
    shk_paths = []
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                sk_path = os.path.join(sk_folder, (os.path.splitext(file)[0] + '.json'))
                img_paths.append(img_path)
                shk_paths.append(sk_path)
                
    return img_paths, shk_paths

def dump_set(data, filename):
    with open( os.path.join(dump_folder, filename), 'w') as f:
        json.dump(data, f)

shoe_images, shape_key_jsons =  get_complete_path(image_folder, json_path)

print("Dataset Size:")
print(f" num_images: {len(shoe_images)}")
print(f" num_shapekeys: {len(shape_key_jsons)}")

print("<========== Splitting the dataset into train and test set ==========>")
X_train, X_test, y_train, y_test = train_test_split(shoe_images, shape_key_jsons, 
                                                    test_size=opt.test_size, random_state=opt.seed)
print(f'Train set size:\n X: {len(X_train)}, y: {len(y_train)}')
print(f'Test set size:\n X: {len(X_test)}, y: {len(y_test)}')

print("<========== Splitting the train set into final training set and validation set ==========>")
X_train_f, X_dev, y_train_f, y_dev = train_test_split(X_train, y_train,
                                                      test_size=opt.dev_size, random_state=opt.seed)
print(f'Train set size:\n X: {len(X_train_f)}, y: {len(y_train_f)}')
print(f'Validation set size:\n X: {len(X_dev)}, y: {len(y_dev)}')

Training_set = {'shoe_images': X_train_f,
                'shape_keys': y_train_f}
Dev_set = {'shoe_images': X_dev,
           'shape_keys': y_dev}
Test_set = {'shoe_images': X_test,
            'shape_keys': y_test}

dump_folder = os.path.join(opt.dump_folder,'dataset_splits')
if not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

print(f"<========== Saving splits in {dump_folder} ==========>")
dump_set(Training_set, 'training.json')
dump_set(Dev_set, 'dev.json')
dump_set(Test_set, 'test.json')

