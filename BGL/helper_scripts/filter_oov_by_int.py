import pandas as pd
import os

print(f"This gets the max token value from the 'train' file and removes all observations containing tokens above this max from the 'test_abnormal' and 'test_normal' files. It then writes these to new files - 'test_abnormal_filtered' and 'test_normal_filtered'.")
input_folder = input("Please input the folder path containing 'train', 'test_normal' and 'test_abnormal'.")

train_path = os.path.join(input_folder, 'train')
test_normal_path = os.path.join(input_folder, 'test_normal')
test_abnormal_path = os.path.join(input_folder, 'test_abnormal')
output_test_normal_path = os.path.join(input_folder, 'test_normal_filtered')
output_test_abnormal_path = os.path.join(input_folder, 'test_abnormal_filtered')


df = pd.read_csv(train_path, header=None)
all_numbers = df[0].str.split(expand=True).stack().astype(int)  # Split strings into individual numbers
max_value = all_numbers.max()
test_norm = pd.read_csv(test_normal_path, header=None)
test_abnorm = pd.read_csv(test_abnormal_path, header=None)
print(f"Before filtering:\n Number of lines for test_norm: {test_norm.shape[0]}\n Number of lines for test_abnormal: {test_abnorm.shape[0]}\n")
norm_filtered = test_norm[test_norm[0].apply(lambda x: all(int(num) <= max_value for num in x.split()))]
abnorm_filtered = test_abnorm[test_abnorm[0].apply(lambda x: all(int(num) <= max_value for num in x.split()))]

print(f"After filtering:\n Number of lines for test_norm: {norm_filtered.shape[0]}\n Number of lines for test_abnormal: {abnorm_filtered.shape[0]}")
 
norm_filtered.to_csv(output_test_normal_path, index=False, header=False)
abnorm_filtered.to_csv(output_test_abnormal_path, index=False, header=False)