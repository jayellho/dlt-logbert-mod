import pandas as pd
import os
 
print(f"This gets the list of tokens from the 'train' file and removes sequences that have >=1 tokens that are not in this list from the 'test_abnormal' and 'test_normal' files. It then writes these to new files - 'test_abnormal_filtered' and 'test_normal_filtered'.")
input_folder = input("Please input the folder path containing 'train', 'test_normal' and 'test_abnormal'.")
 
train_path = os.path.join(input_folder, 'train')
test_normal_path = os.path.join(input_folder, 'test_normal')
test_abnormal_path = os.path.join(input_folder, 'test_abnormal')
output_test_normal_path = os.path.join(input_folder, 'test_normal_filtered')
output_test_abnormal_path = os.path.join(input_folder, 'test_abnormal_filtered')
 
 
df = pd.read_csv(train_path, header=None)
valid_hex_set = df[0].str.split(expand=True).stack().apply(lambda x: int(x, 16)).unique()  # Split strings into hexa
valid_hex_set = set(valid_hex_set)
test_norm = pd.read_csv(test_normal_path, header=None)
test_abnorm = pd.read_csv(test_abnormal_path, header=None)
print(f"Before filtering:\n Number of lines for test_norm: {test_norm.shape[0]}\n Number of lines for test_abnormal: {test_abnorm.shape[0]}\n")

# Filtering normal data: check if each token in the row is in the valid hex set (consistent behavior)
norm_filtered = test_norm[test_norm[0].apply(
    lambda x: all(int(token, 16) in valid_hex_set for token in x.split())  # Drop line if any token is not valid
)]

# Filtering abnormal data: check if each token in the row is in the valid hex set (consistent behavior)
abnorm_filtered = test_abnorm[test_abnorm[0].apply(
    lambda x: all(int(token, 16) in valid_hex_set for token in x.split())  # Drop line if any token is not valid
)] 

print(f"After filtering:\n Number of lines for test_norm: {norm_filtered.shape[0]}\n Number of lines for test_abnormal: {abnorm_filtered.shape[0]}")
norm_filtered.to_csv(output_test_normal_path, index=False, header=False)
abnorm_filtered.to_csv(output_test_abnormal_path, index=False, header=False)
