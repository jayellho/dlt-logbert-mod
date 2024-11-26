file = input("Please input the filepath to the file to get max tokens from.")
with open (file) as f:
	max_token = 0
	counter = 0
	max_seq_length = 0
	sum_seq_length = 0
	count_more_than_512 = 0
	for line in f.readlines():
		counter += 1
		curr_seq_length = len(line.split())
		if curr_seq_length > 512:
			count_more_than_512 += 1
		sum_seq_length += curr_seq_length
		max_seq_length = max(len(line.split()), max_seq_length)
#		if max_seq_length == 47504:
#			print(f"test print of line split: {line.split()}")
#			break
		for i in line.split():
			if int(i) > max_token:
				max_token = int(i)
	print(f"average seq length = {sum_seq_length / counter}")
	print(f"max_token for {file} = {max_token}")
	print(f"counter = {counter}")
	print(f"max_seq_length = {max_seq_length}")
	print(f"count_more_than_512 = {count_more_than_512}")
