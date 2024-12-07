input_file = input("Please key in your input filepath: ")
output_file = input("Please key in your output filepath: ")
truncate_len = input("Please key in a number beyond which the sequence will be truncated: ")

with open(input_file, "r") as in_f, open(output_file, "w") as out_f:
	for line in in_f.readlines():
		seq = line.split()
		seq_len = len(seq)
		if seq_len > int(truncate_len):
			seq = seq[:int(truncate_len)]
		out_f.write(" ".join(seq)+'\n')
