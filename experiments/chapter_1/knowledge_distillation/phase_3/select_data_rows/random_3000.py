import csv
import sys
import random

csv.field_size_limit(sys.maxsize)

input_csv = '/home/malitha/programming/vlm-research/Datasets/RSCID/train.csv'
output_csv = 'random_3000_rows.csv'
num_samples = 3000

with open(input_csv, newline='', encoding='utf-8') as infile:
    reader = list(csv.reader(infile))
    
    header = reader[0]
    rows = reader[1:]
    
    sampled_rows = random.sample(rows, min(num_samples, len(rows)))
    
with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    writer.writerows(sampled_rows)

print(f"Saved {len(sampled_rows)} random rows to {output_csv}")
