from PIL import Image
import numpy as np
import sys

input_file = sys.argv[1]

with open(input_file, 'r') as f:
    lines = f.read().strip().split('\n')

rows = int(lines[0])
cols = int(lines[1])

pixels = [[int(x) for x in line.split()] for line in lines[2:]]

image = Image.fromarray(np.array(pixels, dtype=np.uint8), mode='L')

output_file = input_file.rsplit('.', 1)[0] + '_output.jpg'

image.save(output_file)
print(f"Image saved as '{output_file}'")