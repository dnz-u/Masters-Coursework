from PIL import Image
import numpy as np

im_name = "internet_photo.png"
img = Image.open(im_name)
img_gray = img.convert('L')
img_array = np.array(img_gray, dtype=np.uint8)

file_name = im_name[:-3] + "txt"

with open(file_name, 'w') as outfile:
    outfile.write(str(img_array.shape[0])+"\n")
    outfile.write(str(img_array.shape[1])+"\n")

    for row in img_array:
        for n in row:
            outfile.write(str(n)+ " ")
        outfile.write("\n")
        
print("gray image size:", img_array.shape)
