import numpy as np
from PIL import Image

#UTILITY FUNCTIONS
def group_bits(bits_array):
    bits_array= np.array(bits_array)
    grouped_bits = np.char.add(bits_array[::2].astype(str),bits_array[1::2].astype(str))
    return grouped_bits

#DATA TYPE: Image
def image_to_bits(path):
    img =Image.open(path).convert("RGB")
    img_array = np.array(img)
    img_shape = img_array.shape
    bit_array = np.unpackbits(img_array.astype(np.uint8))
    return bit_array, img_shape

def bits_to_image(bit_array,shape):
    bit_array = np.array(bit_array,dtype=np.uint8)
    byte_array = np.packbits(bit_array)
    img_array = byte_array.reshape(shape)
    return img_array

def insert_array_at_index(arr, insert_arr, index):
    # Ensure the index is within the range
    if index > len(arr):
        raise IndexError("Index is out of bounds")
    # Insert the array using list slicing
    return np.concatenate([arr[:index], insert_arr, arr[index:]])

def check_bits_alteration(original, extracted):
    counter=0
    for i in range(len(original)):
        if(int(original[i])!=int(extracted[i])):
            print(f"Mismatch location: {i}")
            counter=counter+1
    print(f"Number of mismatched bits:{counter}")