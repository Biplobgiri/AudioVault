import numpy as np
import pywt
import soundfile as sf
from config import *
from utils import *
from header import *
from embedding import generate_embedding_location

def extract_secret_value(stego_coeff, two_bit_qim, delta):
    dithers = []
    possible_values = []

    if(two_bit_qim):
        dithers = [i*delta/16 for i in range(1,8,2)]
        possible_values=["00","01","10","11"]
    else:
        possible_values=["0","1"]

    q = [(delta * np.round((stego_coeff - dither)/delta) + dither) for dither in dithers]
    error_value = [abs(stego_coeff - q_value) for q_value in q]
    min_index = error_value.index(min(error_value))

    return possible_values[min_index]

def get_synced_start(audio_path, wavelet, dwt_level):
    audio_stereo, sr = sf.read(audio_path)
    audio_stereo = audio_stereo.T

    audio_left = audio_stereo[0]
    audio_right = audio_stereo[1]

    if len(audio_left) % 2 != 0:
        audio_left = audio_left[:-1]
        audio_right = audio_right[:-1]

    max_val = np.max(np.abs(audio_left))
    audio_left /= max_val
    audio_right /= max_val

    bits_accumulated = []
    start_i = -1
    end_i = 0
    sync_bits = group_bits(SYNC_BITS)

    for i, coeff in enumerate(audio_left):
        bit = extract_secret_value(coeff, two_bit_qim=TWO_BIT_QIM, delta=DELTA)
        bits_accumulated.append(bit)

        if len(bits_accumulated) > len(sync_bits):
            bits_accumulated.pop(0)

        if np.array_equal(bits_accumulated, sync_bits):
            print("Sync Bits Matched at index:", i)
            end_i = i + 1
            break

    if end_i == 0:
        raise ValueError("Sync bits not found in the audio.")

    frame_end = end_i + FRAME_SIZE - len(sync_bits)
    
    if frame_end > len(audio_left):
        raise ValueError("Frame size exceeds audio length after sync detection.")

    dwt_coeffs_left = pywt.wavedec(audio_left[end_i:frame_end], wavelet, level=dwt_level, mode="periodization")
    dwt_coeffs_right = pywt.wavedec(audio_right[end_i:frame_end], wavelet, level=dwt_level, mode="periodization")

    return dwt_coeffs_left, dwt_coeffs_right

#MAIN EXTRACTION FLOW
def extract_data(audio_path,extracted_data_path):
    stego_dwt_coeffs_left, stego_dwt_coeffs_right=get_synced_start(audio_path="stego.wav",wavelet=WAVELET,dwt_level=DWT_LEVEL)

    stego_detail_coeffs_left = stego_dwt_coeffs_left[1:] #Gives an array of array of different level of detail coeffs
    stego_detail_coeffs_right = stego_dwt_coeffs_right[1:]
    #Stego Detail Coeffs left of level zero contains header
    #In the index 8 to 15

    header_len_coeffs = stego_detail_coeffs_left[0][4:8]
    header_len_bits = []
    for _,coeff in enumerate(header_len_coeffs):
        header_len_bits.append(extract_secret_value(coeff,two_bit_qim=TWO_BIT_QIM,delta=DELTA))

    header_length = int("".join(map(str, header_len_bits)), 2)
    header_coeffs = stego_detail_coeffs_left[0][:header_length//2]
    header_bits=[]

    for _,coeff in enumerate(header_coeffs):
        header_bits.append(extract_secret_value(coeff,two_bit_qim=TWO_BIT_QIM, delta=DELTA))

    if(TWO_BIT_QIM):
        header_bits = [bit for pair in header_bits for bit in pair]

    print("------CHECKING HEADER BITS - EXTRACTION----------")
    print(header_bits)
    print("------END OF CHECKING HEADER BITS - EXTRACTION----------")

    # CHECK WHETHER HEADER IS CREATED PROPERLY OR NOT
    header = decode_header(header_bits)
    print("------CHECKING DECODED HEADER - EXTRACTION----------")
    print(header)
    print("------END OF CHECKING DECODED HEADER - EXTRACTION----------")

    #Remove the header coeff part and extract the payload
    stego_detail_coeffs_left[0] = stego_detail_coeffs_left[0][header_length//2:]

    #group_factor
    if(TWO_BIT_QIM):
        group_factor = 2
    else:
        group_factor = 1

    stego_payload_len = int((header["files"][0]["payload_length"])/group_factor)
    payload_left= int(stego_payload_len/2)
    payload_right = stego_payload_len - payload_left

    embed_location_left,payload_dist_left_len = generate_embedding_location(stego_detail_coeffs_left,payload_left,SEED_LEFT)

    embed_location_right, payload_dist_right_len = generate_embedding_location(stego_detail_coeffs_right,payload_right, SEED_RIGHT)

    extracted_data_left = []
    extracted_data_right = []

    # Extract payload from left channel detail coefficients using list comprehension
    extracted_data_left = [
        extract_secret_value(stego_detail_coeffs_left[i][loc], two_bit_qim=TWO_BIT_QIM, delta=DELTA)
        for i in range(len(embed_location_left))
        for loc in embed_location_left[i]
    ]

    # Extract payload from right channel detail coefficients similarly
    extracted_data_right = [
        extract_secret_value(stego_detail_coeffs_right[i][loc], two_bit_qim=TWO_BIT_QIM, delta=DELTA)
        for i in range(len(embed_location_right))
        for loc in embed_location_right[i]
    ]

    extracted_data = extracted_data_right + extracted_data_left
    print(f"Extracted Data Length:{len(extracted_data)}")
    if(TWO_BIT_QIM):
        extracted_data = [bit for pair in extracted_data for bit in pair] #Ungroup the bits
    
    img_metadata = header["files"][0]
    extracted_image = bits_to_image(extracted_data, (img_metadata["height"],img_metadata["width"],img_metadata["channels"]))
    img = Image.fromarray(extracted_image)
    img.save(extracted_data_path)

    return extracted_data


