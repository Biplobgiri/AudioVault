import numpy as np
from config import *
import pywt
import soundfile as sf
from utils import *
from header import *

# EMBEDDING UTILITY FUNCTIONS
def modify_coefficent(coeff,embed_value,delta,two_bit_qim):
    dither = 0

    if(two_bit_qim):
        possible_dithers = [i*delta/16 for i in range(1,8,2)]

        if(embed_value == "00"): 
            dither = possible_dithers[0]
        elif (embed_value == "01"):
            dither = possible_dithers[1]
        elif (embed_value == "10"):
            dither = possible_dithers[2]
        elif (embed_value == "11"):
            dither = possible_dithers[3]
    
    else:
        possible_dithers = [i*delta/8 for i in range(1,4,2)]
        
        if(embed_value == 0):
            dither = possible_dithers[0]
        else:
            dither = possible_dithers[1]
    
    modified_coeff = delta * np.round((coeff-dither)/delta) + dither
    return modified_coeff

def apply_dwt(audio_path, wavelet, dwt_level, payload_size):
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

    # Determine frame size
    frame_size = FRAME_SIZE

    # Split into frames
    num_frames = len(audio_left) // frame_size
    frames_left = [audio_left[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)]
    frames_right = [audio_right[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)]


    print(f"TOTAL FRAMES: {len(frames_left)}")
    # Apply DWT independently to each frame
    dwt_coeffs_left = []
    dwt_coeffs_right=[]

    frames_sync_values_left=[]
    for frame in frames_left:
        frames_sync_values_left.append(frame[:8])
        dwt_coeffs_left.append(pywt.wavedec(frame[8:], wavelet, level=dwt_level, mode="periodization"))


    frames_sync_values_right=[]
    for frame in frames_right:
        frames_sync_values_right.append(frame[:8])
        dwt_coeffs_right.append(pywt.wavedec(frame[8:], wavelet, level=dwt_level, mode="periodization"))

    return dwt_coeffs_left, dwt_coeffs_right, frames_sync_values_left, frames_sync_values_right,sr

def insert_sync_bits(sync_frame, sync_bits):
    sync_bits = group_bits(sync_bits)
    for i,sync_val in enumerate(sync_frame):
        sync_frame[i] = modify_coefficent(sync_val,sync_bits[i],delta=DELTA, two_bit_qim=TWO_BIT_QIM)
    
    return sync_frame

def split_data(payload_array, payload_distribution):
    start_index = 0
    result = []

    for size in payload_distribution:
        end_index = start_index + size
        result.append(payload_array[start_index:end_index])
        start_index = end_index  # Update the start index for the next slice
    return result

def generate_embedding_location(coeffs_array,payload_size,seed):
    embedding_location = []
    np.random.seed(seed)
    total_elements = sum(len(level_coeffs) for level_coeffs in coeffs_array)
    payload_distribution_len = [int(len(level_coeffs) / total_elements * payload_size) for level_coeffs in coeffs_array]
    payload_distribution_len[0] += payload_size - sum(payload_distribution_len)

    for i, level_coeffs in enumerate(coeffs_array):
        embedding_location.append(np.random.choice(
            len(level_coeffs),
            size=payload_distribution_len[i],
            replace=False
        ))
    
    return embedding_location,payload_distribution_len

# Function to reconstruct frames from their DWT coefficients
def reconstruct_frames(frames_dwt_coeffs, wavelet):
    reconstructed_frames = []
    for frame_coeffs in frames_dwt_coeffs:
        # Reconstruct the time-domain signal for this frame
        frame_signal = pywt.waverec(frame_coeffs, wavelet=wavelet, mode="periodization")
        reconstructed_frames.append(frame_signal)
    return reconstructed_frames

#MAIN EMBEDDING FLOW
def embed_data(secret_data, files, audio_path, stego_path):
    header_bits = prepare_header(version=1,files_list=files,two_bit_qim=1)
    print(f"HEADER BITS: {header_bits}")

    # CHECK WHETHER HEADER IS CREATED PROPERLY OR NOT
    decoded = decode_header(header_bits)
    print(f"DECODED HEADER: {decoded}")

    if(TWO_BIT_QIM):
        secret_data = group_bits(secret_data)
        header_bits = group_bits(header_bits)

    frames_dwt_coeffs_left, frames_dwt_coeffs_right,frames_sync_values_left,frames_sync_values_right, sr = apply_dwt(audio_path=audio_path, wavelet=WAVELET, dwt_level=DWT_LEVEL,payload_size=len(secret_data) + len(header_bits))

    modified_frames_dwt_coeffs_left = []
    modified_frames_dwt_coeffs_right = []

    for frame_left, frame_right in zip(frames_dwt_coeffs_left, frames_dwt_coeffs_right):
        dwt_coeffs_left = frame_left
        dwt_coeffs_right = frame_right

        detail_coeffs_left = dwt_coeffs_left[1:]
        detail_coeffs_right = dwt_coeffs_right[1:]

        possible_locations = sum([level.shape[0] for level in detail_coeffs_left]) * 2
        payload_length = len(secret_data)

        print(f"Payload Capacity: {possible_locations} and Payload Length: {payload_length}")
        if payload_length > possible_locations:
            raise ValueError(f"Payload length ({payload_length}) exceeds available valid embedding locations ({possible_locations}).")

        # Inject the header directly into first-level detail coefficients
        for i, value in enumerate(header_bits):
            detail_coeffs_left[0][i] = modify_coefficent(
                coeff=detail_coeffs_left[0][i],
                embed_value=value,
                delta=DELTA,
                two_bit_qim=TWO_BIT_QIM
            )

        detail_coeffs_header_part = detail_coeffs_left[0][:len(header_bits)]

        mid = int(payload_length / 2)
        secret_data_left = secret_data[mid:]
        secret_data_right = secret_data[:mid]
        # Payload is injected in remaining bits
        detail_coeffs_left[0] = detail_coeffs_left[0][len(header_bits):]

        embed_location_left, payload_dist_left_len = generate_embedding_location(detail_coeffs_left, len(secret_data_left), SEED_LEFT)
        payload_dist_left = split_data(secret_data_left, payload_dist_left_len)

        embed_location_right, payload_dist_right_len = generate_embedding_location(detail_coeffs_right, len(secret_data_right), SEED_RIGHT)
        payload_dist_right = split_data(secret_data_right, payload_dist_right_len)

        for i, (level_location_left, level_location_right) in enumerate(zip(embed_location_left, embed_location_right)):
            for j, (loc_left, loc_right) in enumerate(zip(level_location_left, level_location_right)):
                detail_coeffs_left[i][loc_left] = modify_coefficent(
                    coeff=detail_coeffs_left[i][loc_left],
                    embed_value=payload_dist_left[i][j],
                    delta=DELTA,
                    two_bit_qim=TWO_BIT_QIM
                )
                detail_coeffs_right[i][loc_right] = modify_coefficent(
                    coeff=detail_coeffs_right[i][loc_right],
                    embed_value=payload_dist_right[i][j],
                    delta=DELTA,
                    two_bit_qim=TWO_BIT_QIM
                )

        detail_coeffs_left[0] = np.concatenate([detail_coeffs_header_part, detail_coeffs_left[0]])

        modified_frames_dwt_coeffs_left.append([dwt_coeffs_left[0]] + detail_coeffs_left)
        modified_frames_dwt_coeffs_right.append([dwt_coeffs_right[0]] + detail_coeffs_right)

        # Reconstruct each frame separately
        reconstructed_frames_left = reconstruct_frames(modified_frames_dwt_coeffs_left, WAVELET)
        reconstructed_frames_right = reconstruct_frames(modified_frames_dwt_coeffs_right, WAVELET)

        # Now concatenate the frames in the time domain
        stego_audio_left = np.concatenate(reconstructed_frames_left)
        stego_audio_right = np.concatenate(reconstructed_frames_right)

        frame_size = FRAME_SIZE

    for i,sync_portion in enumerate(frames_sync_values_left):
        sync_portion = insert_sync_bits(sync_frame=sync_portion, sync_bits=SYNC_BITS)
        stego_audio_left = insert_array_at_index(stego_audio_left, sync_portion, i*frame_size)
        stego_audio_right = insert_array_at_index(stego_audio_right,frames_sync_values_right[i], i*frame_size)
        
    # #Insert the Sync Bits
    # stego_audio_left = insert_sync_bits(audio=stego_audio_left,interval=len(stego_audio_left)//4,sync_bits=SYNC_BITS)
    # stego_audio_right = insert_sync_bits(audio=stego_audio_right,interval=len(stego_audio_right)//4,sync_bits=SYNC_BITS)

    min_len = min(len(stego_audio_left), len(stego_audio_right))
    reconstructed_stereo = np.column_stack((stego_audio_left[:min_len], stego_audio_right[:min_len]))

    sf.write(stego_path, reconstructed_stereo, sr)



