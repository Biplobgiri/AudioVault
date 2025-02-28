import numpy as np

#HEADER UTILITIES
def prepare_header(version, files_list,two_bit_qim):
    """
    Header fields:
      - Version:        8 bits
      - Header Length:  8 bits -> computed as total header length in bits
      - File Count:     4 bits
    
    For each file:
      - File Type:      4 bytes (32 bits)
      - If an image (jpg/jpeg/png): 
                          Image metadata is stored in 26 bits, with:
                             * Height: 12 bits
                             * Width:  12 bits
                             * Channels: 2 bits
      - Payload Length: 32 bits
    
    Finally:
      - Decision Flag:  1 bit
    
    Returns:
      header_bits:  A list of bits (0/1) representing the header.
    """
    header_bits = []

    header_bits.extend(format(version,"08b"))
    header_bits.extend("00000000")  # placeholder for header length (8 bits)

    file_count = len(files_list) & 0x0F
    header_bits.extend(format(file_count,"04b"))

    for file in files_list:
        file_type = file["file_type"].ljust(4, " ")[:4].encode("ascii")
        for byte in file_type:
            header_bits.extend(format(byte, '08b'))

        # If file is an image, add image metadata
        if file["file_type"].strip().lower() in ["jpg", "jpeg", "png"]:
            height = file["height"]
            width = file["width"]
            channels = file["channels"]
            img_metadata = (height << 14) | (width << 2) | (channels & 0x3)
            header_bits.extend(format(img_metadata, '026b'))
        
        payload_length = file["payload_length"]
        header_bits.extend(format(payload_length, '032b'))
    
    header_bits.extend(format(two_bit_qim & 0x01, "01b"))

    computed_header_len = len(header_bits)
    if(computed_header_len % 2 !=0 ):
        #Add padding bit
        header_bits.append("0")
        computed_header_len +=1

    header_bits[8:16] = format(computed_header_len,"08b")

    return header_bits


def decode_header(header_bits):
    header = {}
    header['version'] = int("".join(header_bits[0:8]), 2)
    header['header_length'] = int("".join(header_bits[8:16]), 2)
    header['file_count'] = int("".join(header_bits[16:20]), 2)
    current_pos = 20
    files = []
    for _ in range(header['file_count']):
        file = {}
        file_type_bits = "".join(header_bits[current_pos:current_pos+32])
        file_type_bytes = bytearray(int(file_type_bits[i:i+8], 2) for i in range(0, 32, 8))
        file['file_type'] = file_type_bytes.decode("ascii").strip()
        current_pos += 32
        if file['file_type'].lower() in ["jpg", "jpeg", "png"]:
            img_metadata_bits = "".join(header_bits[current_pos:current_pos+26])
            img_metadata = int(img_metadata_bits, 2)
            file['height'] = (img_metadata >> 14) & 0xFFF
            file['width'] = (img_metadata >> 2) & 0xFFF
            file['channels'] = img_metadata & 0x3
            current_pos += 26
        payload_length_bits = "".join(header_bits[current_pos:current_pos+32])
        file['payload_length'] = int(payload_length_bits, 2)
        current_pos += 32
        files.append(file)
    header['two_bit_qim'] = int("".join(header_bits[current_pos:current_pos+1]), 2)
    header['files'] = files
    return header
