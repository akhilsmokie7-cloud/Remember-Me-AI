#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emoji Cube Encoding: Trinary Compression Layer
==============================================
Maps 9-trit segments (19,683 unique states) to emojis or multi-emoji clusters.
Based on the 10x10x10 Emoji Cube specification.

Logic:
1 byte = 8 bits = 256 states
1 tryte = 6 trits = 729 states (3x more than byte)
1 emoji = 9 trits = 19,683 states
"""

import math
from typing import List
from ..math.trinary import TrinaryArithmetic

class EmojiEncoder:
    """
    Handles conversion between Trinary Words and Emoji representations.
    """
    # ⚡ Systematic Emoji Mapping (Simplified for stability)
    # We use a base set and offset to generate deterministic high-range characters
    BASE_EMOJI_OFFSET = 0x1F600 # Start of Emoticons block
    
    @staticmethod
    def trits_to_emoji(trits_9: List[int]) -> str:
        """
        Converts a 9-trit list to a single emoji-like character.
        If trits_9 < 9, it pads with 0.
        """
        # Ensure 9 trits
        padded = trits_9[:9] + [0] * (9 - len(trits_9))
        
        # Calculate index (0 to 19,682)
        idx = 0
        for i, t in enumerate(padded):
            # Map -1,0,1 to 0,1,2 for base-3 indexing
            base3_digit = t + 1 
            idx += base3_digit * (3 ** i)
            
        # Map to Unicode Emoji range
        # Total range is roughly 19k. We use the private use area or 
        # systematic offsets if standard blocks are full.
        char_code = EmojiEncoder.BASE_EMOJI_OFFSET + idx
        return chr(char_code)

    @staticmethod
    def emoji_to_trits(emoji: str) -> List[int]:
        """
        Converts a single emoji char back to 9 trits.
        """
        char_code = ord(emoji)
        idx = char_code - EmojiEncoder.BASE_EMOJI_OFFSET
        
        trits = []
        n = idx
        for _ in range(9):
            rem = n % 3
            # Map 0,1,2 back to -1,0,1
            trits.append(rem - 1)
            n //= 3
            
        return trits

class SpaceTimeCompressor:
    """
    Compresses data into Emoji Cube Space.
    10x10x10 cube layout.
    """
    def __init__(self):
        self.cube_dim = 10
        
    def pack_vector(self, vector: List[float]) -> str:
        """
        Compresses a float vector into a string of emojis.
        Each float is quantized to trits first.
        """
        # 1. Quantization: Float -> -1, 0, 1
        # Simple thresholding for demonstration
        trits = [1 if x > 0.3 else (-1 if x < -0.3 else 0) for x in vector]
        
        # 2. Chunking: 9 trits per emoji
        result = ""
        for i in range(0, len(trits), 9):
            chunk = trits[i:i+9]
            result += EmojiEncoder.trits_to_emoji(chunk)
            
        return result

    def unpack_vector(self, encoded: str) -> List[int]:
        """
        Unpacks emoji string back to trit vector.
        """
        all_trits = []
        for char in encoded:
            all_trits.extend(EmojiEncoder.emoji_to_trits(char))
        return all_trits

if __name__ == "__main__":
    # Test
    v = [0.9, -0.1, -0.8, 0.4, 0.0, 0.1, -0.6, 0.7, 0.8] # 9 values
    compressor = SpaceTimeCompressor()
    encoded = compressor.pack_vector(v)
    print(f"Vector: {v}")
    print(f"Emoji Encoded: {encoded}")
    
    decoded = compressor.unpack_vector(encoded)
    print(f"Decoded Trits: {decoded}")
    
    # Check first few
    # 0.9 -> 1, -0.1 -> 0, -0.8 -> -1
    assert decoded[:3] == [1, 0, -1]
    print("Compression test passed.")
