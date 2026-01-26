#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trinary Logic Engine: Balanced Ternary (-1, 0, 1)
================================================
Implements the core arithmetic and state representation for the
Trinary Space-Time Compiler Architecture.

States:
-1 : NEG (Negative) / PAST   / LOCAL
 0 : NULL (Neutral) / PRESENT / SHARED
 1 : POS (Positive) / FUTURE  / REMOTE
"""

import math
from typing import List, Union

class Trit:
    """Represents a single Balanced Ternary unit."""
    NEG = -1
    NULL = 0
    POS = 1

    @staticmethod
    def add(a: int, b: int, carry: int = 0) -> (int, int):
        """
        Balanced Ternary Addition.
        Returns (sum, carry)
        """
        s = a + b + carry
        if s > 1:
            return (s - 3, 1)
        if s < -1:
            return (s + 3, -1)
        return (s, 0)

    @staticmethod
    def mul(a: int, b: int) -> int:
        """Balanced Ternary Multiplication."""
        return a * b

class TrinaryArithmetic:
    """High-level arithmetic for lists of trits (Trytes/Words)."""
    
    @staticmethod
    def add_trits(trits_a: List[int], trits_b: List[int]) -> List[int]:
        """Adds two balanced ternary arrays (little-endian)."""
        max_len = max(len(trits_a), len(trits_b))
        a = trits_a + [0] * (max_len - len(trits_a))
        b = trits_b + [0] * (max_len - len(trits_b))
        
        result = []
        carry = 0
        for i in range(max_len):
            _sum, carry = Trit.add(a[i], b[i], carry)
            result.append(_sum)
        
        if carry != 0:
            result.append(carry)
        return result

    @staticmethod
    def to_int(trits: List[int]) -> int:
        """Converts balanced ternary (little-endian) to decimal integer."""
        val = 0
        for i, t in enumerate(trits):
            val += t * (3 ** i)
        return val

    @staticmethod
    def from_int(n: int) -> List[int]:
        """Converts decimal integer to balanced ternary (little-endian)."""
        if n == 0:
            return [0]
        
        trits = []
        is_neg = n < 0
        n = abs(n)
        
        while n > 0:
            rem = n % 3
            if rem == 2:
                trits.append(-1)
                n = (n // 3) + 1
            else:
                trits.append(rem)
                n = n // 3
                
        if is_neg:
            trits = [-t for t in trits]
            
        return trits

class TemporalState:
    """Mapping for Space-Time Layers."""
    PAST = -1
    PRESENT = 0
    FUTURE = 1
    
    @staticmethod
    def describe(state: int) -> str:
        mapping = { -1: "PAST/LOCAL", 0: "PRESENT/SHARED", 1: "FUTURE/REMOTE" }
        return mapping.get(state, "UNKNOWN")

if __name__ == "__main__":
    # Self-test
    print("Trinary Logic Test:")
    a = 10
    b = 5
    trits_a = TrinaryArithmetic.from_int(a)
    trits_b = TrinaryArithmetic.from_int(b)
    print(f"{a} in trinary: {trits_a}")
    print(f"{b} in trinary: {trits_b}")
    
    sum_trits = TrinaryArithmetic.add_trits(trits_a, trits_b)
    print(f"Sum in trinary: {sum_trits}")
    print(f"Sum in decimal: {TrinaryArithmetic.to_int(sum_trits)}")
    
    assert TrinaryArithmetic.to_int(sum_trits) == 15
    print("Arithmetic check passed.")
