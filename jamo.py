#constants

SBase = 0xAC00
LBase = 0x1100
VBase = 0x1161
TBase = 0x11A7
LCount = 19
VCount = 21
TCount = 28
NCount = 588 
SCount = 11172

def decompose_character(char, final_char = False):
    char = ord(char)
    SIndex = char - SBase
    if (SIndex < 0 or SIndex >= SCount):
        return [chr(char)]
    
    result = []
    L = int(LBase + SIndex / NCount)
    V = int(VBase + (SIndex % NCount) / TCount)
    T = int(TBase + SIndex % TCount)
    result.append(chr(L))
    result.append(chr(V))
    if final_char:
        result.append(chr(T))
    elif (T != TBase):
        result.append(chr(T))
    return result