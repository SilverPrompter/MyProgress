def bitwiseComplement(self, n: int) -> int:
    bits = bin(n)[2:]
    result = ""
    for x in bits:
        if x == "1":
            result += "0"
        else:
            result += "1"
    return int(result, 2)
