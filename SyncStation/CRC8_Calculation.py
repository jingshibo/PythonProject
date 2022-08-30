import math

def CRC8(Vector, Len):
    crc = 0
    j = 0

    while Len > 0:
        Extract = Vector[j]
        for _ in range(8):

            Sum = (crc % 2) ^ (Extract % 2)
            crc = math.floor(crc / 2)

            if (Sum > 0):
                binlist = 8 * [0]
                a = bin(crc)
                b = bin(140)
                if len(a) < 10:
                    a = a[:2] + '0' * (10 - len(a)) + a[2:]
                for k in range(2, 10, 1):
                    binlist[k - 2] = int(not ((a[k] == b[k])))

                num = int("".join(map(str, binlist)))  # convert a list of number to a single integer (binary)
                crc = int(str(num), 2)  # convert binary to int

            Extract = math.floor(Extract / 2)

        Len = Len - 1
        j = j + 1

    return crc
