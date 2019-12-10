def performance_gain(reses):
    for p, res in reses.items():
        gains = []
        for i in range(len(res)):
            if i == 0:
                continue
            gains.append(res[i] - res[0])
        print(p, max(gains), min(gains))


def performance_gain_average():
    pass


if __name__ == '__main__':
    performance_gain_average()
