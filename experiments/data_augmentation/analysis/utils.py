# from experiments.data_augmentation.analysis.plot_Offensive import results as result_oe
# from experiments.data_augmentation.analysis.plot_Stance import results as result_st
# from experiments.data_augmentation.analysis.plot_SST import results as result_sst
# from experiments.data_augmentation.analysis.plot_TREC import results as result_trec

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
