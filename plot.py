import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    path = "log/2023_qua.txt"
    with open(path) as f:
        lines = f.readlines()
    lines = lines[:-1]
    x = []
    train_loss = []
    test_loss = []
    for i in range(len(lines)):
        line = lines[i].split()
        x.append(int(line[1][:-4]))
        train_loss.append(np.log10(float(line[4][:-1])))
        test_loss.append(np.log10(float(line[7])))

    print(train_loss)
    fig = plt.figure(figsize=(10,10))
    plt.xlabel('epoch')
    plt.ylabel('MSE(log10)')
    plt.plot(x, train_loss)
    plt.plot(x, test_loss)
    plt.legend(['train loss', 'test loss'])
    plt.savefig('2023_qua.png')
    plt.show()



    # fig = plt.figure(figsize=(10,10))
    # plt.xlabel('epoch')
    # plt.ylabel('test_MSE(log10)')

    # paths = ["log/2023_qua.txt", "log/2023_aut.txt", "log/2023_cor.txt"]
    # for path in paths:
    #     with open(path) as f:
    #         lines = f.readlines()
    #     lines = lines[:-1]
    #     x = []
    #     train_loss = []
    #     test_loss = []
    #     for i in range(len(lines)):
    #         line = lines[i].split()
    #         x.append(int(line[1][:-4]))
    #         train_loss.append(np.log10(float(line[4][:-1])))
    #         test_loss.append(np.log10(float(line[7])))
    #     plt.plot(x, test_loss)

    # plt.legend(['quality', 'authenticity', 'correspondence'])
    # plt.savefig('2023_compare.png')
    # plt.show()