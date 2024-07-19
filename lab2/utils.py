import matplotlib.pyplot as plt

# 畫training loss的圖
# 把不同訓練的結果資料存成txt然後畫圖


def training_loss_plot(loss_log, filename):

    x = loss_log[0]
    y = loss_log[1]
    plt.clf()
    plt.plot(x, y, marker='o')
    plt.show()
    plt.savefig(f'output/{filename}')

def training_mathod_log(test_name, accuracy, loss):
    with open('output/diff_method_log.txt') as f:
        f.write(f'{test_name},{accuracy},{loss}')


if __name__ == '__main__':
    pass
    # training_loss_log_x = [100,200,300,400,500]
    # training_loss_log_y = [1.3,1.1,0.8,0.5,0.1]
    # training_loss_plot((training_loss_log_x, training_loss_log_y), 'test')