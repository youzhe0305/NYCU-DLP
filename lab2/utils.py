import matplotlib.pyplot as plt

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

def accuracy_bar_chart():
    categories = ['LOSO', 'SD', 'LOSO_FT']
    acc = [0.625, 0.6393,  0.8021]
    plt.bar(categories, acc)
    plt.title('accuracy comparison')
    plt.xlabel('training method')
    plt.ylabel('acuracy')
    plt.savefig('output/acc_comp_fig')

if __name__ == '__main__':
    accuracy_bar_chart()
    # training_loss_log_x = [100,200,300,400,500]
    # training_loss_log_y = [1.3,1.1,0.8,0.5,0.1]
    # training_loss_plot((training_loss_log_x, training_loss_log_y), 'test')
    pass