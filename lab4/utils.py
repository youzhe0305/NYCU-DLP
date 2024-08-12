import matplotlib.pyplot as plt

def training_loss_plot(loss_log, filename, title):

    x = loss_log[0]
    y = loss_log[1]
    plt.clf()
    
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('average training loss')
    plt.yscale('log')
    plt.show()
    plt.savefig(f'output/{filename}')
    
def training_loss_log(loss_log, filename):
    with open(f'output/{filename}') as f:
        for loss in loss_log:
            f.write(loss)

def validation_psnr_plot(psnr_log, filename, title):
    x = psnr_log[0]
    y = psnr_log[1]
    plt.clf()
    
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('frame')
    plt.ylabel('psnr')
    plt.show()
    plt.savefig(f'output/{filename}')