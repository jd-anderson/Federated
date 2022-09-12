import matplotlib.pyplot as plt
import csv
import os

#os.chdir('AA_1.1_hue')


def read_results(filename):
    with open(filename) as out_file:
        csv_read = csv.reader(out_file, delimiter=',')
        for ind, read in enumerate(csv_read):
            if ind == 1:
                train_loss = [float(x) for x in read]
            elif ind == 0:
                train_accuracy = [float(x) for x in read]
    return train_accuracy, train_loss


filenames = ['FedADMM_femnist_lr_0.01_10of20orig.csv', 'FedDR_femnist_lr_0.01_10of20orig.csv']

file_index = ['FedADMM', 'FedDR']
train_acc, train_loss = {}, {}
figures, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
figures.tight_layout(pad=5)
for ind, f in enumerate(filenames):
    train_acc[file_index[ind]], train_loss[file_index[ind]] = read_results(f)
    axes[0].plot(train_acc[file_index[ind]], label=file_index[ind])
    axes[1].plot(train_loss[file_index[ind]], label=file_index[ind])

axes[0].set_xlabel('epochs', fontdict={'size': 16})
axes[0].set_ylabel('accuracy', fontdict={'size': 16})
axes[0].legend()
axes[0].grid()

axes[1].set_xlabel('epochs', fontdict={'size': 16})
axes[1].set_ylabel('loss', fontdict={'size': 16})
axes[1].legend()
axes[1].grid()
plt.suptitle("Femnist- Comparison")
plt.savefig('Femnist-partial-users-10of20.png')
plt.show()

