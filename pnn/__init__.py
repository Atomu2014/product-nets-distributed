import socket

config = {}

host = socket.gethostname()
config['host'] = host.lower()
config['location'] = 'apex'

if config['host'] == 'altria':
    config['data_path'] = '/home/kevin/nas/dataset/Ads-RecSys-Datasets'
    config['env'] = 'cpu'
else:
    if config['location'] == 'apex':
        config['data_path'] = '/newNAS/Datasets/MLGroup/Ads-RecSys-Datasets'
    elif config['location'] == 'huawei':
        config['data_path'] = '/home/distributed_train/Data/'
    config['env'] = 'gpu'
