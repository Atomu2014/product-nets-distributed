import socket

config = {}

host = socket.gethostname()
config['host'] = host.lower()

if config['host'] == 'altria':
    config['data_path'] = '/home/kevin/Dataset/Ads-RecSys-Datasets'
    config['env'] = 'cpu'
else:
    config['data_path'] = '/NAS/Dataset/Ads-RecSys-Datasets'
    config['env'] = 'gpu'

