with open(file='./train.csv', mode='r') as f:
    lines = f.readlines()[1:]
    labels = []
    for line in lines:
        label = line.strip('\n').split(',')[1]
        # if label == 'new_whale':
        #     continue
        if label in labels:
            continue
        labels.append(label)

    print('new_whale' in labels)
    print(len(labels))
    # id = 0
    # with open(file='id_label.csv', mode='w') as ff:
    #     for line in lines:
