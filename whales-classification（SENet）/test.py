from tqdm import tqdm
from .model import *
from .dataProcess import *
from .utils import *
from torch.utils.data import DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_TTA = 2


def test(checkPoint_start=0, fold_index=1, model_name='seresnext50'):
    names_test = os.listdir('./input/test')
    batch_size = 4

    dst_test = Trainset(names_test, mode='test', transform=transform_valid)
    dataloader_test = DataLoader(dst_test, batch_size=batch_size, num_workers=8, collate_fn=train_batch)

    label_id = dst_test.labels_dict
    id_label = {v: k for k, v in label_id.items()}
    id_label[5004] = 'new_whale'
    model = net(num_classes=5004 * 2, inchannels=4, model_name=model_name).cuda()

    resultDir = './result/{}_{}'.format(model_name, fold_index)
    checkPoint = os.path.join(resultDir, 'checkpoint')

    if not checkPoint_start == 0:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)), skip=[])
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        best_t = ckp['best_t']
        print('best_t:', best_t)

    labelstrs = []
    allnames = []

    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
            images, names = data
            images = images.cuda()
            _, _, outs = model(images)
            outs = torch.sigmoid(outs)
            outs_zero = (outs[::2, :5004] + outs[1::2, 5004:])/2
            outs = outs_zero
            for out, name in zip(outs, names):
                out = torch.cat([out, torch.ones(1).cuda()*best_t], 0)
                out = out.data.cpu().numpy()
                top5 = out.argsort()[-5:][::-1]
                str_top5 = ''
                for t in top5:
                    str_top5 += '{} '.format(id_label[t])
                str_top5 = str_top5[:-1]
                allnames.append(name)
                labelstrs.append(str_top5)

    pd.DataFrame({'Image': allnames, 'Id': labelstrs}).to_csv('test_{}_sub_fold{}.csv'.format(model_name, fold_index), index=None)


if __name__ == '__main__':
    checkPoint_start = 18200
    fold_index = 1
    model_name = 'seresnext50'
    test(checkPoint_start, fold_index, model_name)

