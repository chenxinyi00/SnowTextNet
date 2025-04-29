import os


def get_images(img_path):
    '''
    find image files in data path
    :return: list of files found
    '''
    img_path = os.path.abspath(img_path)
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'PNG']
    for parent, dirnames, filenames in os.walk(img_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return sorted(files)


def get_txts(txt_path):
    '''
    find gt files in data path
    :return: list of files found
    '''
    txt_path = os.path.abspath(txt_path)
    files = []
    exts = ['txt']
    for parent, dirnames, filenames in os.walk(txt_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} txts'.format(len(files)))
    return sorted(files)


if __name__ == '__main__':
    # img_path = './train/img'
    img_path = '/data1/cxy/sodbnettd500/test/img'
    files = get_images(img_path)

    img_path1 = '/data1/cxy/sodbnettd500/test/snow'
    files1 = get_images(img_path1)

    # txt_path = './train/gt'
    txt_path = '/data1/cxy/sodbnettd500/test/gt'
    txts = get_txts(txt_path)
    n = len(files)
    n1=len(files1)
    assert len(files) == len(txts) ==len(files1)
    with open('/data1/cxy/sodbnettd500/test.txt', 'w') as f:
        for i in range(n):
            line = files[i] + '\t' +files1[i] + '\t' + txts[i] + '\n'
            f.write(line)

    print('dataset generated ^_^ ')