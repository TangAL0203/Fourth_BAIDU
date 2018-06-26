import os
import util

if __name__ == '__main__':
    test_txt = '../datasets/test.txt'
    fp = open(test_txt, 'r')
    lines = fp.readlines()
    fp.close()
    out_dir = '../datasets/need_annotation'
    util.mkdir(out_dir)
    for i, line in enumerate(lines[:901]):
        img_name = os.path.join('../datasets/test', line[:-1])
        util.copy(img_name, out_dir)
        
