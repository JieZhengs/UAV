import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import random

def get_parser():
    parser = argparse.ArgumentParser(description='multi-stream ensemble')

    parser.add_argument(
        '--tegcn_Score_1',
        type=str,
        default=r'pre_res\pred_ctr_joint_bone1.npy'),
    parser.add_argument(
        '--ctrgcn_Score_2',
        type=str,
        default=r'pre_res\pred_ctr_joint_bone2.npy'),
    parser.add_argument(
        '--tdgcn_Score_3',
        type=str,
        default=r'pre_res\pred_td_joint_bone3.npy'),
    parser.add_argument(
        '--tegcn_Score_4',
        type=str,
        default=r'pre_res\pred_te_joint4.npy'),
    parser.add_argument(
        '--tegcn_Score_5',
        type=str,
        default=r'pre_res\pred_te_joint_bone5.npy'),
    parser.add_argument(
        '--tegcn_Score_6',
        type=str,
        default=r'pre_res\pred_te_bone6.npy'),
    parser.add_argument(
        '--ctrgcn_Score_7',
        type=str,
        default=r'pre_res\train_ctr_bone-52_4284.npy'),
    parser.add_argument(
        '--ctrgcn_Score_8',
        type=str,
        default=r'pre_res\train_ctr_bone247_4312.npy'),
    parser.add_argument(
        '--ctrgcn_Score_9',
        type=str,
        default=r'pre_res\train_ctr_joint141_4300.npy'),
    parser.add_argument(
        '--tdgcn_Score_10',
        type=str,
        default=r'pre_res\train_td_bone-43_4219.npy'),
    parser.add_argument(
        '--tdgcn_Score_11',
        type=str,
        default=r'pre_res\train_td_joint247_4300.npy'),
    parser.add_argument(
        '--tdgcn_Score_12',
        type=str,
        default=r'pre_res\ctr_joint_4404.npy'),
    parser.add_argument(
        '--tdgcn_Score_13',
        type=str,
        default=r'pre_res\td_joint_bone_4393_13.npy'),

    
    parser.add_argument(
        '--val_sample',
        type=str,
        default='./Process_data/CS_test_V2.txt'),
    parser.add_argument(
        '--benchmark',
        type=str,
        default='V2')
    return parser


def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        inf = np.load(file, allow_pickle=True)
        score = torch.tensor(inf, dtype=torch.float32)
        if score.shape != (ntu60XS_num, Numclass):
            raise ValueError("The shape of the score tensor does not match the expected shape.")

        final_score += Rate[idx] * score
    final_score_np = final_score.numpy()
    np.save('work_dir/pred.npy', final_score_np)
    return final_score


def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)

    wrong_num = np.array(wrong_index).shape[0]
    # print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    # print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc


def gen_label(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype=str)
    for idx, name in enumerate(val_txt):
        label = int(name.split('A')[1][:3])
        true_label.append(label)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label
def ran():
    
    rates = [4.494278788653091, 4.177561537065381, 7.514595123450889, 0.9555765737436815, 0.9703255527941663, 0.13635316021713514, 2.0129549249226533, 3.664663385062701, 3.643147746503847, 0.9706497073919321, 7.638879058728796]

    
    array_length = 10

    
    tolerance = 0.7

    
    random_array = []
    for rate in rates:
        
        random_value = random.uniform(rate - tolerance, rate + tolerance)
        random_array.append(random_value)
    return random_array

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    Score_file_1 = args.tegcn_Score_1
    Score_file_2 = args.ctrgcn_Score_2
    Score_file_3 = args.tdgcn_Score_3
    Score_file_4 = args.tegcn_Score_4
    Score_file_5 = args.tegcn_Score_5
    Score_file_6 = args.tegcn_Score_6
    Score_file_7 = args.ctrgcn_Score_7
    Score_file_8 = args.ctrgcn_Score_8
    Score_file_9 = args.ctrgcn_Score_9
    Score_file_10 = args.tdgcn_Score_10
    Score_file_11 = args.tdgcn_Score_11
    Score_file_12 = args.tdgcn_Score_12
    Score_file_13 = args.tdgcn_Score_13

    File = [Score_file_1, Score_file_2, Score_file_3, Score_file_4, Score_file_5, Score_file_6, Score_file_7, Score_file_8, Score_file_9, Score_file_10, Score_file_11, Score_file_12, Score_file_13]

    if args.benchmark == 'V2':
        Numclass = 155
        Sample_Num = 2000
        best_Acc = 0
        best_rate = []
        for i in range(1):
            Rate = [2.043657545258659, 3.187101109269344, 5.873476202521488, 4.678717740853027, 1.2074025263468688, 2.1962578433865425, -1.3164161768069882, 6.603675832620936, 2.1962578433865425, 8.067121711896956, -1.1023075930863073,0.27986643043507275,0.2643313882902165]
            # Rate = ran()
            # print(Rate)
            final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
            true_label = np.load(r'data\val_label.npy')
            Acc = Cal_Acc(final_score, true_label)
            if Acc > best_Acc:
                best_Acc = Acc
                best_rate = Rate
                print('best_Acc: ', best_Acc)
                print('best_rate: ', best_rate)