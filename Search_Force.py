import torch
import argparse
import numpy as np
from tqdm import tqdm 

def get_parser():
    parser = argparse.ArgumentParser(description='多流集成')

    parser.add_argument('--tegcn_Score_1', type=str, default=r'pre_res\pred_ctr_joint_bone1.npy')
    parser.add_argument('--ctrgcn_Score_2', type=str, default=r'pre_res\pred_ctr_joint_bone2.npy')
    parser.add_argument('--tdgcn_Score_3', type=str, default=r'pre_res\pred_td_joint_bone3.npy')
    parser.add_argument('--tegcn_Score_4', type=str, default=r'pre_res\pred_te_joint4.npy')
    parser.add_argument('--tegcn_Score_5', type=str, default=r'pre_res\pred_te_joint_bone5.npy')
    parser.add_argument('--tegcn_Score_6', type=str, default=r'pre_res\pred_te_bone6.npy')
    parser.add_argument('--ctrgcn_Score_7', type=str, default=r'pre_res\train_ctr_bone-52_4284.npy')
    parser.add_argument('--ctrgcn_Score_8', type=str, default=r'pre_res\train_ctr_bone247_4312.npy')
    parser.add_argument('--ctrgcn_Score_9', type=str, default=r'pre_res\train_ctr_joint141_4300.npy')
    parser.add_argument('--tdgcn_Score_10', type=str, default=r'pre_res\train_td_bone-43_4219.npy')
    parser.add_argument('--tdgcn_Score_11', type=str, default=r'pre_res\train_td_joint247_4300.npy')
    parser.add_argument('--tdgcn_Score_12', type=str, default=r'pre_res\ctr_joint_4404.npy')
    parser.add_argument('--tdgcn_Score_13', type=str, default=r'pre_res\td_joint_bone_4393_13.npy')
    
    parser.add_argument('--val_sample', type=str, default='./Process_data/CS_test_V2.txt')
    parser.add_argument('--benchmark', type=str, default='V2')
    return parser


def Cal_Score(File, Rate, ntu60XS_num, Numclass, device):
    final_score = torch.zeros(ntu60XS_num, Numclass, device=device)  # 使用GPU加速计算
    for idx, file in enumerate(File):
        inf = np.load(file, allow_pickle=True)
        score = torch.tensor(inf, dtype=torch.float32, device=device)  # 将数据直接加载到GPU
        if score.shape != (ntu60XS_num, Numclass):
            raise ValueError("分数张量的形状与预期形状不匹配。")
        final_score += Rate[idx] * score
    return final_score


def Cal_Acc(final_score, true_label, device):
    _, predict_label = torch.max(final_score, 1)
    wrong_num = (predict_label != true_label.to(device)).sum().item()  # 将标签也移动到GPU并计算误差
    total_num = true_label.shape[0]
    return (total_num - wrong_num) / total_num


def brute_force_search(File, Sample_Num, Numclass, true_label, device, initial_rates):
    best_Acc = 0
    best_rate = []
    
    rate_ranges = [range(max(0, int(rate) - 1), min(10, int(rate) + 2)) for rate in initial_rates]
    
    total_combinations = 1
    for r in rate_ranges:
        total_combinations *= len(r)
    

    with tqdm(total=total_combinations, desc="暴力破解进度", unit="组合") as pbar:
        for r1 in rate_ranges[0]:
            for r2 in rate_ranges[1]:
                for r3 in rate_ranges[2]:
                    for r4 in rate_ranges[3]:
                        for r5 in rate_ranges[4]:
                            for r6 in rate_ranges[5]:
                                for r7 in rate_ranges[6]:
                                    for r8 in rate_ranges[7]:
                                        for r9 in rate_ranges[8]:
                                            for r10 in rate_ranges[9]:
                                                for r11 in rate_ranges[10]:
                                                    Rate = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11]
                                                    final_score = Cal_Score(File, Rate, Sample_Num, Numclass, device)
                                                    Acc = Cal_Acc(final_score, true_label, device)
                                                    
                                                    if Acc > best_Acc:
                                                        best_Acc = Acc
                                                        best_rate = Rate
                                                        print(f'新最佳准确率: {best_Acc}, 使用Rate: {best_rate}')
                                                    
                                                    
                                                    pbar.update(1)
    
    print('最终最佳准确率:', best_Acc)
    print('最佳Rate值组合:', best_rate)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    File = [
        args.tegcn_Score_1, args.ctrgcn_Score_2, args.tdgcn_Score_3, 
        args.tegcn_Score_4, args.tegcn_Score_5, args.tegcn_Score_6, 
        args.ctrgcn_Score_7, args.ctrgcn_Score_8, args.ctrgcn_Score_9, 
        args.tdgcn_Score_10, args.tdgcn_Score_11,args.tdgcn_Score_12,args.tdgcn_Score_13
    ]

    if args.benchmark == 'V2':
        Numclass = 155
        Sample_Num = 2000

        
        initial_rates = [
            1, 1, 1, 
            1, 1, 1, 
            1, 1, 1, 
            1, 1, 1,
            1
        ]

        
        true_label = np.load(r'data\val_label.npy')
        true_label = torch.from_numpy(true_label)

        
        brute_force_search(File, Sample_Num, Numclass, true_label, device, initial_rates)
