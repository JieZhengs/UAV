import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import random


def get_parser():
    parser = argparse.ArgumentParser(description='multi-stream ensemble')
    # 添加命令行参数
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

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        inf = np.load(file, allow_pickle=True)
        score = torch.tensor(inf, dtype=torch.float32)
        if score.shape != (ntu60XS_num, Numclass):
            raise ValueError("The shape of the score tensor does not match the expected shape.")
        final_score += Rate[idx] * score
    final_score_np = final_score.numpy()
    np.save('work_dir/final_score_np.npy', final_score_np)
    return final_score


def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
    wrong_num = np.array(wrong_index).shape[0]
    total_num = true_label.shape[0]
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype=str)
    for name in val_txt:
        label = int(name.split('A')[1][:3])
        true_label.append(label)
    true_label = torch.from_numpy(np.array(true_label))
    return true_label


class Particle:
    def __init__(self, rates, tolerance=0.8):
        
        self.position = [random.uniform(rate - tolerance, rate + tolerance) for rate in rates]
        self.velocity = [random.uniform(-1, 1) for _ in rates]
        self.best_position = self.position[:]
        self.best_score = float('-inf')
    
    def update_velocity(self, global_best_position, inertia=1.49618, cognitive=0.7298, social=0.9298):
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive_velocity = cognitive * r1 * (self.best_position[i] - self.position[i])
            social_velocity = social * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia * self.velocity[i] + cognitive_velocity + social_velocity
    
    def update_position(self, rates, tolerance=0.8):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            self.position[i] = max(min(self.position[i], rates[i] + tolerance), rates[i] - tolerance)


def pso(rates, File, ntu60XS_num, Numclass, true_label, num_particles=100, max_iterations=100):
    particles = [Particle(rates) for _ in range(num_particles)]
    global_best_position = rates[:]
    global_best_score = float('-inf')

    for iteration in range(max_iterations):
        for particle in particles:
            final_score = Cal_Score(File, particle.position, ntu60XS_num, Numclass)
            Acc = Cal_Acc(final_score, true_label)
            if Acc > particle.best_score:
                particle.best_score = Acc
                particle.best_position = particle.position[:]
            
            if Acc > global_best_score:
                global_best_score = Acc
                global_best_position = particle.position[:]

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position(rates)

        print(f"Iteration {iteration + 1}/{max_iterations}, Best Accuracy: {global_best_score}")

    return global_best_position, global_best_score

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # 读取score文件路径
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
        true_label = np.load(r'data\val_label.npy')

        best_position, best_score = pso([5, 2, 7, 1, 1, 0, 2, 4, 3, 2, 8], File, Sample_Num, Numclass, true_label)
        
        print('Best Accuracy: ', best_score)
        print('Best Rate: ', best_position)