import numpy as np
import re
import csv
import time
from gurobipy import *
from functools import reduce
import os
import sys
import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    # return v.reshape((v.size,1))
    return v


def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size, 1))
    # return v


def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i + 1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result


def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])


def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i + 1])
            b = parse_bias(lines[i + 2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer += 1
            i += 3
        else:
            raise Exception('parse error: ' + lines[i])
    return res


def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',', dtype=np.double)
    low = np.copy(data[:, 0])
    high = np.copy(data[:, 1])
    return low, high


def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if (LB_N0[i] < 0):
            LB_N0[i] = 0
        if (UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def predict(nn, input_arr):
    r = input_arr
    for layer_no in range(nn.numlayer):
        weight = nn.weights[layer_no]
        bias = nn.biases[layer_no]
        h = np.matmul(weight,r)+bias
        r = np.clip(h,0,np.inf)
    label = np.argmax(r)
    return label

def calculate_score(ground_truth, output_file):
    score = 0
    with open(ground_truth,'r') as gt, open(output_line, 'r') as output:
        gt_lines = gt.readlines()
        output_lines = output.readlines()
        assert len(gt_lines) == len(output_lines)
        for i in range(len(gt_lines)):
            gt_result = gt_lines[i].split(' ')[3]
            output_result = output_lines[i].split('\t')


def analyze(nn, LB_N0, UB_N0, label):
    # Create a new model
    m = Model("mip1")
    m.setParam('OutputFlag', False)
    # Create variables
    input_dim = len(LB_N0)
    r = [m.addVar(name="i%s" % str(i),vtype='C',lb=LB_N0[i],ub=UB_N0[i]) for i in range(input_dim)]
    r_lb_vec = LB_N0
    r_ub_vec = UB_N0
    ub_precise_list = []
    ub_sound_list = []
    lb_precise_list = []
    lb_sound_list = []
    ub_diff_list = []
    lb_diff_list = []
    for layer_no in range(nn.numlayer):
        weight = nn.weights[layer_no]
        bias = nn.biases[layer_no]
        temp = weight * r
        h = [reduce((lambda x,y: x+y),temp[i]) for i in range(weight.shape[0])]+bias

        potential_bd = []
        potential_bd.append(weight * r_lb_vec)
        potential_bd.append(weight * r_ub_vec)
        h_lb_vec_sound = np.sum(np.min(potential_bd, axis=0), axis=1)  # hidden units bound
        h_ub_vec_sound = np.sum(np.max(potential_bd, axis=0), axis=1)

        h_lb_vec_precise = []
        h_ub_vec_precise = []
        thre_no = 9
        if 0 < layer_no < thre_no:
            for i in range(len(h)):
                m.setObjective(h[i], GRB.MINIMIZE)
                m.optimize()
                try:
                    h_lb_vec_precise.append(m.objVal)
                    diff = m.objVal-h_lb_vec_sound[i]
                    h_ub_vec_precise.append(h_ub_vec_sound[i]-diff)
                except AttributeError:
                    h_lb_vec_precise.append(h_lb_vec_sound[i])
                    h_ub_vec_precise.append(h_ub_vec_sound[i])
                # m.setObjective(h[i], GRB.MAXIMIZE)
                # m.optimize()
                # try:
                #     h_ub_vec_precise.append(m.objVal)
                # except AttributeError:
                #     h_ub_vec_precise.append(h_ub_vec_sound[i])
            ub_diff_list.append(np.array(h_ub_vec_precise)-np.array(h_ub_vec_sound))
            lb_diff_list.append(np.array(h_lb_vec_precise)-np.array(h_lb_vec_sound))
            ub_precise_list.append(np.array(h_ub_vec_precise))
            lb_precise_list.append(np.array(h_lb_vec_precise))
            ub_sound_list.append(np.array(h_ub_vec_sound))
            lb_sound_list.append(np.array(h_lb_vec_sound))
        elif layer_no == 0:
            h_lb_vec_precise = h_lb_vec_sound
            h_ub_vec_precise = h_ub_vec_sound

        r_lb_vec = np.clip(h_lb_vec_precise,0,np.inf) # relu units bound estimate
        r_ub_vec = np.clip(h_ub_vec_precise,0,np.inf)

        r = [m.addVar(name="r%s_%s" %(layer_no,hidunit_no), vtype='C', lb=0) for hidunit_no in range(len(h))]
        for hidunit_no in range(len(h)):
            if h_lb_vec_precise[hidunit_no] >= 0:  # r = h （maybe we can change to see if it will be faster）
                m.addConstr(r[hidunit_no] <= h[hidunit_no])
                m.addConstr(r[hidunit_no] >= h[hidunit_no])
            elif h_ub_vec_precise[hidunit_no] <= 0:  # r = 0
                m.addConstr(r[hidunit_no] <= 0)
            else:  # r <= \lambad*h+\mu
                lambda_ = h_ub_vec_precise[hidunit_no]/(h_ub_vec_precise[hidunit_no]-h_lb_vec_precise[hidunit_no])
                mu_ = -h_lb_vec_precise[hidunit_no]*lambda_
                m.addConstr(r[hidunit_no] <= lambda_*h[hidunit_no]+mu_)
                m.addConstr(r[hidunit_no] >= h[hidunit_no])
    bound_list =np.concatenate((ub_precise_list,ub_sound_list,lb_precise_list,lb_sound_list),axis=0)
    diff_list = np.concatenate((ub_diff_list,lb_diff_list),axis=0)
    if np.sum(r_ub_vec >= r_lb_vec[label]) > 1:
        verified_flag = False
    else:
        verified_flag = True
    return r_lb_vec,r_ub_vec,verified_flag,bound_list,diff_list

if __name__ == '__main__':
    from sys import argv

    if len(argv) < 2 or len(argv) > 3:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)
    netname = argv[1]
    # specname = argv[2]
    epsilon = float(argv[2])

    #netname = 'mnist_nets/mnist_relu_6_100.txt'
    #epsilon = 0.01
    result_file_name = netname.split('/')[1].split('.')[0]+'_eps_'+str(epsilon)+'_first_3_precise_with_2_constrain_single'
    result_file_path = os.path.join('riai_project_output',result_file_name)
    f_output = open(result_file_path,'w')

    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    nn = parse_net(netstring)
    count = 0
    total_count = 100
    for img_id in [1]:
        specname = os.path.join('mnist_images','img'+str(img_id)+'.txt')
        with open(specname, 'r') as specfile:
            specstring = specfile.read()
        x0_low, x0_high = parse_spec(specstring)
        LB_N0, UB_N0 = get_perturbed_image(x0_low, 0)

        label= predict(nn, LB_N0)
        start = time.time()
        signal.alarm(420)
        if (label == int(x0_low[0])):
            LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)
            try:
                lb,ub,verified_flag,bound_list,diff_list = analyze(nn, LB_N0, UB_N0,label)
                np.save('bound.npy',bound_list)
                np.save('diff.npy',diff_list)
            except TimeoutException:
                verified_flag = False
            else:
                signal.alarm(0)
            if (verified_flag):
                print("verified")
                verified_output = 'verified'
                count += 1
            else:
                print("can not be verified")
                verified_output = 'failed'
        else:
            print("image not correctly classified by the network. expected label ", int(x0_low[0]), " classified label: ",
                  label)
            verified_output = 'not considered'
            total_count -= 1
        end = time.time()
        print("analysis time: ", (end - start), " seconds")
        sys.stdout.flush()
        output_line = '\t'.join(['img',str(img_id),verified_output,str(end-start)])+'\n'
        f_output.write(output_line)
    f_output.write('analysis precision  {} /  {}'.format(count,total_count))

# max_purt = []
# cor = []
# for i in (range(7)):
#     bdlen = bound[i]-bound[i+14]
#     purt = np.sum(np.abs(nn.weights[i+1]*bdlen),axis = 1)
#     max_purt.append(purt)
#     cor.append(np.corrcoef(purt,diff[i])[0,1])
# max_purt = np.array(max_purt)

