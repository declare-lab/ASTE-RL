import numpy as np
import torch
from AccCalc import calc_acc, calcF1


def calcTopReward(top_action, gold_labels):
    """
    Intermediate top reward.
    """
    lenth = len(top_action)
    r = [0. for i in range(lenth)]
    rem = [0 for i in range(len(gold_labels))]
    for i in range(lenth)[::-1]:
        if top_action[i] > 0:
            ok = -1
            for j, label in enumerate(gold_labels):
                if label['type'] == top_action[i]:
                    if rem[j] == 0:
                        ok = 1
                        rem[j] = 1
                        break
                    else:
                        ok = -0.2
            r[i] = ok
    return r


def calcTopFinalReward(top_action, gold_labels, top_bias = 0.):
    """
    Final top reward, using F1 score.
    """
    r = 0.
    a1, t1, c1 = calc_acc(top_action, None, None, gold_labels, ["RE"])
    if c1 != 0:
        r = calcF1(a1, c1, t1)
    else:
        r = -2
    if c1 > t1:
        r -= 0.5 * (c1 - t1)
    r *= len(top_action)
    return r - top_bias


def calcBotReward(top_action, bot_aspect_action, bot_opinion_action, gold_labels):
    """
    Intermediate bottom reward.
    """
    lenth = len(top_action)
    r = [[0. for i in range(lenth)] for j in range(len(bot_aspect_action))]
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            for label in gold_labels:
                if label['type'] == top_action[i]:
                    for t in range(lenth):
                        if label['aspect_tags'][t] == bot_aspect_action[j][t]:
                            if label['aspect_tags'][t] == 2:
                                r[j][t] = 0.5
                            elif label['aspect_tags'][t] == 1:
                                r[j][t] = 0.2
                        else:
                            r[j][t] = -0.5
                        if label['opinion_tags'][t] == bot_opinion_action[j][t]:
                            if label['opinion_tags'][t] == 2:
                                r[j][t] += 0.5
                            elif label['opinion_tags'][t] == 1:
                                r[j][t] += 0.2
                        else:
                            r[j][t] -= 0.5
            j += 1
    return r


def calcBotFinalReward(top_action, bot_aspect_action, bot_opinion_action, gold_labels, bot_bias = 0.):
    """
    Final bottom reward, using tagging sequence matching.
    """
    lenth = len(top_action)
    r = [0. for j in range(len(bot_aspect_action))]
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            r[j] = -1.0
            for label in gold_labels:
                if label['type'] == top_action[i]:
                    ok = True
                    for t in range(lenth):
                        if label['aspect_tags'][t] != bot_aspect_action[j][t] or label['opinion_tags'][t] != bot_opinion_action[j][t]:
                            ok = False
                            break
                    if ok:
                        r[j] = 1

            # negative reward if there are impossible results
            # 1) if either beginning tag of aspect or opinion span is absent or there is more than 1 of either
            if bot_aspect_action[j].count(2) !=1:
                r[j] -= 0.2
            if bot_opinion_action[j].count(2) !=1:
                r[j] -= 0.2
            # 2) if no aspect or opinion span detected or more than 1 of either is detected
            aspect_cnt = 0
            prev_aspect = False
            for k, aspect_k in enumerate(bot_aspect_action[j]):
                if aspect_k > 0:
                    prev_aspect = True
                    if k == len(bot_aspect_action[j]) - 1:
                        aspect_cnt += 1
                elif aspect_k == 0:
                    if prev_aspect:
                        aspect_cnt += 1
                        prev_aspect = False
                if aspect_cnt > 1:
                    break
            opinion_cnt = 0
            prev_opinion = False
            for k, opinion_k in enumerate(bot_opinion_action[j]):
                if opinion_k > 0:
                    prev_opinion = True
                    if k == len(bot_opinion_action[j]) - 1:
                        opinion_cnt += 1
                elif opinion_k == 0:
                    if prev_opinion:
                        opinion_cnt += 1
                        prev_opinion = False
                if opinion_cnt > 1:
                    break
            if aspect_cnt != 1:
                r[j] -= 0.2
            if opinion_cnt != 1:
                r[j] -= 0.2

            j += 1
    for j in range(len(bot_aspect_action)):
        r[j] -= bot_bias
    return r


def calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain=False, device=torch.device("cpu")):
    lenth = len(top_action)
    decay_reward = top_final_reward 
    grads = torch.FloatTensor(1, ).fill_(0).to(device)
    for i in range(lenth)[::-1]:
        decay_reward = decay_reward * 0.95 + top_reward[i]
        to_grad = -torch.log(top_actprob[i]).to(device)
        if not pretrain:
            to_grad *= torch.FloatTensor(1, ).fill_(decay_reward).to(device)
        if top_action[i] == 0:
            to_grad *= 0.3
        grads = grads + to_grad
    return grads


def calcBotGrad(top_action, bot_aspect_action, bot_aspect_actprob, bot_opinion_action, bot_opinion_actprob, bot_reward, bot_final_reward, pretrain=False, device=torch.device("cpu")):
    lenth = len(top_action)
    bot_tot_reward = [0. for i in range(lenth)]
    grads = torch.FloatTensor(1, ).fill_(0).to(device)
    grads = torch.unsqueeze(grads, dim=0)
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            bot_tot_reward[i] = sum(bot_reward[j]) / lenth + bot_final_reward[j]
            for k in range(lenth)[::-1]:
                aspect_to_grad = -torch.log(bot_aspect_actprob[j][k])
                aspect_to_grad = torch.unsqueeze(aspect_to_grad, dim=0).to(device)
                if not pretrain:
                    aspect_to_grad *= torch.FloatTensor(1, ).fill_(bot_tot_reward[i]).to(device)
                if bot_aspect_action[j][k] == 0:
                    aspect_to_grad *= 0.3
                elif bot_aspect_action[j][k] == 1:
                    aspect_to_grad *= 0.7
                else:
                    aspect_to_grad *= 1.0

                opinion_to_grad = -torch.log(bot_opinion_actprob[j][k])
                opinion_to_grad = torch.unsqueeze(opinion_to_grad, dim=0).to(device)
                if not pretrain:
                    opinion_to_grad *= torch.FloatTensor(1, ).fill_(bot_tot_reward[i]).to(device)
                if bot_opinion_action[j][k] == 0:
                    opinion_to_grad *= 0.3
                elif bot_opinion_action[j][k] == 1:
                    opinion_to_grad *= 0.7
                else:
                    opinion_to_grad *= 1.0
                    
                grads = grads + aspect_to_grad + opinion_to_grad
            j += 1
    return bot_tot_reward, grads


def optimize(model, top_action, top_actprob, bot_aspect_action, bot_aspect_actprob, bot_opinion_action, bot_opinion_actprob, gold_labels, mode, top_bias = 0., bot_bias = 0., device=torch.device("cpu")):
    lenth = len(top_action)
    top_reward = calcTopReward(top_action, gold_labels)
    top_final_reward = calcTopFinalReward(top_action, gold_labels, top_bias)
    pretrain = True if "pretrain" in mode else False
    if "NER" in mode:
        bot_reward = calcBotReward(top_action, bot_aspect_action, bot_opinion_action, gold_labels)
        bot_final_reward = calcBotFinalReward(top_action, bot_aspect_action, bot_opinion_action, gold_labels, bot_bias)
        bot_tot_reward, grads = calcBotGrad(top_action, bot_aspect_action, bot_aspect_actprob, bot_opinion_action, bot_opinion_actprob, bot_reward, bot_final_reward, pretrain, device)
        for i in range(lenth):
            top_reward[i] += bot_tot_reward[i]
    else:
        grads = torch.FloatTensor(1, ).fill_(0).to(device)
        grads = torch.unsqueeze(grads, dim=0)
    if "RE" in mode:
        grads += calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain, device)
    loss = grads.cpu().data[0]
    grads.backward()
    return loss


def optimize_round(model, top_actions, top_actprobs, bot_aspect_actions, bot_aspect_actprobs, bot_opinion_actions, bot_opinion_actprobs, gold_labels, mode, device):
    sample_round = len(top_actions)
    # get bias first
    if "RE" in mode:
        top_bias = 0.
        for i in range(sample_round):
            top_bias += calcTopFinalReward(top_actions[i], gold_labels, 0.)
        top_bias /= sample_round
    else:
        top_bias = 0.
    if "NER" in mode:
        bot_bias, bot_cnt = 0., 0
        for i in range(sample_round):
            tmp = calcBotFinalReward(top_actions[i], bot_aspect_actions[i], bot_opinion_actions[i], gold_labels, 0.)
            bot_cnt += len(tmp)
            bot_bias += np.sum(tmp)
        if bot_cnt != 0:
            bot_bias /= bot_cnt
    else:
        bot_bias = 0.
    loss = .0
    # real optimisation with top/bot biases taken into account
    for i in range(sample_round):
        loss += optimize(model, top_actions[i], top_actprobs[i], bot_aspect_actions[i], \
                bot_aspect_actprobs[i], bot_opinion_actions[i], bot_opinion_actprobs[i], gold_labels, mode, top_bias, bot_bias, device)
    return loss / sample_round
