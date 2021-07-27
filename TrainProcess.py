import time
import torch.optim as optim
import queue
from AccCalc import calc_acc, rule_actions
from Optimize import optimize_round


def workProcess(model, datas, sample_round, mode, device, sentiments, test):
    """
    Get model outputs and train model.
    """
    acc, cnt, tot = 0, 0, 0
    loss = .0
    for data in datas:
        top_actions, top_actprobs, bot_aspect_actions, bot_aspect_actprobs, bot_opinion_actions, bot_opinion_actprobs = [], [], [], [], [], []
        if not test:
            preoptions, pre_aspect_actions, pre_opinion_actions = rule_actions(data['triplets'])
        bert_to_whitespace = data['bert_to_whitespace']
        whitespace_tokens = data['whitespace_tokens']
        for i in range(sample_round):
            # pretraining
            if "pretrain" in mode and "test" not in mode:
                top_action, top_actprob, bot_aspect_action, bot_aspect_actprob, bot_opinion_action, bot_opinion_actprob = \
                        model(mode, data['pos_tags'], data['sentext'], \
                        preoptions, pre_aspect_actions, pre_opinion_actions, device, sentiments)
            # train from scratch
            else:
                top_action, top_actprob, bot_aspect_action, bot_aspect_actprob, bot_opinion_action, bot_opinion_actprob = \
                        model(mode, data['pos_tags'], data['sentext'], None, None, None, device, sentiments)

            top_actions.append(top_action)
            top_actprobs.append(top_actprob)
            bot_aspect_actions.append(bot_aspect_action)
            bot_aspect_actprobs.append(bot_aspect_actprob)
            bot_opinion_actions.append(bot_opinion_action)
            bot_opinion_actprobs.append(bot_opinion_actprob)

            if not test:
                acc1, tot1, cnt1 = calc_acc(top_action, bot_aspect_action, bot_opinion_action, \
                        data['triplets'], mode)
                acc += acc1
                tot += tot1
                cnt += cnt1
            
        # training optimisation
        if "test" not in mode:
            loss += optimize_round(model, top_actions, top_actprobs, bot_aspect_actions,\
                    bot_aspect_actprobs, bot_opinion_actions, bot_opinion_actprobs, data['triplets'], mode, device)
        # print for inference/testing
        elif test:
            all_preds = []
            j = 0
            for i in range(len(top_action)):
                if top_action[i] > 0:
                    aspect_bot = bot_aspect_action[j]
                    opinion_bot = bot_opinion_action[j]
                    if aspect_bot.count(2) !=1 or opinion_bot.count(2) !=1:
                        j += 1
                        continue
                    # 2) skip if no aspect or opinion span detected or more than 1 of either is detected
                    aspect_cnt = 0
                    prev_aspect = False
                    for k, aspect_k in enumerate(aspect_bot):
                        if aspect_k > 0:
                            prev_aspect = True
                            if k == len(aspect_bot) - 1:
                                aspect_cnt += 1
                        elif aspect_k == 0:
                            if prev_aspect:
                                aspect_cnt += 1
                                prev_aspect = False
                        if aspect_cnt > 1:
                            break
                    opinion_cnt = 0
                    prev_opinion = False
                    for k, opinion_k in enumerate(opinion_bot):
                        if opinion_k > 0:
                            prev_opinion = True
                            if k == len(opinion_bot) - 1:
                                opinion_cnt += 1
                        elif opinion_k == 0:
                            if prev_opinion:
                                opinion_cnt += 1
                                prev_opinion = False
                        if opinion_cnt > 1:
                            break
                    if aspect_cnt != 1 or opinion_cnt != 1:
                        j += 1
                        continue
                    
                    # print inference outputs
                    aspect = []
                    curr = None
                    for m, aspect_tag in enumerate(aspect_bot):
                        if aspect_tag.item() != 0:
                            if curr == None:
                                curr = bert_to_whitespace[m][0]
                            else:
                                if bert_to_whitespace[m][0] == curr:
                                    continue
                                else:
                                    curr = bert_to_whitespace[m][0]
                            aspect.append(whitespace_tokens[bert_to_whitespace[m][0]])
                    opinion = []
                    curr = None
                    for m, opinion_tag in enumerate(opinion_bot):
                        if opinion_tag.item() != 0:
                            if curr == None:
                                curr = bert_to_whitespace[m][0]
                            else:
                                if bert_to_whitespace[m][0] == curr:
                                    continue
                                else:
                                    curr = bert_to_whitespace[m][0]
                            opinion.append(whitespace_tokens[bert_to_whitespace[m][0]])
                    all_preds.append((sentiments[top_action[i].item()-1], ' '.join(aspect), ' '.join(opinion)))
                    j += 1
            print(data['sentext'], '==>', str(all_preds))

    if len(datas) == 0:
        return 0, 0, 0, 0
    return acc, cnt, tot, loss / len(datas)


def worker(model, rank, dataQueue, resultQueue, freeProcess, lock, flock, lr, sentiments):
    # get data from queue to train/val/test
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Process ", rank, " start service.")
    flock.acquire()
    freeProcess.value += 1
    flock.release()
    while True:
        datas, sample_round, mode, dataID, device, sentiments, test = dataQueue.get()
        flock.acquire()
        freeProcess.value -= 1
        flock.release()
        model.zero_grad()
        acc, cnt, tot, loss = workProcess(model, datas, sample_round, mode, device, sentiments, test)
        resultQueue.put((acc, cnt, tot, dataID, rank, loss))
        if not "test" in mode:
            lock.acquire()
            optimizer.step()
            lock.release()
        flock.acquire()
        freeProcess.value += 1
        flock.release()


def train(dataID, model, datas, sample_round, mode, dataQueue, resultQueue, freeProcess, lock, numprocess, device, sentiments, test):
    # put data into queue
    dataPerProcess = len(datas) // numprocess
    while freeProcess.value != numprocess:
        pass
    acc, cnt, tot = 0, 0, 0
    loss = .0
    for r in range(numprocess):
        endPos = ((r+1)*dataPerProcess if r+1 != numprocess else len(datas))
        data = datas[r*dataPerProcess: endPos]
        dataQueue.put((data, sample_round, mode, dataID, device, sentiments, test))
    lock.acquire()
    try:
        for r in range(numprocess):
            while True:
                item = resultQueue.get()
                if item[3] == dataID:
                    break
                else:
                    print ("receive wrong dataID: ", item[3], "from process ", item[4])
            acc += item[0]
            cnt += item[1]
            tot += item[2]
            loss += item[5]
    except queue.Empty:
        print("The result of some process missed...")
        print(freeProcess.value)
        lock.release()
        time.sleep(2)
        print(freeProcess.value)
        while True:
            pass

    lock.release()
    return (acc, cnt, tot)


def test(dataID, model, datas, mode, dataQueue, resultQueue, freeProcess, lock, numprocess, device, sentiments, test):
    testmode = mode + ["test"]
    if dataID < -2:
        print(testmode)
    return train(-dataID-1, model, datas, 1, testmode, dataQueue, resultQueue, freeProcess, lock, numprocess, device, sentiments, test)

