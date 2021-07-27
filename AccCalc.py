def calcF1(acc, cnt, tot, beta=1.0):
    """
    Get F1 score.
    """
    if cnt == 0 or tot == 0:
        return 0
    precision = float(acc) / float(cnt)
    recall = float(acc) / float(tot)
    if precision + recall < 1e-5:
        return 0
    return (1+beta*beta) * precision * recall / (beta*beta*precision + recall)


def calc_acc(top_action, bot_aspect_action, bot_opinion_action, gold_labels, mode):
    """
    Get accuracy.

    Args:
        top_action (list): List of predicted sentiments with their positions in a sequence.
        bot_action (list): List of list(s) of sequence length, where each list of predicted entity corresponds to a position in top_action, in respective order.
        gold_labels (list): List of dictionaries of ground truth labels.
        mode (list): List of experiment modes.
    """
    acc, cnt, tot = 0, 0, len(gold_labels)
    used = [0 for i in range(len(top_action))]
    for label in gold_labels:
        tp, aspect_tags, opinion_tags = label['type'], label['aspect_tags'], label['opinion_tags']
        j, ok = 0, 0
        for i in range(len(top_action)):
            # if sentiment matches ground truth, ground truth is a sentiment, position is not filled, and gold triplet has not been correctly matched before
            if top_action[i] == tp and tp > 0 and used[i] == 0 and ok == 0:
                match = 1
                if "NER" in mode:
                    # remove impossible predictions when not calculating rewards
                    # 1) skip if either beginning tag of aspect or opinion span is absent or there is more than 1 of either
                    if bot_aspect_action[j].count(2) !=1 or bot_opinion_action[j].count(2) !=1:
                        j += 1
                        continue
                    # 2) skip if no aspect or opinion span detected or more than 1 of either is detected
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
                    if aspect_cnt != 1 or opinion_cnt != 1:
                        j += 1
                        continue
                    # make sure there is an exact match for both aspect and opinion spans
                    if bot_aspect_action[j] != aspect_tags or bot_opinion_action[j] != opinion_tags:
                        match = 0
                if match == 1:
                    ok = 1
                    used[i] = 1
            if top_action[i] > 0:
                j += 1
                cnt += 1
        acc += ok
    cnt //= tot
    return acc, tot, cnt


def find_tail(tags, num):
    """
    Get the position of the end of the aspect/opinion span in a tagging sequence.

    Args:
        tags (list): List of tags in a sentence.
        num (int): Number to look out for that signals the end of an aspect/opinion.

    Returns:
        
    """
    last = False
    for i, x in enumerate(tags):
        if x != num and last:
            return i - 1
        if x == num + 1:
            last = True
    return len(tags)-1 if last else -1
    
    
def rule_actions(gold_labels):
    """
    Get gold actions.

    Args:
        gold_labels (list): List of ground-truth aspect/opinions and their sentiments.

    Returns:
        options (list): List of sentiments at their respective positions in the sentence.
        aspect_actions (list): List of aspect positions for each sentiment in options. The positions are in a list in the position of the aspects' sentiments in options.
        opinion_actions (list): List of opinion positions for each sentiment in options. The positions are in a list in the position of the opinions' sentiments in options.
    """
    length = len(gold_labels[0]['aspect_tags'])
    options = [0 for i in range(length)]
    aspect_actions = [[] for i in range(length)]
    opinion_actions = [[] for i in range(length)]
    for label in gold_labels:
        tp, aspect_tags, opinion_tags = label['type'], label['aspect_tags'], label['opinion_tags']
        # get aspect and opinion spans for each label in gold_labels for each data point
        aspect_span = find_tail(aspect_tags, 1)
        assert aspect_span != -1
        opinion_span = find_tail(opinion_tags, 1)
        assert opinion_span != -1
        pos = max(aspect_span, opinion_span)
        while pos < len(aspect_tags) and options[pos] != 0:
            pos += 1
        if pos != len(aspect_tags):
            options[pos] = tp
            aspect_actions[pos] = aspect_tags
            opinion_actions[pos] = opinion_tags
        else:
            pos = max(aspect_span, opinion_span) - 1
            while pos >= 0 and options[pos] != 0:
                pos -= 1
            if pos != -1:
                options[pos] = tp
                aspect_actions[pos] = aspect_tags
                opinion_actions[pos] = opinion_tags
    return options, aspect_actions, opinion_actions	
				
