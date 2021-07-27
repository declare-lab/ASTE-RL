import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForQuestionAnswering


class TopModel(nn.Module):
    def __init__(self, dim, statedim, sent_count):
        super(TopModel, self).__init__()
        self.dim = dim
        self.hid2state = nn.Linear(dim + statedim + 1024 + 25, statedim)
        self.state2prob = nn.Linear(statedim, sent_count+1)

    def forward(self, pos_vec, top_word_vec, sent_vec, memory, training, dropout):
        inp = torch.cat([pos_vec, top_word_vec, sent_vec, memory], dim=1)
        outp = F.dropout(torch.tanh(self.hid2state(inp)), p=dropout, training=training)
        prob = F.softmax(self.state2prob(outp), dim=1)
        return outp, prob 


class BotAspectModel(nn.Module):
    def __init__(self, dim, statedim, sent_count):
        super(BotAspectModel, self).__init__()
        self.dim = dim
        self.hid2state = nn.Linear(dim + statedim*2 + 1024*2 + 25, statedim)
        self.state2probL = nn.ModuleList([nn.Linear(statedim, 3) for i in range(0, sent_count)])

    def forward(self, pos_vec, bot_bert_cls, aspect_vec, bot_word_vec, memory, sent, target, training, dropout): 
        inp = torch.cat([pos_vec, bot_bert_cls, bot_word_vec, aspect_vec, memory, target], dim=1)
        outp = F.dropout(torch.tanh(self.hid2state(inp)), p=dropout, training=training)
        prob = F.softmax(self.state2probL[sent-1](outp), dim=1)
        return outp, prob


class BotOpinionModel(nn.Module):
    def __init__(self, dim, statedim, sent_count):
        super(BotOpinionModel, self).__init__()
        self.dim = dim
        self.hid2state = nn.Linear(dim + statedim*2 + 1024*2 + 25, statedim)
        self.state2probL = nn.ModuleList([nn.Linear(statedim, 3) for i in range(0, sent_count)])

    def forward(self, pos_vec, bot_bert_cls, opinion_vec, bot_word_vec, memory, sent, target, training, dropout): 
        inp = torch.cat([pos_vec, bot_bert_cls, bot_word_vec, opinion_vec, memory, target], dim=1)
        outp = F.dropout(torch.tanh(self.hid2state(inp)), p=dropout, training=training)
        prob = F.softmax(self.state2probL[sent-1](outp), dim=1)
        return outp, prob 


class Model(nn.Module):
    def __init__(self, lr, dim, statedim, sent_count, dropout, all_pos_tags):
        super(Model, self).__init__()
        self.dim = dim
        self.statedim = statedim
        self.sent_count = sent_count
        self.topModel = TopModel(dim, statedim, sent_count)
        self.botAspectModel = BotAspectModel(dim, statedim, sent_count)
        self.botOpinionModel = BotOpinionModel(dim, statedim, sent_count)
        self.sentimentvector = nn.Embedding(sent_count+1, dim)
        self.posvector = nn.Embedding(len(all_pos_tags), 25)
        self.aspectvector = nn.Embedding(3, dim)
        self.opinionvector = nn.Embedding(3, dim)
        self.top2target = nn.Linear(statedim, statedim)
        self.top2bot = nn.Linear(statedim, statedim)
        self.bot2top = nn.Linear(statedim, statedim)
        self.dropout = dropout
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bertqa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').bert
        self.opinion2aspect = nn.Linear(statedim, statedim)
    
    def sample(self, prob, training, preoptions, position, device):
        if not training:
            # testing
            return torch.argmax(prob, 1)
        elif preoptions is not None:
            # pre-training
            return torch.LongTensor(1, ).fill_(preoptions[position]).to(device)
        else:
            # RL training
            return torch.squeeze(torch.multinomial(prob, 1), dim=0).to(device)

    def forward(self, mode, pos_tags, sentext, preoptions=None, pre_aspect_actions=None, pre_opinion_actions=None, device=torch.device("cpu"), sentiments=None):
        # POS tag vectors
        posin = torch.LongTensor(pos_tags).to(device)
        posvs = self.posvector(posin)
        posvs = torch.unsqueeze(posvs, dim=1)
        top_action, top_actprob = [], []
        bot_aspect_action, bot_aspect_actprob = [], []
        bot_opinion_action, bot_opinion_actprob = [], []
        training = True if "test" not in mode else False
        #-----------------------------------------------------------------
        # BERT encoding for high-level process
        right_input = self.tokenizer(sentext, return_tensors='pt').to(device)
        sentence_len = right_input['input_ids'].shape[1] - 2
        left_input = self.tokenizer("Which tokens indicate sentiments relating pairs of aspect spans and opinion spans?", return_tensors='pt').to(device)
        query_len = left_input['input_ids'].shape[1]
        if training:
            self.bertqa.train()
        else:
            self.bertqa.eval()
        two_sentence_inputs_ids = torch.cat([left_input['input_ids'], right_input['input_ids'][:,1:-1]],dim=1)
        two_sentence_token_type_ids = torch.cat([left_input['token_type_ids'], torch.ones_like(right_input['token_type_ids'])[:,1:-1]],dim=1)
        two_sentence_attention_mask = torch.cat([left_input['attention_mask'], right_input['attention_mask'][:,1:-1]],dim=1)
        output = self.bertqa(input_ids=two_sentence_inputs_ids, token_type_ids=two_sentence_token_type_ids, attention_mask=two_sentence_attention_mask).last_hidden_state
        wordintop = torch.unsqueeze(output[0,query_len:,:], dim=1)
        #------------------------------------------------------------------
        # First Layer
        mem = torch.FloatTensor(1, self.statedim, ).fill_(0).to(device)
        action = torch.LongTensor(1, ).fill_(0).to(device)
        sent_action = torch.LongTensor(1, ).fill_(0).to(device)
        for x in range(sentence_len):
            mem, prob = self.topModel(posvs[x], wordintop[x],\
                    self.sentimentvector(sent_action), mem, training, self.dropout)
            action = self.sample(prob, training, preoptions, x, device)
            # only change sent_action if sentiment identified
            if action.data[0] != 0:
                sent_action = action
            # get sentiment probability for chosen sentiment
            actprob = prob[0][action]
            top_action.append(action.cpu().data[0])
            if not training:
                top_actprob.append(actprob.cpu().data[0])
            else:
                top_actprob.append(actprob)
            #----------------------------------------------------------------
            # Second Layer
            if "NER" in mode and action.data[0] > 0:
                sent = action.data[0]
                # make use of state from higher layer state initialisations
                target = self.top2target(mem)
                mem = self.top2bot(mem)

                # get sentiment text
                sentiment_text = sentiments[sent-1]
                if sentiment_text == 'POS':
                    sentiment_text = 'positive'
                elif sentiment_text == 'NEG':
                    sentiment_text = 'negative'
                elif sentiment_text == 'NEU':
                    sentiment_text = 'neutral'
                sentiment_indicator_token = self.tokenizer.convert_ids_to_tokens(right_input['input_ids'][0,1:-1][x].item())

                # BERT encoding for low-level opinion process
                actionb = torch.LongTensor(1, ).fill_(0).to(device)
                actions, actprobs = [], []
                bot_left_sentence = "What is the opinion span for the {} sentiment indicated at {}?".format(sentiment_text, sentiment_indicator_token)
                left_input = self.tokenizer(bot_left_sentence, return_tensors='pt').to(device)
                query_len = left_input['input_ids'].shape[1]
                two_sentence_inputs_ids = torch.cat([left_input['input_ids'], right_input['input_ids'][:,1:-1]],dim=1)
                two_sentence_token_type_ids = torch.cat([left_input['token_type_ids'], torch.ones_like(right_input['token_type_ids'])[:,1:-1]],dim=1)
                two_sentence_attention_mask = torch.cat([left_input['attention_mask'], right_input['attention_mask'][:,1:-1]],dim=1)
                output = self.bertqa(input_ids=two_sentence_inputs_ids, token_type_ids=two_sentence_token_type_ids, attention_mask=two_sentence_attention_mask).last_hidden_state
                bot_bert_cls = torch.unsqueeze(output[0,0,:], dim=0)
                wordinbot = torch.unsqueeze(output[0,query_len:,:], dim=1)
                for y in range(sentence_len):
                    mem, probb = self.botOpinionModel(\
                            posvs[y], bot_bert_cls, self.opinionvector(actionb), wordinbot[y], \
                            mem, sent, target, training, self.dropout)
                    actionb = self.sample(probb, training, pre_opinion_actions[x] if pre_opinion_actions is not None else None, y, device)
                    actprobb = probb[0][actionb]
                    actions.append(actionb.cpu().data[0])
                    if not training:
                        actprobs.append(actprobb.cpu().data[0])
                    else:
                        actprobs.append(actprobb)
                bot_opinion_action.append(actions)
                bot_opinion_actprob.append(actprobs)

                # BERT encoding for low-level aspect process
                actionb = torch.LongTensor(1, ).fill_(0).to(device)
                actions, actprobs = [], []
                bot_left_sentence = "What is the aspect span for the {} sentiment indicated at {}?".format(sentiment_text, sentiment_indicator_token)
                left_input = self.tokenizer(bot_left_sentence, return_tensors='pt').to(device)
                query_len = left_input['input_ids'].shape[1]
                two_sentence_inputs_ids = torch.cat([left_input['input_ids'], right_input['input_ids'][:,1:-1]],dim=1)
                two_sentence_token_type_ids = torch.cat([left_input['token_type_ids'], torch.ones_like(right_input['token_type_ids'])[:,1:-1]],dim=1)
                two_sentence_attention_mask = torch.cat([left_input['attention_mask'], right_input['attention_mask'][:,1:-1]],dim=1)
                output = self.bertqa(input_ids=two_sentence_inputs_ids, token_type_ids=two_sentence_token_type_ids, attention_mask=two_sentence_attention_mask).last_hidden_state
                bot_bert_cls = torch.unsqueeze(output[0,0,:], dim=0)
                wordinbot = torch.unsqueeze(output[0,query_len:,:], dim=1)
                for y in range(sentence_len):
                    mem, probb = self.botAspectModel(\
                            posvs[y], bot_bert_cls, self.aspectvector(actionb), wordinbot[y], \
                            mem, sent, target, training, self.dropout)
                    actionb = self.sample(probb, training, pre_aspect_actions[x] if pre_aspect_actions is not None else None, y, device)
                    actprobb = probb[0][actionb]
                    actions.append(actionb.cpu().data[0])
                    if not training:
                        actprobs.append(actprobb.cpu().data[0])
                    else:
                        actprobs.append(actprobb)
                bot_aspect_action.append(actions)
                bot_aspect_actprob.append(actprobs)

                mem = self.bot2top(mem)
                
        return top_action, top_actprob, bot_aspect_action, bot_aspect_actprob, bot_opinion_action, bot_opinion_actprob