import torch
import model as model
import data_preprocess as data_pre
import config
import os
import torch.nn as nn

SOS_token = 1  # 句子的开始
# 加载参数
con = config.Config()
# 加载数据集，生成词表
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)  # 把目录和文件名合成一个路径
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
voc, pairs = data_pre.loadPrepareData(corpus, corpus_name, datafile)
pairs = data_pre.trimRareWords(voc, pairs, MIN_COUNT=con.MIN_COUNT)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 贪心解码(Greedy decoding)算法
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Encoder的Forward计算
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 把Encoder最后时刻的隐状态作为Decoder的初始值
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # 因为我们的函数都是要求(time,batch)，因此即使只有一个数据，也要做出二维的。
        # Decoder的初始输入是SOS
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # 用于保存解码结果的tensor
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 循环，这里只使用长度限制，后面处理的时候把EOS去掉了。
        for _ in range(max_length):
            # Decoder forward一步
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden,
								encoder_outputs)
            # decoder_outputs是(batch=1, vob_size)
            # 使用max返回概率最大的词和得分
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 把解码结果保存到all_tokens和all_scores里
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # decoder_input是当前时刻输出的词的ID，这是个一维的向量，因为max会减少一维。
            # 但是decoder要求有一个batch维度，因此用unsqueeze增加batch维度。
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 返回所有的词和得分。
        return all_tokens, all_scores


# 测试对话函数
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=con.MAX_LENGTH):
    ### 把输入的一个batch句子变成id
    indexes_batch = [data_pre.indexesFromSentence(voc, sentence)]
    # 创建lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 放到合适的设备上(比如GPU)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # 用searcher解码
    tokens, scores = searcher(input_batch, lengths, max_length)
    # ID变成词。
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 得到用户终端的输入
            input_sentence = input('> ')
            # 是否退出
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 句子归一化
            input_sentence = data_pre.normalizeString(input_sentence)
            # 生成响应Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 去掉EOS后面的内容
            words = []
            for word in output_words:
                if word == 'EOS':
                    break
                elif word != 'PAD':
                    words.append(word)
            print('Bot:', ' '.join(words))

        except KeyError:
            print("Error: Encountered unknown word.")

def load_model():
    # 如果loadFilename不空，则从中加载模型
    if con.loadFilename:
        # 如果训练和加载是一条机器，那么直接加载
        checkpoint = torch.load(con.loadFilename)
        # 否则比如checkpoint是在GPU上得到的，但是我们现在又用CPU来训练或者测试，那么注释掉下面的代码
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # 初始化word embedding
    embedding = nn.Embedding(voc.num_words, con.hidden_size)
    if con.loadFilename:
        embedding.load_state_dict(embedding_sd)
    # 初始化encoder和decoder模型
    encoder = model.EncoderRNN(con.hidden_size, embedding, con.encoder_n_layers, con.dropout)
    decoder = model.LuongAttnDecoderRNN(con.attn_model, embedding, con.hidden_size, voc.num_words,
                                  con.decoder_n_layers, con.dropout)
    if con.loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # 使用合适的设备
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    return encoder,decoder

if __name__ == '__main__':
    encoder, decoder = load_model()
    # 进入eval模式，从而去掉dropout。
    encoder.eval()
    decoder.eval()
    # 构造searcher对象
    searcher = GreedySearchDecoder(encoder, decoder)
    # 测试
    evaluateInput(encoder, decoder, searcher, voc)