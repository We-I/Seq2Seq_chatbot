import torch
import model as model
import data_preprocess as data_pre
import config
import os
import torch.nn as nn
from torch import optim
import random

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

"定义训练过程"
# Masked损失
def maskNLLLoss(inp, target, mask):
    # 计算实际的词的个数，因为padding是0，非padding是1，因此sum就可以得到词的个数
    nTotal = mask.sum()

    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# 一次迭代的训练过程
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=con.MAX_LENGTH):

    # 梯度清空
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 设置device，从而支持GPU，当然如果没有GPU也能工作。
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # encoder的Forward计算
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Decoder的初始输入是SOS，我们需要构造(1, batch)的输入，表示第一个时刻batch个输入。
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # 注意：Encoder是双向的，而Decoder是单向的，因此从下往上取n_layers个
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 确定是否teacher forcing
    use_teacher_forcing = True if random.random() < con.teacher_forcing_ratio else False

    # 一次处理一个时刻
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: 下一个时刻的输入是当前正确答案
            decoder_input = target_variable[t].view(1, -1)  # view(1, -1)：查看全部位置
            # 计算累计的loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 不是teacher forcing: 下一个时刻的输入是当前模型预测概率最高的值
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 计算累计的loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 反向计算
    loss.backward()

    # 对encoder和decoder进行梯度裁剪
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

# 训练迭代过程
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename):
    # 随机选择n_iteration个batch的数据(pair)
    training_batches = [data_pre.batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # 初始化
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # 训练
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # 训练一个batch的数据
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # 进度
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
                  .format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # 保存checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(con.save_dir, con.model_name, corpus_name, '{}-{}_{}'
                                     .format(con.encoder_n_layers, con.decoder_n_layers, con.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

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
    return encoder,decoder,embedding

# 训练

def begin_to_train(encoder, decoder, embedding):
    # 设置进入训练模式，从而开启dropout
    encoder.train()
    decoder.train()

    # 初始化优化器
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=con.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=con.learning_rate * con.decoder_learning_ratio)
    if con.loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # 开始训练
    print("Starting Training!")
    trainIters(con.model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, con.encoder_n_layers, con.decoder_n_layers, con.save_dir, con.n_iteration, con.batch_size,
               con.print_every, con.save_every, con.clip, corpus_name, con.loadFilename)

if __name__ == '__main__':
    encoder, decoder, embedding = load_model()
    begin_to_train(encoder, decoder, embedding)
