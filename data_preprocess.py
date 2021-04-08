from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda" if USE_CUDA else "cpu")

"加载和预处理数据"

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)  # 把目录和文件名合成一个路径

def printLines(file, n=10): # 定义函数查看数据集
    with open(file, 'rb') as datafile:  # read方式，binary mode，read()操作返回的是bytes
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# printLines(os.path.join(corpus, "movie_lines.txt"))


"数据处理：把原始的数据处理成一个对话一行，每轮对话语句用'\t'进行分隔"

def loadLines(fileName, fields):    # 把每一行的数据分割后存入一个字典，key是lineID，value是对应行的lineID, characterID, movieID, character, text的值
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:  # 一行一行进行循环
            values = line.split(" +++$+++ ")    # 返回分割后的列表values = [lineID, characterID, movieID, character, text]
            # 抽取fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]  # 每个数值对应相关的ID名称，lineObj = {"lineId":"L1045", "characterID":"u0", ..., "text":"They do not!"}
            lines[lineObj['lineID']] = lineObj  # 一个lineID的值对应一个lineObj字典，lines = {"l1045":{"lineID:"L1045",..., "text":"They do not!"},...}
    return lines

def loadConversations(fileName, lines, fields): # fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # 抽取fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # convObj["utteranceIDs"]是一个字符串，形如"['L198', 'L199']"
            # 我们用eval把这个字符串变成一个字符串的list。
            lineIds = eval(convObj["utteranceIDs"]) # 转换后lineIds变成一个数组['L198', 'L199']
            # 根据lineIds构造一个数组，根据lineId去lines里检索出存储utterance对象。
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            # convObj = {"character1ID":..., "character2ID":..., "movieID":..., "utteranceIDs":...,"lines":[lines1,lines2,...]}
            conversations.append(convObj)
    return conversations

def extractSentencePairs(conversations):    # 从对话中提取问答句子对
    qa_pairs = []
    for conversation in conversations:
        # 遍历对话中的每一个句子，忽略最后一个句子，因为没有答案。
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()    # .strip()去除首尾空格
            targetLine = conversation["lines"][i+1]["text"].strip()
            # 如果有空的句子就去掉
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

def build_new_datafile():
    # 定义新的文件
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # 对分隔符delimiter进行decode，这里对tab进行decode结果并没有变
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # 初始化dict lines，list conversations以及前面我们介绍过的field的id数组。
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # 首先使用loadLines函数处理movie_lines.txt
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    # 接着使用loadConversations处理上一步的结果，得到conversations
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    # 输出到一个新的csv文件
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        # 使用extractSentencePairs从conversations里抽取句对。
        for pair in extractSentencePairs(conversations):    # pair = [q, a]
            writer.writerow(pair)   # 两个对话句子间加入'\t'

    # 输出一些行用于检查
    print("\nSample lines from file:")
    printLines(datafile)


"创建词典"

# 预定义的token
PAD_token = 0  # 表示padding
SOS_token = 1  # 句子的开始
EOS_token = 2  # 句子的结束

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}    # 词到ID的映射
        self.word2count = {}    # 记录词频
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}    # 从ID到词的映射
        self.num_words = 3  # 词的数量，目前有SOS, EOS, PAD这3个token

    def addSentence(self, sentence):    # 把句子分割成一个个词加入词表
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):    # 对句子中的每一个单词进行操作
        if word not in self.word2index: # 给词表中没有的单词进行
            self.word2index[word] = self.num_words  # 根据单词出现顺序给单词ID
            self.word2count[word] = 1   # 词表中没有出现过的d
            self.index2word[self.num_words] = word  # 从3开始index2word
            self.num_words += 1 # 单词+1
        else:
            self.word2count[word] += 1  # 若词表已存在该单词，则频率+1

    # 删除频次小于min_count的token
    def trim(self, min_count):
        if self.trimmed:    # 判断是否经过删除
            return
        self.trimmed = True # 这一次过后就标识为经过删除

        # sorted(self.word2count.items(), key=lambda kv: (kv[1], kv[0]))  # 进行一次词频的排序，把高词频的单词放到前边

        keep_words = [] # 用来保存符合词频条件的单词

        for k, v in self.word2count.items():    # 筛选词频小于min_count的单词
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))  # 打印筛选前后的词表长度和对应的比例

        # 重新构造词典
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        # 重新构造后词频就都是1
        for word in keep_words:
            self.addWord(word)

# 创建词表前的预处理

# MAX_LENGTH = 10 # 句子最大长度是10个词(包括EOS等特殊词)
# 把Unicode字符串变成ASCII，比如把à变成a，只用于处理西方文字，若是处理中文数据则不需要这个函数
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )   # 逐词转换中间不加字符

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())   # lower()变成小写、strip()去掉字符串前后指定字符（默认空格），然后unicode变成ascii
    s = re.sub(r"([.!?])", r" \1", s)   # 进行检索替换，在标点.!?前增加空格，这样把标点当成一个词，\1表示字符本身
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)   # 将字母和标点之外的字符都变成空格
    # 因为把不用的字符都变成空格，所以可能存在多个连续空格
    # 下面的正则替换把多个空格变成一个空格，最后去掉前后空格
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# 读取问答句对并且返回Voc词典对象
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # 文件每行读取到list lines中。
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')  # 返回一个一行对话为一个元素的列表
    # 每行用tab切分成问答两个句子，然后调用normalizeString函数进行处理。
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p, MAX_LENGTH=10):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 过滤太长的句对
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 使用上面的函数进行处理，返回Voc对象和句对的list
def loadPrepareData(corpus, corpus_name, datafile):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:  # 一个句子对有两个语句
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

# 调用函数实现以上操作
# voc, pairs = loadPrepareData(corpus, corpus_name, datafile)     # pairs = [[pair1,pari2],...]

# MIN_COUNT = 3    # 阈值为3

def trimRareWords(voc, pairs, MIN_COUNT):
    # 去掉voc中频次小于3的词
    voc.trim(MIN_COUNT)
    # 保留的句对
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 检查问题
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 检查答案
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # 如果问题和答案都只包含高频词，我们才保留这个句对
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),
		len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

# 实际进行处理
# pairs = trimRareWords(voc, pairs, MIN_COUNT)


"为模型准备数据"

# 把句子的词转化成ID
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token] # 使用空格分开词，取其在词表中对应的序号最后加上结束符

# 把列表s_list中不同长度的句子(装有batch个句子）pad成固定长度，长度为最大句子长度
def zeroPadding(s_list, fill_value=PAD_token):
    return list(itertools.zip_longest(*s_list, fillvalue=fill_value))       # 取列表里的列表的每一列作为一个列表的第一个列表，不够长度的补pad，从(batch, sentence)-> (sentence, batch)

# 把装有转换好、pad好的句子的列表（2维），根据有无pad返回mask，有pad的地方为0否则为1
def binaryMatrix(s_list, value=PAD_token):
    mask = []
    for i, seq in enumerate(s_list):
        mask.append([])     # 列表里面新建列表
        for token in seq:
            if token == PAD_token:
                mask[i].append(0)
            else:
                mask[i].append(1)
    return mask

# 对输入句子进行处理：转换成ID、padding、记录每个句子的实际长度
def inputVar(s_list, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in s_list]       # 把对应的句子转成id
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])     #计算每一个句子的长度，一个一维的列表
    padList = zeroPadding(indexes_batch)        # 对转化好的句子进行padding
    padVar = torch.LongTensor(padList)      # 使用tensor记录padding好的句子，shape是(batch, max_length)
    return padVar, lengths

# 对输出句子进行处理，对batch个输出句子列表进行操作
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]      # 转换ID
    max_target_len = max([len(indexes) for indexes in indexes_batch])       # 取最大长度
    padList = zeroPadding(indexes_batch)        # padding
    mask = binaryMatrix(padList)        # 算出mask
    mask = torch.ByteTensor(mask)       # 进行tensor
    padVar = torch.LongTensor(padList)      # 构建一个2*3 Long类型的张量
    return padVar, mask, max_target_len

# 调用上述函数处理一个batch的pair句对
def batch2TrainData(voc, pair_batch):
    # 按照句子的长度(词数)排序
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)       # 按照第一个句子，降序排列
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    input, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return input, lengths, output, mask, max_target_len

# 调用示例
# small_batch_size = 5
# batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])     # '_'只起循环的作用
# input_variable, lengths, target_variable, mask, max_target_len = batches
#
# print("input_variable:", input_variable)
# print("lengths:", lengths)
# print("target_variable:", target_variable)
# print("mask:", mask)
# print("max_target_len:", max_target_len)


