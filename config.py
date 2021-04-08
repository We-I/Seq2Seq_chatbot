import os
# 配置模型参数进行训练和测试
class Config(object):
    def __init__(self):

        # 模型配置参数
        self.model_name = 'cb_model'
        self.attn_model = 'dot'
        # attn_model = 'general'
        # attn_model = 'concat'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64


        # 训练配置参数
        # 配置训练的超参数和优化器
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.n_iteration = 4000
        self.print_every = 1
        self.save_every = 500
        self.save_dir = os.path.join("data","save")

        self.MAX_LENGTH = 10    # 句子最大长度是10个词(包括EOS等特殊词)
        self.MIN_COUNT =3    # 词表最小词频

        self.checkpoint_iter = 4000
        # 从哪个checkpoint恢复，如果是None，那么从头开始训练。
        self.loadFilename = os.path.join(self.save_dir, self.model_name, "cornell movie-dialogs corpus", '{}-{}_{}'
                                         .format(self.encoder_n_layers, self.decoder_n_layers, self.hidden_size),
                                         '{}_{}.tar'.format(self.checkpoint_iter, 'checkpoint'))

