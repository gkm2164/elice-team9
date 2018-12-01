import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os


class Seq2Seq(object):
    def __init__(self, flags, iterator, encoder_word_embeddings, decoder_word_embeddings):

        '''
        :param flags: 설정값들이 입력되어 있는 tf.flags
        :param iterator: tf.Data를 사용하여 데이터를 받아오는 interator
        :param encoder_word_embeddings: 입력에서 사용되는 word embeddings
        :param decoder_word_embeddings: 출력에서 사용되는 word embeddings
        '''

        self.flags = flags
        self.iterator = iterator
        self.encoder_word_embeddings = encoder_word_embeddings
        self.decoder_word_embeddings = decoder_word_embeddings
        # parameter 사전 설정
        # global_step : 한번 학습할 때마다 1씩 상승 하는 값
        # learning_rate : 학습률
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.Variable(initial_value=self.flags.learning_rate, trainable=False)
        # max_encoder_length : 입력(encoder) 문장의 최대 길이
        # max_decoder_length : 출력(decoder) 문장의 최대 길이
        # encoder_vocab_size : 입력 문장들의 단어 사전
        # decoder_vocab_size : 출력 문장들의 단어 사전
        # embedding_size : word embedding의 벡터 사이즈
        # hidden_size : 모델에서 사용되는 중간계층(hidden layer 및 여러 벡터들)의 벡터 사이즈
        # start_idx : 문장의 시작을 나타내는 심볼의 번호
        # end_idx : 문장의 끝을 나타내는 심볼의 번호, Decoding 때 사용됨
        # batch_size : 한번에 학습할 데이터의 수
        # beam_width : Beam Search시 beam size
        # attention_hidden_size : attention에서 사용될 hidden 벡터의 수
        # attention type : Attention 종류
        # PAD_ID : padding이 임베딩에서 나타내는 index번호
        # mode : 학습, 예측 플래그
        self.max_encoder_length = self.flags.max_encoder_length
        self.max_decoder_length = self.flags.max_decoder_length
        self.encoder_vocab_size = len(encoder_word_embeddings)
        self.decoder_vocab_size = len(decoder_word_embeddings)
        self.embedding_size = self.flags.embedding_size
        self.hidden_size = self.flags.hidden_size
        self.start_idx = self.flags.start_idx
        self.end_idx = self.flags.end_idx
        self.batch_size = self.flags.batch_size
        self.beam_width = self.flags.beam_width
        self.attention_hidden_size = self.flags.attention_hidden_size
        self.attention_type = self.flags.attention_type
        self.PAD_ID = self.flags.PAD_ID
        self.mode = self.flags.mode

        self._input_init()
        self._target_init()
        self._embedding_init()
        self._encoder_init()
        self._attention_init()
        self._decoder_init()
        self._predict_init()
        self._train_init()

    def _input_init(self):
        # tf.Data API를 활용하여 parsing된 encoder 입력의 단어 번호벡터들과 그와 쌍을 이루는 답변의 번호벡터를 가져옴
        self.encoder_inputs, self.decoder_inputs = self.iterator.get_next()

        # RNN에 사용될 length, 입력된 문장의 길이를 나타냄
        self.encoder_length = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.encoder_inputs, self.PAD_ID)), 1,
            name="encoder_input_lengths")

        # RNN 및 decoding때 사용될 length, 출력될 문장의 길이를 나타냄(학습시)
        self.decoder_length = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.decoder_inputs, self.PAD_ID)), 1,
            name="decoder_input_lengths")

        # loss function을 통해 loss 값을
        self.decoder_weights = tf.sequence_mask(
            lengths=self.decoder_length,
            maxlen=self.max_decoder_length, dtype=tf.float32, name='weight')

        # keep_prob는 1.0 - dropout확률을 말함 즉, 노드를 보전할 확률
        self.keep_prob = tf.placeholder(tf.float32, [], "keep_prob")

    def _target_init(self):
        if self.mode == "test":
            self.decoder_targets = None
        else:
            self.batch_size = tf.shape(self.encoder_inputs)[0]
            decoder_input_shift = tf.slice(self.decoder_inputs, [0, 1], [self.batch_size, self.max_decoder_length - 1])
            pad_tokens = tf.zeros([self.batch_size, 1], dtype=tf.int32)
            self.decoder_targets = tf.concat([decoder_input_shift, pad_tokens], axis=1)

            self.encoder_max = tf.reduce_max(self.encoder_length)
            self.decoder_max = tf.reduce_max(self.decoder_length)

            # optimize
            self.encoder_inputs = tf.slice(self.encoder_inputs, [0, 0], [self.batch_size, self.encoder_max])
            self.decoder_targets = tf.slice(self.decoder_targets, [0, 0], [self.batch_size, self.decoder_max])
            self.decoder_weights = tf.slice(self.decoder_weights, [0, 0], [self.batch_size, self.decoder_max])

    def _embedding_init(self):
        with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
            self.encoder_embeddings = tf.get_variable("encoder_embeddings",
                                                      shape=[self.encoder_vocab_size, self.embedding_size],
                                                      dtype=tf.float32, trainable=True,
                                                      initializer=tf.constant_initializer(self.encoder_word_embeddings))
            self.decoder_embeddings = tf.get_variable("decoder_embeddings",
                                                      shape=[self.encoder_vocab_size, self.embedding_size],
                                                      dtype=tf.float32, trainable=True,
                                                      initializer=tf.constant_initializer(self.encoder_word_embeddings))

    def _encoder_init(self):
        with tf.variable_scope("encoder_layer"):
            encoder_lookup_inputs = tf.nn.embedding_lookup(self.encoder_embeddings, self.encoder_inputs)
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)			
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=encoder_lookup_inputs,
                sequence_length=self.encoder_length, dtype=tf.float32, time_major=False)

            self.encoder_outputs = tf.concat([fw_outputs, bw_outputs], -1)
            self.encoder_output_state = tf.concat([fw_state[-1], bw_state[-1]], -1)

    def _attention_init(self):
        with tf.variable_scope("attention_layer"):
            if self.mode == "test":
                self.encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, self.beam_width)
                self.encoder_length = tf.contrib.seq2seq.tile_batch(self.encoder_length, self.beam_width)
            if self.attention_type == "luong":
                self.mechanism = tf.contrib.seq2seq.LuongAttention(self.attention_hidden_size,
                                                                   self.encoder_outputs,
                                                                   memory_sequence_length=self.encoder_length)
            else:
                self.mechanism = tf.contrib.seq2seq.BahdanauAttention(self.attention_hidden_size,
                                                                      self.encoder_outputs,
                                                                      memory_sequence_length=self.encoder_length)

    def _decoder_init(self):
        with tf.variable_scope("decode_layer"):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size * 2)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            cell = tf.contrib.seq2seq.AttentionWrapper(cell, self.mechanism,
                                                       attention_layer_size=self.attention_hidden_size,
                                                       alignment_history=False)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.decoder_vocab_size)

            if self.mode == "test" and self.beam_width > 0:
                decoder_start_state = tf.contrib.seq2seq.tile_batch(self.encoder_output_state, self.beam_width)
                init_state = cell.zero_state(self.batch_size * self.beam_width, dtype=tf.float32)
                init_state.clone(cell_state=decoder_start_state)
            else:
                init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
                init_state.clone(cell_state=self.encoder_output_state)

            if self.mode == "test":
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell, embedding=self.decoder_embeddings,
                                                               start_tokens=tf.fill([self.batch_size], self.start_idx),
                                                               end_token=self.end_idx, initial_state=init_state,
                                                               beam_width=self.beam_width)
            else:
                decoder_lookup_inputs = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_lookup_inputs,
                                                           sequence_length=self.decoder_length)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, init_state)

            self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                              maximum_iterations=self.max_decoder_length)
            if self.mode == "train":
                self.logits = self.outputs.rnn_output

    def _predict_init(self):
        with tf.variable_scope("predict"):
            if self.mode == "test":
                self.predict_op = self.outputs.predicted_ids[0]
            else:
                self.predict_op = self.outputs.sample_id

    def _train_init(self):
        if self.mode == "train":
            with tf.variable_scope("train_layer"):
                self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.decoder_targets, self.decoder_weights)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.grads = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(self.grads, global_step=self.global_step)


def get_record_parser(flags):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               "encoder_inputs": tf.FixedLenFeature([], tf.string),
                                               "decoder_inputs": tf.FixedLenFeature([], tf.string),
                                           })
        encoder_inputs = tf.reshape(tf.decode_raw(
            features["encoder_inputs"], tf.int32), [flags.max_encoder_length])
        decoder_inputs = tf.reshape(tf.decode_raw(
            features["decoder_inputs"], tf.int32), [flags.max_decoder_length])

        return encoder_inputs, decoder_inputs
    return parse


def get_batch_dataset(record_file, parser, flags):
    num_threads = tf.constant(flags.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(flags.capacity).repeat()

    dataset = dataset.batch(flags.batch_size)
    return dataset


def get_dataset(record_file, parser, flags):
    num_threads = tf.constant(flags.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        #parser, num_parallel_calls=num_threads).repeat().batch(flags.batch_size)
        parser, num_parallel_calls = num_threads).batch(flags.batch_size)
    return dataset


def prepro(flags):
    file_name = flags.source_data
    examples = []
    with open(file_name, "r", encoding='utf-8') as fp:
        source = json.load(fp)
        data_list = source["data"]
        for data in data_list:
            question = data["input_text"]
            answer = data["output_text"]
            example = {"question": question.split(), "answer": answer.split()}
            examples.append(example)

    vec_size = flags.embedding_size
    ###임베딩 파일을 사용하는 경우###
    emb_file = flags.embeddings_file

    embedding_dict = {}
    with open(emb_file, "r", encoding="utf8") as fp:
        for line in fp:
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            embedding_dict[word] = vector
    #################################

    ###데이터를 기반으로 벡터를 랜덤초기화로 생성하는 경우###
    # embedding_dict = {}
    # for example in examples:
    #     question = example["question"]
    #     answer = example["answer"]
    #     for token in question:
    #         if token not in embedding_dict:
    #             embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    #     for token in answer:
    #         if token not in embedding_dict:
    #             embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]

    word2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(), 4)}
    word2idx["--PAD--"] = 0
    word2idx["--UNK--"] = 1
    word2idx["--START--"] = 2
    word2idx["--END--"] = 3

    idx2word = {word2idx[word]: word for word in word2idx.keys()}

    embedding_dict["--PAD--"] = [0.0 for _ in range(vec_size)]
    embedding_dict["--UNK--"] = [0.0 for _ in range(vec_size)]
    embedding_dict["--START--"] = [0.0 for _ in range(vec_size)]
    embedding_dict["--END--"] = [0.0 for _ in range(vec_size)]

    idx2emb = {idx: embedding_dict[token] for token, idx in word2idx.items()}
    encoder_word_embeddings = [idx2emb[idx] for idx in range(len(idx2emb))]

    def _get_word_idx(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx:
                return word2idx[each]
        return word2idx["--UNK--"]

    writer = tf.python_io.TFRecordWriter(flags.record_file)
    for example in examples:
        encoder_inputs = np.zeros([flags.max_encoder_length], dtype=np.int32)
        decoder_inputs = np.zeros([flags.max_decoder_length], dtype=np.int32)

        for i, word in enumerate(example["question"]):
            if i < flags.max_encoder_length:
                encoder_inputs[i] = _get_word_idx(word)
            else:
                pass
        for i, word in enumerate(list(["--START--"] + example["answer"] + ["--END--"])):
            if i < flags.max_decoder_length:
                decoder_inputs[i] = _get_word_idx(word)
            else:
                pass

        record = tf.train.Example(features=tf.train.Features(feature={
            "encoder_inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoder_inputs.tostring()])),
            "decoder_inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[decoder_inputs.tostring()]))
        }))
        writer.write(record.SerializeToString())

    writer.close()

    def _save(filename, obj, message=None):
        if message is not None:
            print("Saving {}...".format(message))
            with open(filename, "w", encoding='utf-8') as fh:
                json.dump(obj, fh)

    _save(flags.word_emb_file, encoder_word_embeddings, "Embedding Json")
    _save(flags.word2idx_file, word2idx, "Word2idx Json")
    _save(flags.idx2word_file, idx2word, "Idx2Word Json")


def train(flags):
    print(flags.word_emb_file)
    with open(flags.word_emb_file, "r", encoding='utf-8') as fp:
        word_mat = np.array(json.load(fp), dtype=np.float32)

    print("Building model...")

    parser = get_record_parser(flags)
    train_data_set = get_batch_dataset(flags.record_file, parser, flags)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_data_set.output_types, train_data_set.output_shapes)
    train_iterator = train_data_set.make_one_shot_iterator()

    model = Seq2Seq(flags, iterator, word_mat, word_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        train_handle = sess.run(train_iterator.string_handle())

        for _ in tqdm(range(1, flags.num_steps + 1)):
            loss, train_op, global_step = sess.run([model.loss, model.train_op, model.global_step], feed_dict={
                                      handle: train_handle, model.keep_prob: flags.keep_prob})

            if global_step % flags.checkpoint == 0:
                filename = os.path.join(
                    flags.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def test(flags):
    with open(flags.word_emb_file, "r", encoding='utf-8') as fp:
        word_mat = np.array(json.load(fp), dtype=np.float32)

    with open(flags.idx2word_file, "r", encoding='utf-8') as fp:
        idx2word = json.load(fp)

    print("Building model...")

    parser = get_record_parser(flags)
    test_data_set = get_dataset(flags.record_file, parser, flags)
    test_iterator = test_data_set.make_one_shot_iterator()

    model = Seq2Seq(flags, test_iterator, word_mat, word_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(tf.train.latest_checkpoint(flags.save_dir))
        saver.restore(sess, tf.train.latest_checkpoint(flags.save_dir))

        def list2str(input_list):
            temp_list = []
            for e in input_list:
                if e == flags.end_idx or e == -1 or e == flags.PAD_ID:
                    break
                word = idx2word[str(e)]
                slash_idx = word.rfind('/')
                if slash_idx != -1:
                    temp_list.append(word[:slash_idx])
                else:
                    if word == "<sp>":
                        word = " "
                    temp_list.append(word)
            result_str = "".join(temp_list)
            return result_str

        while True:
            try:
                encoder_input, predict = sess.run([model.encoder_inputs, model.predict_op],
                                                  feed_dict={model.keep_prob: 1.0})

                input_str = list2str(encoder_input[0])
                predict_str = list2str(predict[:, 0])

                print("######################################")
                print("Input : ", input_str)
                print("Output : ", predict_str)
                print("######################################")
            except tf.errors.OutOfRangeError:
                break


def main(_):
    flags = tf.flags.FLAGS
    if flags.mode == "train":
        train(flags)
    elif flags.mode == "prepro":
        prepro(flags)
    elif flags.mode == "test":
        flags.batch_size = 1
        test(flags)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    target_dir = "../data"
    save_dir = "../model"
    # target_dir = "./drive/elice-team9/data"
    # save_dir = "./drive/elice-team9/model"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    source_data = os.path.join(target_dir, "dual_data_dict.json")
    embeddings_file = os.path.join(target_dir, "embedding.txt")
    record_file = os.path.join(target_dir, "record.tfrecords")
    word_emb_file = os.path.join(target_dir, "word_emb.json")
    word2idx_file = os.path.join(target_dir, "word2idx.json")
    idx2word_file = os.path.join(target_dir, "idx2word.json")

    tf.app.flags.DEFINE_string("mode", "train", "Running mode train/prepro/test")
    tf.app.flags.DEFINE_string("attention_type", "luong", "attention_type")

    tf.app.flags.DEFINE_string("source_data", source_data, "source_data")
    tf.app.flags.DEFINE_string("embeddings_file", embeddings_file, "embeddings_file")
    tf.app.flags.DEFINE_string("record_file", record_file, "record_file")
    tf.app.flags.DEFINE_string("word_emb_file", word_emb_file, "word_emb_file")
    tf.app.flags.DEFINE_string("word2idx_file", word2idx_file, "word2idx_file")
    tf.app.flags.DEFINE_string("idx2word_file", idx2word_file, "idx2word_file")
    tf.app.flags.DEFINE_string("save_dir", save_dir, "save_dir")

    tf.app.flags.DEFINE_integer("max_encoder_length", 100, "max_encoder_length")
    tf.app.flags.DEFINE_integer("max_decoder_length", 100, "max_decoder_length")
    tf.app.flags.DEFINE_integer("embedding_size", 200, "embedding_size")
    tf.app.flags.DEFINE_integer("hidden_size", 100, "hidden_size")
    tf.app.flags.DEFINE_integer("start_idx", 2, "start_idx")
    tf.app.flags.DEFINE_integer("end_idx", 3, "end_idx")
    tf.app.flags.DEFINE_integer("PAD_ID", 0, "PAD_ID")
    tf.app.flags.DEFINE_integer("UNK_ID", 1, "UNK_ID")
    tf.app.flags.DEFINE_integer("attention_hidden_size", 200, "attention_hidden_size")
    tf.app.flags.DEFINE_integer("beam_width", 20, "beam_width")
    tf.app.flags.DEFINE_integer("batch_size", 32, "batch_size")
    tf.app.flags.DEFINE_integer("num_threads", 4, "num_threads")
    tf.app.flags.DEFINE_integer("capacity", 15000, "capacity")
    tf.app.flags.DEFINE_integer("num_steps", 50000, "num_steps")
    tf.app.flags.DEFINE_integer("checkpoint", 100, "checkpoint")

    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
    tf.app.flags.DEFINE_float("keep_prob", 0.7, "keep_prob")

    tf.app.run()
