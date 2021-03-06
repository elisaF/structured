from __future__ import division
import tensorflow as tf
import math
from neural import dynamicBiRNN, LReLu, MLP, get_structure
import numpy as np


class StructureModel():
    def __init__(self, config, xavier_init):
        self.config = config
        self.xavier_init = xavier_init
        t_variables = {}
        t_variables['keep_prob'] = tf.placeholder(tf.float64)
        t_variables['batch_l'] = tf.placeholder(tf.int32)
        t_variables['token_idxs'] = tf.placeholder(tf.int32, [None, None, None])
        t_variables['sent_l'] = tf.placeholder(tf.int32, [None, None])
        t_variables['doc_l'] = tf.placeholder(tf.int32, [None])
        t_variables['max_sent_l'] = tf.placeholder(tf.int32)
        t_variables['max_doc_l'] = tf.placeholder(tf.int32)
        t_variables['gold_labels'] = tf.placeholder(tf.int32, [None])
        t_variables['doc_ids'] = tf.placeholder(tf.int32, [None])
        t_variables['mask_tokens'] = tf.placeholder(tf.float64, [None, None, None])
        t_variables['mask_sents'] = tf.placeholder(tf.float64, [None, None])
        t_variables['mask_parser_1'] = tf.placeholder(tf.float64, [None, None, None])
        t_variables['mask_parser_2'] = tf.placeholder(tf.float64, [None, None, None])
        self.t_variables = t_variables

    def get_feed_dict(self, batch):
        batch_size = len(batch)
        doc_l_matrix = np.zeros([batch_size], np.int32)
        for i, instance in enumerate(batch):
            n_sents = len(instance.token_idxs)
            doc_l_matrix[i] = n_sents
        max_doc_l = np.max(doc_l_matrix)
        max_sent_l = max([max([len(sent) for sent in doc.token_idxs]) for doc in batch])
        token_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l], np.int32)
        sent_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)
        gold_matrix = np.zeros([batch_size], np.int32)
        id_matrix = np.zeros([batch_size], np.int32)
        mask_tokens_matrix = np.ones([batch_size, max_doc_l, max_sent_l], np.float64)
        mask_sents_matrix = np.ones([batch_size, max_doc_l], np.float64)
        for i, instance in enumerate(batch):
            n_sents = len(instance.token_idxs)
            gold_matrix[i] = instance.goldLabel
            id_matrix[i] = instance.id
            for j, sent in enumerate(instance.token_idxs):
                token_idxs_matrix[i, j, :len(sent)] = np.asarray(sent)
                mask_tokens_matrix[i, j, len(sent):] = 0
                sent_l_matrix[i, j] = len(sent)
            mask_sents_matrix[i, n_sents:] = 0
        mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float64)
        mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float64)
        mask_parser_1[:, :, 0] = 0  # zero out 1st column for each doc
        mask_parser_2[:, 0, :] = 0  # zero out 1st row for each doc
        if self.config.large_data:
            if (batch_size * max_doc_l * max_sent_l * max_sent_l > 16 * 200000):
                return [batch_size * max_doc_l * max_sent_l * max_sent_l / (16 * 200000) + 1]

        feed_dict = {self.t_variables['token_idxs']: token_idxs_matrix, self.t_variables['sent_l']: sent_l_matrix,
                     self.t_variables['mask_tokens']: mask_tokens_matrix, self.t_variables['mask_sents']: mask_sents_matrix,
                     self.t_variables['doc_l']: doc_l_matrix, self.t_variables['gold_labels']: gold_matrix,
                     self.t_variables['doc_ids']: id_matrix,
                     self.t_variables['max_sent_l']: max_sent_l, self.t_variables['max_doc_l']: max_doc_l,
                     self.t_variables['mask_parser_1']: mask_parser_1, self.t_variables['mask_parser_2']: mask_parser_2,
                     self.t_variables['batch_l']: batch_size, self.t_variables['keep_prob']:self.config.keep_prob}
        return feed_dict

    def build(self):
        with tf.variable_scope("Embeddings"):
            self.embeddings = tf.get_variable("emb", [self.config.n_embed, self.config.d_embed], dtype=tf.float64,
                                         initializer=self.xavier_init)
            embeddings_root = tf.get_variable("emb_root", [1, 1, 2 * self.config.dim_sem], dtype=tf.float64,
                                                  initializer=self.xavier_init)
            embeddings_root_s = tf.get_variable("emb_root_s", [1, 1,2* self.config.dim_sem], dtype=tf.float64,
                                                    initializer=self.xavier_init)
        with tf.variable_scope("Model"):
            w_comb = tf.get_variable("w_comb", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float64,
                            initializer=self.xavier_init)
            w_comb_both = tf.get_variable("w_comb_both", [6 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float64,
                                     initializer=self.xavier_init)
            b_comb = tf.get_variable("bias_comb", [2 * self.config.dim_sem], dtype=tf.float64, initializer=tf.constant_initializer())

            w_comb_s = tf.get_variable("w_comb_s", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float64,
                            initializer=self.xavier_init)
            b_comb_s = tf.get_variable("bias_comb_s", [2 * self.config.dim_sem], dtype=tf.float64, initializer=tf.constant_initializer())

            w_softmax = tf.get_variable("w_softmax", [2 * self.config.dim_sem, self.config.dim_output], dtype=tf.float64,
                            initializer=self.xavier_init)
            b_softmax = tf.get_variable("bias_softmax", [self.config.dim_output], dtype=tf.float64,
                            initializer=self.xavier_init)

            w_sem_doc = tf.get_variable("w_sem_doc", [2 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float64,
                                        initializer=self.xavier_init)

            w_str_doc = tf.get_variable("w_str_doc", [2 * self.config.dim_sem, 2 * self.config.dim_str], dtype=tf.float64,
                                        initializer=self.xavier_init)

        with tf.variable_scope("Structure/doc"):
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float64,
                            initializer=self.xavier_init)
        with tf.variable_scope("Structure/sent"):
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float64,
                            initializer=self.xavier_init)

            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float64,
                            initializer=self.xavier_init)
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float64,
                            initializer=self.xavier_init)

        sent_l = self.t_variables['sent_l']
        doc_l = self.t_variables['doc_l']
        max_sent_l = self.t_variables['max_sent_l']
        max_doc_l = self.t_variables['max_doc_l']
        batch_l = self.t_variables['batch_l']

        tokens_input = tf.nn.embedding_lookup(self.embeddings, self.t_variables['token_idxs'][:, :max_doc_l, :max_sent_l])
        tokens_input = tf.nn.dropout(tokens_input, self.t_variables['keep_prob'])  # [batch_size, doc_l, sent_l, d_embed]

        mask_tokens = self.t_variables['mask_tokens'][:, :max_doc_l, :max_sent_l]
        mask_sents = self.t_variables['mask_sents'][:, :max_doc_l]  # [batch_size, doc_l]

        tokens_input_do = tf.reshape(tokens_input, [batch_l * max_doc_l, max_sent_l, self.config.d_embed])
        sent_l = tf.reshape(sent_l, [batch_l * max_doc_l])
        mask_tokens = tf.reshape(mask_tokens, [batch_l * max_doc_l, -1])
        tokens_output, _ = dynamicBiRNN(tokens_input_do, sent_l, n_hidden=self.config.dim_hidden, xavier_init=self.xavier_init,
                                        cell_type=self.config.rnn_cell, cell_name='Model/sent')
        tokens_sem = tf.concat([tokens_output[0][:,:,:self.config.dim_sem], tokens_output[1][:,:,:self.config.dim_sem]], 2)
        tokens_str = tf.concat([tokens_output[0][:,:,self.config.dim_sem:], tokens_output[1][:,:,self.config.dim_sem:]], 2)

        if self.config.skip_sent_attention:
            tokens_output = LReLu(tf.tensordot(tf.concat([tokens_sem, tokens_input_do], 2), w_comb_s, [[2], [0]]) + b_comb_s)
        else:
            temp1 = tf.zeros([batch_l * max_doc_l, max_sent_l,1], tf.float64)
            temp2 = tf.zeros([batch_l * max_doc_l,1,max_sent_l], tf.float64)

            mask1 = tf.ones([batch_l * max_doc_l, max_sent_l, max_sent_l-1], tf.float64)
            mask2 = tf.ones([batch_l * max_doc_l, max_sent_l-1, max_sent_l], tf.float64)
            mask1 = tf.concat([temp1,mask1],2)
            mask2 = tf.concat([temp2,mask2],1)

            if self.config.skip_mask_bug_fix:
                str_scores_s_, _, LL_tokens = get_structure('sent', tokens_str, mask1, mask2, None, None, None)  # batch_l,  sent_l+1, sent_l
            else:
                # create mask for setting all padded cells to 0
                mask_ll_tokens = tf.expand_dims(mask_tokens, 2)
                mask_ll_tokens_trans = tf.transpose(mask_ll_tokens, perm=[0, 2, 1])
                mask_ll_tokens = mask_ll_tokens
                mask_tokens_mult = mask_ll_tokens * mask_ll_tokens_trans

                # create mask for setting the padded diagonals to 1
                mask_diags = tf.matrix_diag_part(mask_tokens_mult)
                mask_diags_invert = tf.cast(tf.logical_not(tf.cast(mask_diags, tf.bool)), tf.float64)
                zero_matrix = tf.zeros([batch_l * max_doc_l, max_sent_l, max_sent_l], tf.float64)
                mask_tokens_add = tf.matrix_set_diag(zero_matrix, mask_diags_invert)

                str_scores_s_, _, LL_tokens = get_structure('sent', tokens_str, mask1, mask2, mask_tokens_mult,
                                                            mask_tokens_add, tf.expand_dims(mask_tokens,
                                                                                            2))  # batch_l,  sent_l+1, sent_l

            str_scores_s = tf.matrix_transpose(str_scores_s_)  # soft parent
            tokens_sem_root = tf.concat([tf.tile(embeddings_root_s, [batch_l * max_doc_l, 1, 1]), tokens_sem], 1)
            tokens_output_ = tf.matmul(str_scores_s, tokens_sem_root)
            tokens_output = LReLu(tf.tensordot(tf.concat([tokens_sem, tokens_output_], 2), w_comb_s, [[2], [0]]) + b_comb_s)

        if (self.config.sent_attention == 'sum'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)
        elif (self.config.sent_attention == 'mean'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)/tf.expand_dims(tf.cast(sent_l,tf.float64),1)
        elif (self.config.sent_attention == 'max'):
            tokens_output = tokens_output + tf.expand_dims((mask_tokens-1)*999,2)
            tokens_output = tf.reduce_max(tokens_output, 1)

        # batch_l * max_doc_l, 200
        if self.config.skip_doc_bilstm:
            if self.config.use_positional_encoding:
                tokens_output = tf.reshape(tokens_output, [batch_l, max_doc_l, 2 * self.config.dim_sem])
                tokens_output = self.add_timing_signal(tokens_output, max_doc_l, num_timescales=self.config.dim_sem)
                tokens_output = tf.reshape(tokens_output, [batch_l * max_doc_l, 2 * self.config.dim_sem])

            sents_sem = tf.matmul(tokens_output, w_sem_doc)
            sents_sem = tf.reshape(sents_sem, [batch_l, max_doc_l, 2 * self.config.dim_sem])
            sents_str = tf.matmul(tokens_output, w_str_doc)
            sents_str = tf.reshape(sents_str, [batch_l, max_doc_l, 2 * self.config.dim_str])
        else:
            sents_input = tf.reshape(tokens_output, [batch_l, max_doc_l, 2 * self.config.dim_sem])
            sents_output, _ = dynamicBiRNN(sents_input, doc_l, n_hidden=self.config.dim_hidden, xavier_init=self.xavier_init, 
                                           cell_type=self.config.rnn_cell, cell_name='Model/doc')
            sents_sem = tf.concat([sents_output[0][:,:,:self.config.dim_sem], sents_output[1][:,:,:self.config.dim_sem]], 2)  # [batch_l, doc+l, dim_sem*2]
            sents_str = tf.concat([sents_output[0][:,:,self.config.dim_sem:], sents_output[1][:,:,self.config.dim_sem:]], 2)  # [batch_l, doc+l, dim_str*2]

        if self.config.skip_doc_attention:
            if self.config.skip_doc_bilstm:
                sents_input = tf.reshape(tokens_output, [batch_l, max_doc_l, 2 * self.config.dim_sem])
                sents_output = LReLu(tf.tensordot(tf.concat([sents_sem, sents_input], 2), w_comb, [[2], [0]]) + b_comb)
            else:
                sents_output = LReLu(tf.tensordot(tf.concat([sents_sem, sents_input], 2), w_comb, [[2], [0]]) + b_comb)
        else:
            if self.config.skip_mask_bug_fix:
                str_scores_, str_scores_no_root, LL_sents = get_structure('doc', sents_str, self.t_variables['mask_parser_1'],
                                                                  self.t_variables['mask_parser_2'], None, None, None)  # [batch_size, doc_l+1, doc_l]
            else:
                # create mask for setting all padded cells to 0
                mask_ll_sents = tf.expand_dims(mask_sents, 2)
                mask_ll_sents_trans = tf.transpose(mask_ll_sents, perm=[0, 2, 1])
                mask_ll_sents = mask_ll_sents
                mask_sents_mult = mask_ll_sents * mask_ll_sents_trans

                # create mask for setting the padded diagonals to 1
                mask_sents_diags = tf.matrix_diag_part(mask_sents_mult)
                mask_sents_diags_invert = tf.cast(tf.logical_not(tf.cast(mask_sents_diags, tf.bool)), tf.float64)
                zero_matrix_sents = tf.zeros([batch_l, max_doc_l, max_doc_l], tf.float64)
                mask_sents_add = tf.matrix_set_diag(zero_matrix_sents, mask_sents_diags_invert)

                str_scores_, str_scores_no_root, LL_sents = get_structure('doc', sents_str, self.t_variables['mask_parser_1'],
                                                                  self.t_variables['mask_parser_2'], mask_sents_mult,
                                                                  mask_sents_add, tf.expand_dims(mask_sents,
                                                                                                 2))  # [batch_size, doc_l+1, doc_l]

            str_scores = tf.matrix_transpose(str_scores_)
            self.str_scores = str_scores  # shape is [batch_size, doc_l, doc_l+1]

            sents_children = tf.matmul(str_scores_no_root, sents_sem)
            if self.config.tree_percolation == "child":
                sents_output = LReLu(tf.tensordot(tf.concat([sents_sem, sents_children], 2), w_comb, [[2], [0]]) + b_comb)
            else:
                sents_sem_root = tf.concat([tf.tile(embeddings_root, [batch_l, 1, 1]), sents_sem], 1)
                sents_parents = tf.matmul(str_scores, sents_sem_root)
                if self.config.tree_percolation == "parent":
                    sents_output = LReLu(tf.tensordot(tf.concat([sents_sem, sents_parents], 2), w_comb, [[2], [0]]) + b_comb)
                elif self.config.tree_percolation == "both":
                    sents_output = LReLu(tf.tensordot(tf.concat([sents_sem, sents_parents, sents_children], 2), w_comb_both, [[2], [0]]) + b_comb)

            # percolation is only supported for "child" option
            if self.config.tree_percolation_levels > 0:
                count = 0
                while count < self.config.tree_percolation_levels:
                    sents_children_2 = tf.matmul(str_scores_no_root, sents_output)
                    sents_output = LReLu(tf.tensordot(tf.concat([sents_output, sents_children_2], 2), w_comb, [[2], [0]]) + b_comb)
                    count += 1

        if (self.config.doc_attention == 'sum'):
            sents_output = sents_output * tf.expand_dims(mask_sents, 2)  # mask is [batch_size, doc_l, 1]
            sents_output = tf.reduce_sum(sents_output, 1)  # [batch_size, dim_sem*2]
        elif (self.config.doc_attention == 'mean'):
            sents_output = sents_output * tf.expand_dims(mask_sents, 2)
            sents_output = tf.reduce_sum(sents_output, 1)/tf.expand_dims(tf.cast(doc_l,tf.float64),1)
        elif (self.config.doc_attention == 'max'):
            sents_output = sents_output + tf.expand_dims((mask_sents-1)*999,2)
            sents_output = tf.reduce_max(sents_output, 1)
        elif (self.config.doc_attention == 'weighted_sum'):
            sents_weighted = sents_output * tf.expand_dims(str_scores[:,:,0], 2)
            sents_output = sents_weighted * tf.expand_dims(mask_sents, 2)  # apply mask
            sents_output = tf.reduce_sum(sents_output, 1)

        final_output = MLP(sents_output, 'output', self.t_variables['keep_prob'], self.config.seed, self.xavier_init)
        self.final_output = tf.matmul(final_output, w_softmax) + b_softmax

    def get_loss(self):
        if (self.config.opt == 'Adam'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif (self.config.opt == 'Adagrad'):
            optimizer = tf.train.AdagradOptimizer(self.config.lr)
        with tf.variable_scope("Model"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.final_output,
                                                                  labels=self.t_variables['gold_labels'])
            self.loss = tf.reduce_mean(self.loss)
            model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Model')
            str_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Structure')
            for p in model_params + str_params:
                if ('bias' not in p.name):
                    self.loss += self.config.norm * tf.nn.l2_loss(p)
            if self.config.clip_ratio > 0:
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip_ratio)
                self.opt = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.opt = optimizer.minimize(self.loss)
    # blatantly copied from https://github.com/tensorflow/tensor2tensor/
    def get_timing_signal(self, length,
                          min_timescale=1,
                          max_timescale=1e4,
                          num_timescales=16):
        """Create Tensor of sinusoids of different frequencies.
        Args:
          length: Length of the Tensor to create, i.e. Number of steps.
          min_timescale: a float
          max_timescale: a float
          num_timescales: an int
        Returns:
          Tensor of shape (length, 2*num_timescales)
        """
        positions = tf.to_float(tf.range(length))
        log_timescale_increment = (
            math.log(max_timescale / min_timescale) / (num_timescales - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
        return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

    def add_timing_signal(self, x, length, min_timescale=1, max_timescale=1e4, num_timescales=16):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.
        This allows attention to learn to use absolute and relative positions.
        The timing signal should be added to some precursor of both the source
        and the target of the attention.
        The use of relative position is possible because sin(x+y) and cos(x+y) can be
        expressed in terms of y, sin(x) and cos(x).
        In particular, we use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the depth dimension, padded with zeros to be the same depth as the input,
        and added into input.
        Args:
          x: a Tensor with shape [?, length, ?, depth]
          min_timescale: a float
          max_timescale: a float
          num_timescales: an int <= depth/2
        Returns:
          a Tensor the same shape as x.
        """
        signal = self.get_timing_signal(length, min_timescale, max_timescale,
                                        num_timescales)
        padded_signal = tf.pad(signal, [[0, 0], [0, (2 * self.config.dim_sem) - 2 * num_timescales]])
        return x + tf.reshape(padded_signal, [1, length, (2 * self.config.dim_sem)])
