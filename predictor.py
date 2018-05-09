import cPickle
import tensorflow as tf
import logging
import numpy as np
from data_structure import DataSet, ProcessedDoc
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
import utils


class InMemoryClient:
    def __init__(self, model_path, vocab_fname, output_fname, logger, skip_attention):

        self.logger = logger
        self.model_path = model_path
        self.output_fname = output_fname

        if not tf.saved_model.loader.maybe_saved_model_directory(self.model_path):
            raise ValueError('No model found in', self.model_path)

        self.sess = tf.Session(graph=tf.Graph())

        meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.model_path)
        signature_def = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        meta_graph_def_sig = signature_def_utils.get_signature_def_by_key(meta_graph_def, signature_def)

        input_tensor_keys = [k for k in meta_graph_def_sig.inputs]
        input_tensor_names = [meta_graph_def_sig.inputs[k].name for k in input_tensor_keys]

        self.t_variables = {key: name for key, name in zip(input_tensor_keys, input_tensor_names)}

        self.final_output = meta_graph_def_sig.outputs['output'].name
        self.str_scores = meta_graph_def_sig.outputs['str_scores'].name

        self.vocab = utils.load_dict(vocab_fname)

    def predict(self, test_batches, skip_attention, evaluate_split="test"):
        self.logger.info('Sending request to inmemory model')
        self.logger.info('Model path: ' + str(self.model_path))

        processed_docs = []
        corr_count, all_count = 0, 0
        for ct, batch in test_batches:
            feed_dict = self.get_feed_dict(batch)
            outputs, str_scores_batched = self.sess.run([self.final_output, self.str_scores], feed_dict=feed_dict)
            predictions = np.argmax(outputs, 1)
            corr_count += np.sum(predictions == feed_dict[self.t_variables['input_gold_labels']])
            all_count += len(batch)
            # only save the scores if the model was configured with attention
            if not skip_attention:
                batch_processed_docs = self.process_batch(len(batch), feed_dict, str_scores_batched, outputs)
                self.logger.info("Processed %s %s docs in batch %s", len(batch_processed_docs), evaluate_split, ct)
                processed_docs.extend(batch_processed_docs)
        acc_test = 1.0 * corr_count / all_count
        print('{} ACC: {}'.format(evaluate_split, acc_test))
        self.logger.info('{} ACC: {}'.format(evaluate_split, acc_test))
        cPickle.dump(processed_docs, open(self.output_fname, 'w'))
        self.logger.info("Finished processing all batches. Dumped to pickle file %s.", self.output_fname)
        return acc_test

    def process_batch(self, batch_size, feed_dict, str_scores_batched, outputs):
        processed_docs = []
        for batch_num in range(batch_size):
            doc_id = feed_dict[self.t_variables['input_doc_ids']][batch_num]
            gold_label = feed_dict[self.t_variables['input_gold_labels']][batch_num]
            predicted_label = np.argmax(outputs[batch_num])
            token_idxs = feed_dict[self.t_variables['input_token_idxs']][batch_num]
            mask_tokens = feed_dict[self.t_variables['input_mask_tokens']][batch_num]  # doc_l x max_token_l
            mask_sents = feed_dict[self.t_variables['input_mask_sents']][batch_num]
            str_scores_batch = str_scores_batched[batch_num]  # doc_l x doc_l+1
            text = []

            # unmask tokens
            # apply sent mask to remove tokens from missing sents
            mask_tokens = mask_sents[:, np.newaxis] * mask_tokens

            for sent_num in range(token_idxs.shape[0]):
                unmasked_token_idxs = token_idxs[sent_num][mask_tokens[sent_num].astype(bool)]
                if unmasked_token_idxs.size:
                    text.extend([self.vocab[token_idx] for token_idx in unmasked_token_idxs])
                    text.extend(["<split>"])

            # unmask str scores
            # prepend row for ROOT to make it square,
            # and set to neg inf since no node can be the parent of ROOT
            str_scores = np.insert(str_scores_batch, 0, np.inf * -1, axis=0)
            # insert 1 into mask for ROOT node
            mask_sents = np.insert(mask_sents, 0, 1)
            mask_sents_squared = (mask_sents * np.repeat(mask_sents[:, np.newaxis], mask_sents.shape, 1)).astype(bool)
            num_sents_with_root = np.count_nonzero(mask_sents)
            unmasked_str_scores = str_scores[mask_sents_squared].reshape((num_sents_with_root, num_sents_with_root))

            processed_docs.append(ProcessedDoc(doc_id, gold_label, predicted_label, unmasked_str_scores, text))
        return processed_docs

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
        mask_tokens_matrix = np.ones([batch_size, max_doc_l, max_sent_l], np.float32)
        mask_sents_matrix = np.ones([batch_size, max_doc_l], np.float32)
        for i, instance in enumerate(batch):
            n_sents = len(instance.token_idxs)
            gold_matrix[i] = instance.goldLabel
            id_matrix[i] = instance.id
            for j, sent in enumerate(instance.token_idxs):
                token_idxs_matrix[i, j, :len(sent)] = np.asarray(sent)
                mask_tokens_matrix[i, j, len(sent):] = 0
                sent_l_matrix[i, j] = len(sent)
            mask_sents_matrix[i, n_sents:] = 0

        mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_1[:, :, 0] = 0
        mask_parser_2[:, 0, :] = 0

        feed_dict = {self.t_variables['input_token_idxs']: token_idxs_matrix, self.t_variables['input_sent_l']: sent_l_matrix,
                     self.t_variables['input_mask_tokens']: mask_tokens_matrix, self.t_variables['input_mask_sents']: mask_sents_matrix,
                     self.t_variables['input_doc_l']: doc_l_matrix, self.t_variables['input_gold_labels']: gold_matrix,
                     self.t_variables['input_doc_ids']: id_matrix,
                     self.t_variables['input_max_sent_l']: max_sent_l, self.t_variables['input_max_doc_l']: max_doc_l,
                     self.t_variables['input_mask_parser_1']: mask_parser_1, self.t_variables['input_mask_parser_2']: mask_parser_2,
                     self.t_variables['input_batch_l']: batch_size, self.t_variables['input_keep_prob']: 1}
        return feed_dict

    def load_data(self, config, evaluate_split="test"):
        train, dev, test, _, _ = cPickle.load(open(config.data_file))
        eval_set = None
        if evaluate_split == "test":
            eval_set = DataSet(test)
        elif evaluate_split == "train":
            eval_set = DataSet(train)
        elif evaluate_split == "dev":
            eval_set = DataSet(dev)

        eval_batches = eval_set.get_batches(config.batch_size, 1, rand=False)
        eval_batches = [i for i in eval_batches]
        return eval_batches
