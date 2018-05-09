from data_structure import DataSet
from predictor import InMemoryClient
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold='nan')
import cPickle
import logging
from models import StructureModel
import tqdm
import utils


def load_data(config):
    train, dev, test, embeddings, vocab = cPickle.load(open(config.data_file))
    trainset, devset, testset = DataSet(train), DataSet(dev), DataSet(test)
    vocab = dict([(v.index,k) for k,v in vocab.items()])
    trainset.sort()
    train_batches = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = devset.get_batches(config.batch_size, 1, rand=False)
    test_batches = testset.get_batches(config.batch_size, 1, rand=False)
    dev_batches = [i for i in dev_batches]
    test_batches = [i for i in test_batches]
    return len(train), train_batches, dev_batches, test_batches, embeddings, vocab


def evaluate(sess, model, test_batches, logger):
    corr_count, all_count = 0, 0
    num_exceptions = 0
    num_runs = 0
    for ct, batch in test_batches:
        num_runs += 1
        feed_dict = model.get_feed_dict(batch)
        feed_dict[model.t_variables['keep_prob']] = 1.0
        try:
           predictions = sess.run(model.final_output, feed_dict=feed_dict)
        except tf.errors.InvalidArgumentError as err:
           print("Test/Dev InvalidArg error: {0}".format(err))
           logger.debug("\nTest/Dev InvalidArg error: {0}".format(err))
           num_exceptions += 1
           print("Test/Dev Caught invalid arg error exception now ", num_exceptions, " out of ", num_runs, " times: ", num_exceptions/num_runs, ". The count and batch: ", ct)#, batch)
           continue        
        predictions = np.argmax(predictions, 1)
        corr_count += np.sum(predictions == feed_dict[model.t_variables['gold_labels']])
        all_count += len(batch)
    acc_test = 1.0 * corr_count / all_count
    return acc_test


def run(config):
    import random

    hash = random.getrandbits(32)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ah = logging.FileHandler(str(hash)+'.log')
    ah.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ah.setFormatter(formatter)
    logger.addHandler(ah)

    if config.model_dir:
        evaluate_pretrained_model(config, logger)

    else:
        logger.debug("Going to load data")
        num_examples, train_batches, dev_batches, test_batches, embedding_matrix, vocab = load_data(config)
        logger.debug("Finished loading data.")
        # save vocab to file
        utils.save_dict(vocab, str(hash)+'.dict')
        print("Embedding matrix size: ", embedding_matrix.shape)
        config.n_embed, config.d_embed = embedding_matrix.shape

        config.dim_hidden = config.dim_sem + config.dim_str

        print(config)
        logger.critical(str(config))

        model = StructureModel(config)
        model.build()
        model.get_loss()
        # trainer = Trainer(config)

        num_batches_per_epoch = int(num_examples / config.batch_size)
        num_steps = config.epochs * num_batches_per_epoch
        best_acc_dev = 0.0

        with tf.Session() as sess:
            gvi = tf.global_variables_initializer()
            sess.run(gvi)
            sess.run(model.embeddings.assign(embedding_matrix.astype(np.float32)))
            loss = 0
            num_exceptions = 0
            num_runs = 0
            for ct, batch in tqdm.tqdm(train_batches, total=num_steps):
                feed_dict = model.get_feed_dict(batch)
                num_runs += 1
                try:
                    outputs, _, _loss = sess.run([model.final_output, model.opt, model.loss], feed_dict=feed_dict)
                except tf.errors.InvalidArgumentError as err:
                   print("InvalidArg error: {0}".format(err))
                   logger.debug("\nInvalidArg error: {0}".format(err))
                   num_exceptions += 1
                   print("Caught invalid arg error exception now ", num_exceptions, " out of ", num_runs, " times: ", num_exceptions/num_runs, ". The count and batch: ", ct)#, batch)
                   continue
                loss+=_loss
                if(ct%config.log_period==0):
                    acc_test = evaluate(sess, model, test_batches, logger)
                    acc_dev = evaluate(sess, model, dev_batches, logger)
                    print('\nStep: {} Loss: {}'.format(ct, loss))
                    print('Test ACC: {}'.format(acc_test))
                    print('Dev  ACC: %s (%s)', acc_dev, best_acc_dev)
                    logger.debug('\nStep: {} Loss: {}'.format(ct, loss))
                    logger.debug('Test ACC: {}'.format(acc_test))
                    logger.debug('Dev  ACC: %s (%s)', acc_dev, best_acc_dev)
                    logger.handlers[0].flush()
                    loss = 0
                    if acc_dev > best_acc_dev:
                        best_acc_dev = acc_dev
                        save_model(sess, ct, model, logger, config)


def save_model(sess, step, model, logger, config):
    export_path = config.model_dir_prefix + "-" + str(step)
    print('Exporting trained model to %s' % export_path)
    logger.info('Exporting trained model to %s' % export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    input_token_idxs = tf.saved_model.utils.build_tensor_info(model.t_variables['token_idxs'])
    input_sent_l = tf.saved_model.utils.build_tensor_info(model.t_variables['sent_l'])
    input_mask_tokens = tf.saved_model.utils.build_tensor_info(model.t_variables['mask_tokens'])
    input_mask_sents = tf.saved_model.utils.build_tensor_info(model.t_variables['mask_sents'])
    input_doc_l = tf.saved_model.utils.build_tensor_info(model.t_variables['doc_l'])
    input_gold_labels = tf.saved_model.utils.build_tensor_info(model.t_variables['gold_labels'])
    input_doc_ids = tf.saved_model.utils.build_tensor_info(model.t_variables['doc_ids'])
    input_max_sent_l = tf.saved_model.utils.build_tensor_info(model.t_variables['max_sent_l'])
    input_max_doc_l = tf.saved_model.utils.build_tensor_info(model.t_variables['max_doc_l'])
    input_mask_parser_1 = tf.saved_model.utils.build_tensor_info(model.t_variables['mask_parser_1'])
    input_mask_parser_2 = tf.saved_model.utils.build_tensor_info(model.t_variables['mask_parser_2'])
    input_batch_l = tf.saved_model.utils.build_tensor_info(model.t_variables['batch_l'])
    input_keep_prob = tf.saved_model.utils.build_tensor_info(model.t_variables['keep_prob'])

    output = tf.saved_model.utils.build_tensor_info(model.final_output)
    if config.skip_doc_attention:
        str_scores = tf.saved_model.utils.build_tensor_info(tf.convert_to_tensor(np.empty([1,1]), np.float32))
    else:
        str_scores = tf.saved_model.utils.build_tensor_info(model.str_scores)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_token_idxs': input_token_idxs, 'input_sent_l': input_sent_l,
                    'input_mask_tokens': input_mask_tokens, 'input_mask_sents': input_mask_sents,
                    'input_doc_l': input_doc_l, 'input_gold_labels': input_gold_labels,
                    'input_doc_ids': input_doc_ids,
                    'input_max_sent_l': input_max_sent_l, 'input_max_doc_l': input_max_doc_l,
                    'input_mask_parser_1': input_mask_parser_1, 'input_mask_parser_2': input_mask_parser_2,
                    'input_batch_l': input_batch_l, 'input_keep_prob': input_keep_prob},
            outputs={'output': output, 'str_scores': str_scores},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        },
    )
    builder.save()
    print('Done exporting!')
    logger.info('Done exporting!')


def evaluate_pretrained_model(config, logger):
    client = InMemoryClient(config.model_dir, config.vocab_file, config.data_output_file, logger)
    test_batches = client.load_data(config, config.evaluate_split)
    client.predict(test_batches, config.skip_doc_attention, config.evaluate_split)

