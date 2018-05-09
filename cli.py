import os
import argparse
import tensorflow as tf


from main import run as m

# options = dict( rnn_cell='lstm', mode='two-structure', data_file='yelp-2013-largevocab3.pkl',
#                max_sents=100, max_tokens=200, batch_size=16, lstm_hidden_dim_t=75, sent_attention='max',
#                doc_attention='att',
#                mlp_output=False, dropout=0.7, grad_clip='global', opt='Adagrad', lr=0.02, norm=1e-4, short_att=True,
#                lstm_hidden_dim_d=75, sem_dim=75, str_dim=50,
#                comb_atv='tanh')

parser = argparse.ArgumentParser()
parser.add_argument("--rnn_cell", default="lstm")
parser.add_argument("--data_file", default= "/data/yelp-sentiment.pkl")

parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epochs", default=30, type=int)

parser.add_argument("--dim_str", default=50, type=int)
parser.add_argument("--dim_sem", default=75, type=int)
parser.add_argument("--dim_output", default=5, type=int)
parser.add_argument("--keep_prob", default=0.7, type=float)
parser.add_argument("--opt", default='Adagrad')
parser.add_argument("--lr", default=0.05, type=float)
parser.add_argument("--norm", default=1e-4, type=float)
parser.add_argument("--gpu", default="-1")
parser.add_argument("--model_dir_prefix")

parser.add_argument("--sent_attention", default="max")
parser.add_argument("--doc_attention", default="max")
parser.add_argument("--tree_percolation_levels", default=0, type=int)
parser.add_argument("--skip_doc_bilstm", action='store_true')
parser.add_argument("--skip_doc_attention", action='store_true')
parser.add_argument("--large_data", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--log_period", default=5000, type=int)

parser.add_argument("--model_dir")
parser.add_argument("--vocab_file")
parser.add_argument("--evaluate_split", default="test")
parser.add_argument("--data_output_file", default="data/yelp-sentiment-output.pkl")


def main(_):
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    m(config)


if __name__ == "__main__":
    tf.app.run()
