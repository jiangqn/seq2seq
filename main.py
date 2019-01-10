import argparse
from trainer import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser()
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--vocab_size', type=int, default=37411)
parser.add_argument('--hidden_size', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--num_epoches', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=0)
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--embedding_file_name', type=str, default='data/vocab/glove.npy')
parser.add_argument('--vocab_path', type=str, default='./data/vocab/index2word.pickle')
parser.add_argument('--train_path', type=str, default='./data/processed/train.npz')
parser.add_argument('--dev_path', type=str, default='./data/processed/dev.npz')

config = parser.parse_args()

trainer = Trainer(config)
trainer.run()