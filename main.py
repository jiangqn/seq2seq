import argparse
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=0)
parser.add_argument('--clip', type=float, default=3.0)
parser.add_argument('--dropout', type=float, default=0.0)
# parser.add_argument('--embedding_file_name', default='data/glove.840B.300d.txt')
parser.add_argument('--embedding', default=0)
parser.add_argument('--vocab_size', type=int, default=0)
parser.add_argument('--hidden_size', type=int, default=300)

config = parser.parse_args()

trainer = Trainer(config)
trainer.run()