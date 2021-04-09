import argparse
import random
parser = argparse.ArgumentParser(description='Shuffle examples')
parser.add_argument('--input', help='input file', required=True)
parser.add_argument('--seed', help='random seed', required=True)
args = parser.parse_args()
f = open(args.input, 'r')
q = f.read().split('\n\n')
random.Random(int(args.seed)).shuffle(q)
print('\n\n'.join(q))
