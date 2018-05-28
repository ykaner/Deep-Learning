import hw2.cifar10_network
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-load', const=True, default=False, nargs='?',
                    help='if set loads the saved model')
parser.add_argument('-epochs', type=int, default=100, nargs='?',
                    help='the number of training epochs')
parser.add_argument('-gpus', type=int, default=4, nargs='?',
                    help='the number of gpus you want use')

args = parser.parse_args()

if __name__ == "__main__":
	hw2.cifar10_network.main(args)
