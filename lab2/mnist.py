#! python3

import os

from six.moves.urllib.request import urlretrieve

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
# for those who have no access to google storage, use lecun's repo please
WORK_DIRECTORY = "/tmp/mnist-data"


def maybe_download(filename):
	"""A helper to download the data files if not present."""
	if not os.path.exists(WORK_DIRECTORY):
		os.mkdir(WORK_DIRECTORY)
	filepath = os.path.join(WORK_DIRECTORY, filename)
	if not os.path.exists(filepath):
		filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	else:
		print('Already downloaded', filename)
	return filepath


def main():
	train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
	train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
	test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
	test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')


if __name__ == "__main__":
	main()
