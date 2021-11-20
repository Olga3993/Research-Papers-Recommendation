from .dataset import Dataset
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


class Ann(Dataset):
    def __init__(self,
                 train_graph_path=CURRENT_DIR.joinpath('../data/ann/train_edges.txt'),
                 test_graph_path=CURRENT_DIR.joinpath('../data/ann/test_edges.txt'),
                 graph_path=CURRENT_DIR.joinpath('../data/ann/edges.txt'),
                 texts_path=CURRENT_DIR.joinpath('../data/ann/docs.txt'),
                 labels_path=CURRENT_DIR.joinpath('../data/ann/labels.txt')):
        super(Ann, self).__init__(train_graph_path,test_graph_path,graph_path, texts_path, labels_path)
