import torch
from tfrecord.torch.dataset import TFRecordDataset


# load_tfrecord_dataset
# Interface for loading tfrecord datasets. Uses the tfrecord python package (https://github.com/vahidk/tfrecord)
# Parameters:
#   tfrecord_path (String): path to tfrecord file
#   index_path (String): path to .idx file
#                        Can be generated through 'python -m tfrecord.tools.tfrecord2idx <tfrecord_path> <idx_path>'
#   description (dict): dictionary containing the tfrecord column headers and types.
#                       Default for insightface is {"data": "byte", "label": "int"}
#   batch size (int):   Batch size for pytorch DataLoader
# Return values:
#   pytorch DataLoader (torch.utils.DataLoader) with the loaded tfrecord dataset
def load_tfrecord_dataset(tfrecord_path, index_path=None, description={"data": "byte", "label": "int"}, batch_size=32):
    #tfrecord_path = "/idiap/home/lluevano/my_databases/DataZoo_MS1M-ArcFace/tensorflow/faces_emore.tfrecord"
    index_path = None
    dataset = TFRecordDataset(tfrecord_path, index_path, description)
    return dataset
    #return torch.utils.data.DataLoader(dataset, batch_size=32)
