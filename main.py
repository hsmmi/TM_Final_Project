from my_io import my_io
from preproccessing import preproccessing

# Read the dataset
dataset_file_path = "./Dataset/train.txt"
data = my_io().read_dataset(dataset_file_path)

# Preproccess the dataset

# Do it for sentence 1 and 2
data[:, 0] = preproccessing().preproccess(data[:, 0])
data[:, 1] = preproccessing().preproccess(data[:, 1])


print("Have a Graat Time! =D")
