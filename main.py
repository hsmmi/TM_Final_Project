import os
from my_io import my_io
from preproccessing import preproccessing
from STS import STS

dataset_train_file_path = "./Dataset/train.txt"
dataset_test_file_path = "./Dataset/test.txt"
preproccessed_train_data = "./Dataset/preproccessed_train_data.pkl"
preproccessed_test_data = "./Dataset/preproccessed_test_data.pkl"


# Check if preproccessed dataset exists
if not os.path.exists(preproccessed_train_data) or not os.path.exists(
    preproccessed_test_data
):
    # Read the dataset
    data_train = my_io().read_dataset(dataset_train_file_path)
    data_test = my_io().read_dataset(dataset_test_file_path)

    # Preproccess the dataset
    # For sentence 1 and 2
    train_sentence_1 = preproccessing().preproccess(data_train[:, 0])
    train_sentence_2 = preproccessing().preproccess(data_train[:, 1])
    train_score = data_train[:, 2].astype(float)

    test_sentence_1 = preproccessing().preproccess(data_test[:, 0])
    test_sentence_2 = preproccessing().preproccess(data_test[:, 1])
    test_score = data_test[:, 2].astype(float)

    # Sentence to sentence embeddings
    train_sentence_1 = preproccessing().to_sentence_embeddings(
        train_sentence_1
    )
    train_sentence_2 = preproccessing().to_sentence_embeddings(
        train_sentence_2
    )

    test_sentence_1 = preproccessing().to_sentence_embeddings(test_sentence_1)
    test_sentence_2 = preproccessing().to_sentence_embeddings(test_sentence_2)

    # Save the preproccessed dataset
    my_io().save_data(
        [train_sentence_1, train_sentence_2, train_score],
        preproccessed_train_data,
    )
    my_io().save_data(
        [test_sentence_1, test_sentence_2, test_score], preproccessed_test_data
    )
else:
    # Load the preproccessed dataset
    train_sentence_1, train_sentence_2, train_score = my_io().load_data(
        preproccessed_train_data
    )
    test_sentence_1, test_sentence_2, test_score = my_io().load_data(
        preproccessed_test_data
    )

# Train the model
sts_model = STS()
train_acc = sts_model.fit([train_sentence_1, train_sentence_2], train_score)
test_predicted_score = sts_model.predict([test_sentence_1, test_sentence_2])
test_acc = sts_model.evaluate(test_predicted_score, test_score)

print("Have a Graat Time! =D")
