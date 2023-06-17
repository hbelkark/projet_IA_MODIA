import argparse

import pandas as pd
import torch

from model import NCF

if __name__=='__main__':

    # define the device for computations (either GPU or CPU based on availability)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, help="path to weights file")
    parser.add_argument('--test_script_path', type=str, help="path to test dataset")
    args = parser.parse_args()

    # assign the input arguments to respective variables
    weights_path = args.weights_path
    test_script_path = args.test_script_path

    # load the weights of the trained model
    weights = torch.load(weights_path, map_location=device)

    # extract the number of users, items and factors from the weight dimensions
    n_users = weights["user_embeddings.weight"].shape[0]
    n_items = weights["item_embeddings.weight"].shape[0]
    n_factors = weights["user_embeddings.weight"].shape[1]

    # initialize the NCF model with the extracted numbers of users, items and factors
    model = NCF(n_users=n_users, n_items=n_items, n_factors=n_factors)

    # load the model weights
    model.load_state_dict(weights)

    # load the test dataset
    testset = pd.read_csv(test_script_path)