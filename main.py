import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle
from model import NCF

class Ratings_Dataset(Dataset):
    def __init__(self, df, user2id, item2id):
        self.df = df.reset_index()
        self.user2id = user2id
        self.item2id = item2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = self.item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating

if __name__ == '__main__':
    # Define the device for computations (either GPU or CPU based on availability)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help="Path to weights file")
    parser.add_argument('--testfile', type=str, help="Path to test dataset")
    args = parser.parse_args()

    # Assign the input arguments to respective variables
    weights_path = args.weights
    testfile_path = args.testfile

    # Load the weights of the trained model
    weights = torch.load(weights_path, map_location=device)

    # Extract the number of users, items, and factors from the weight dimensions
    n_users = weights["user_embeddings.weight"].shape[0]
    n_items = weights["item_embeddings.weight"].shape[0]
    n_factors = weights["user_embeddings.weight"].shape[1]

    # Load the test dataset
    testset = pd.read_csv(testfile_path)

    # Load the mappings from the pickle file
    with open('mappings.pkl', 'rb') as file:
        mappings = pickle.load(file)

    # Get the user and item mappings
    user2id = mappings['user2id']
    item2id = mappings['item2id']

    # Create the trainloader using the Ratings_Dataset
    trainset = pd.read_csv('./data/interactions_train.csv')
    trainloader = DataLoader(Ratings_Dataset(trainset, user2id, item2id), batch_size=512, shuffle=True, num_workers=2)

    # Initialize the NCF model with the extracted numbers of users, items, and factors
    model = NCF(n_users=n_users, n_items=n_items, n_factors=n_factors).to(device)

    # Load the model weights
    model.load_state_dict(weights)

    # Set the model in evaluation mode
    model.eval()

    # Iterate over the test dataset and make predictions
    with torch.no_grad():
        for _, row in testset.iterrows():
            # Check if user and recipe IDs are in the mappings
            user_id = row['user_id']
            recipe_id = row['i']

            # Convert recipe ID to the appropriate data type
            recipe_id = int(recipe_id)

            # Check if the user ID exists in the dictionary
            if recipe_id in item2id:
                # Extract the necessary inputs for prediction
                user = torch.tensor(user2id[user_id], dtype=torch.long).to(device)
                recipe = torch.tensor(item2id[recipe_id], dtype=torch.long).to(device)

                # Perform the prediction
                prediction = model(user, recipe)*5

                # Get the real rating from the test dataset
                real_rating = row['rating']

                # Display the prediction and the real rating in a single line
                print(f"User: {user_id}, Recipe: {recipe_id}, Prediction: {prediction.item()}, Real Rating: {real_rating}")
            else:
                # Recipe ID not found in the dictionary
                print('Recipe ID not found:', recipe_id)

