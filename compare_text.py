import pandas as pd

def find_common_texts_multiple(file1, test_files):
    # Load the train dataset
    df1 = pd.read_csv(file1)
    texts1 = set(df1['text'])
    
    # Iterate over each test file
    for test_file in test_files:
        # Load the test dataset
        df2 = pd.read_csv(test_file)
        # Handle different possible column names for text
        if 'Tweet' in df2.columns:
            texts2 = set(df2['Tweet'])
        elif 'Text' in df2.columns:
            texts2 = set(df2['Text'])
        else:
            texts2 = set(df2['text'])
        
        # Find common texts
        common_texts = texts1.intersection(texts2)
        
        # Print common texts for each test file
        print(f"\nCommon texts with {test_file}:")
        if common_texts:
            for text in common_texts:
                print(text)
        else:
            print("No common texts found.")

# Specify the file paths
file1 = 'datasets/train_dataset.csv'
test_files = [
    'test_datasets/C19_test.csv',
    'test_datasets/noun_phrases.csv',
    'test_datasets/raw_test_all_onecol.csv'
]

# Find and print common texts with each test dataset
find_common_texts_multiple(file1, test_files) 