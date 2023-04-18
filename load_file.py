import pandas as pd

def loading_data():
    """
    This function would provide us with the loaded data from the excel file

    :return: Loaded Dataframe
    """
    df_true = pd.read_excel("True.xlsx")
    df_fake = pd.read_excel("Fake.xlsx")

    df_true['target'] = 1
    df_fake['target'] = 0

    df = pd.concat([df_true, df_fake]).reset_index(drop = True)

    df['original'] = df['title'] + ' ' + df['text']
    return df







