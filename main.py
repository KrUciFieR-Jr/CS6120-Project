from load_file import loading_data
from preprocessing import data_preprocessing
from modeling import model
from eda import create_visualizations
import nltk
nltk.download('stopwords')

def main():
    """
    This main function controls the flow of the program

    :return: None
    """
    "Load the data from the csv files"
    loaded_dataframe = loading_data()

    "Cleaning Data from tweets"
    cleaned_dataframe = data_preprocessing(loaded_dataframe)

    "Visualizations"
    create_visualizations(cleaned_dataframe)

    "Modeling"
    model(cleaned_dataframe)


main()