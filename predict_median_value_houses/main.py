from Houses import Boston_Houses


def run_model():
    """
    Define features and run the model
    :return:
    """

    boston_house = Boston_Houses()
    # define the model with paramteters
    boston_house.instantiate_model('reg:linear', 0.3, 0.1, 5, 10, 10)
    # train the  model
    boston_house.train_model()
    # Precit test set
    boston_house.predict()
    # compute the performance
    boston_house.compute_performance()
    # perform cross validation
    boston_house.cross_validation()




if __name__ == '__main__':
    run_model()
