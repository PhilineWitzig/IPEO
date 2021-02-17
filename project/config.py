"""This file contains all configurations for the IPEO project."""

import os


CLIENT_ID = 'accdbcb0-083a-445a-b60b-602f1b47f5ab'
CLIENT_SECRET = 'qUnIL,2k&Tw)/b:*!C@|OVS>hyWxiWTMF|n7heZ4'

DATA_PATH = os.path.join(os.getcwd(), 'data')
TEST_DATA_2017 = os.path.join(DATA_PATH, 'jura', '2017_10_13', 'bands.npy')
TEST_DATA_2020 = os.path.join(
    DATA_PATH, 'koenigsforst', '2020_09_21', 'bands.npy')

TEST_DATA_2018 = os.path.join(
    DATA_PATH, 'koenigsforst', '2018_10_06', 'bands.npy')

RES = 10
COORDS_JURA = [6.11478, 46.490356, 6.40434, 46.700319]  # coordinates of Jura National Park
COORDS_SCHAFFHAUSEN = [8.624954, 47.770253, 8.410034, 47.649200]
COORDS_KOENIGSFORST = [7.16842, 50.94826, 7.10489, 50.91959]

START_DATES = ['2018-10-01', '2020-09-01']
END_DATES = ['2018-10-31', '2020-09-30']
ORBITAL_PERIOD = 5

# Training data for SVM
DATSET_PATH = os.path.join(os.getcwd(), "data", "archive", "EuroSAT")
SVM_PATH = os.path.join(os.getcwd(), "models", "svm_eurosat.pkl")
