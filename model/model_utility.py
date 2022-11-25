import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import re

def getInitialFeature(df):
    initial = []
    for i in range(len(df)):
        name = df['Name'].values[i]
        initial.append(name.split(',')[1].split('.')[0].strip())
    #x = np.array(initial)
    #print(np.unique(x))
    df['initial'] = initial
    initial_cats  = [['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'the Countess']]
    initial_ohe = OneHotEncoder(categories = initial_cats)
    initial_feature_arr = initial_ohe.fit_transform(df[['initial']]).toarray()
    #confirm that categories are sorted in the same as pre-defined list
    #print(initial_ohe.categories_)
    initial_feature_labels = initial_cats[0]
    initialDummy = pd.DataFrame(initial_feature_arr, columns=initial_feature_labels)
    return initialDummy


def getAlphabetPrefixTicket(df):
    # alphaPrefixDict = []
    alphaPrefixTicket = []
    for i in range(len(df)):
        ticket = df['Ticket'].values[i]
        res = re.findall('^[\w/.]+\s|$', ticket)[0]
        # res = re.findall('^[\w?.!/;:]+\s|$', ticket)[0]

        # res = re.search('^[\w/.]+\s|$', ticket).group()
        # alphaPrefixDict.append((ticket, res.strip()))

        ticketCode = 'Ticket-' + res.strip().translate({ord(i): None for i in '/.'})
        alphaPrefixTicket.append(ticketCode)
    #     x = np.array(alphaPrefixTicket)
    #     print(np.unique(x))
    df['alphaPrefixTicket'] = alphaPrefixTicket
    alphaPrefix_cats = [['Ticket-', 'Ticket-A4', 'Ticket-A5', 'Ticket-AS', 'Ticket-C', 'Ticket-CA',
                         'Ticket-CASOTON', 'Ticket-FC', 'Ticket-FCC', 'Ticket-Fa', 'Ticket-PC',
                         'Ticket-PP', 'Ticket-PPP', 'Ticket-SC', 'Ticket-SCA4', 'Ticket-SCAH',
                         'Ticket-SCOW', 'Ticket-SCPARIS', 'Ticket-SCParis', 'Ticket-SOC', 'Ticket-SOP',
                         'Ticket-SOPP', 'Ticket-SOTONO2', 'Ticket-SOTONOQ', 'Ticket-SP',
                         'Ticket-STONO', 'Ticket-STONO2', 'Ticket-SWPP', 'Ticket-WC', 'Ticket-WEP']]
    #     alphaPrefixDummy = pd.get_dummies(df['alphaPrefixTicket'], columns = alphaPrefix_cats, drop_first=True)

    alphaPrefix_ohe = OneHotEncoder(categories=alphaPrefix_cats, drop='first')
    alphaPrefix_feature_arr = alphaPrefix_ohe.fit_transform(df[['alphaPrefixTicket']]).toarray()
    # confirm that categories are sorted in the same as pre-defined list
    # print(alphaPrefix_ohe.categories_)
    alphaPrefix_cats[0].remove('Ticket-')
    alphaPrefix_feature_labels = alphaPrefix_cats[0]
    alphaPrefixDummy = pd.DataFrame(alphaPrefix_feature_arr, columns=alphaPrefix_feature_labels)
    return alphaPrefixDummy

def getOneHotEncodeSex(df):
    #x = np.array(df['Sex'])
    #print(np.unique(x))
    sex_cats  = [['male', 'female']]
    sex_ohe = OneHotEncoder(categories = sex_cats, drop = 'first')
    sex_feature_arr = sex_ohe.fit_transform(df[['Sex']]).toarray()
    #confirm that categories are sorted in the same as pre-defined list
    #print(sex_ohe.categories_)
    sex_cats[0].remove('male')
    sex_feature_labels = sex_cats[0]
    sexDummy = pd.DataFrame(sex_feature_arr, columns=sex_feature_labels)
    return  sexDummy


def getOneHotEncodeEmbarked(df):
    #x = np.array(df['Embarked'])
    #print(np.unique(x))
    embarked_cats  = [['C', 'Q', 'S']]
    embarked_ohe = OneHotEncoder(categories = embarked_cats, drop = 'first')
    embarked_feature_arr = embarked_ohe.fit_transform(df[['Embarked']]).toarray()
    #confirm that categories are sorted in the same as pre-defined list
    #print(embarked_ohe.categories_)
    embarked_cats[0].remove('C')
    embarked_feature_labels = embarked_cats[0]
    embarkedDummy = pd.DataFrame(embarked_feature_arr, columns=embarked_feature_labels)
    return  embarkedDummy

def extract_feat(in_df):
    feat = in_df.copy()
    initial = getInitialFeature(feat)
    sex = getOneHotEncodeSex(feat)
    alphaPrefix = getAlphabetPrefixTicket(feat)
    embarked = getOneHotEncodeEmbarked(feat)

    #print(initial.shape)
    #print(alphaPrefix.shape)
    feat.reset_index(drop=True, inplace=True)
    feat = pd.concat([feat, initial, sex, alphaPrefix, embarked],axis=1)
    #feat = pd.concat([feat, initial],axis=1)
    feat = feat.drop('PassengerId', axis=1) # infinite features
    feat = feat.drop('Name', axis=1)            # infinite features
    feat = feat.drop('Sex', axis=1)               # have alterative features, one-hot female
    feat = feat.drop('Ticket', axis=1)            # have alterative features, AlphaPrefixTicket
    feat = feat.drop('Cabin', axis=1)            # have alterative features, one-hot CabinZone
    feat = feat.drop('Embarked', axis=1)     # have alterative features, one-hot Embarked
    feat = feat.drop('initial', axis=1)             # have alterative features, one-hot initial
    feat = feat.drop('alphaPrefixTicket', axis=1) # have alterative features, one-hot alpha Prefix Ticket
    #feat = feat._get_numeric_data() #magic method to filter in only numeric features
    return feat

def encode_data(df, DEBUG = False):
    gle = LabelEncoder()

    for col in df.columns:
        # print(col)
        if df[col].dtype == "object": #encode all columns that are categorical data
            if DEBUG:
                print("Encoding columns: " + col)
            labels = gle.fit_transform(df[col])
            mappings = {index: label for index, label in enumerate(gle.classes_)}
            #print(mappings)
            df[col] = labels
    return df

def load_deployable_model(file):
    print("load a pre-trained model from...")
    print(file)
    with open(file,'rb') as f:
        model = pickle.load(f)
    return model


def clean_query(in_df):
    df = in_df.copy()
    df['Cabin'] = df['Cabin'].fillna('nocabin')
    df['CabinZone'] = 'Cabin-' + df['Cabin'].str[0]
    cabin_zone_cats = [['Cabin-n', 'Cabin-A', 'Cabin-B', 'Cabin-C', 'Cabin-D', 'Cabin-E', 'Cabin-F', 'Cabin-G', 'Cabin-T']]
    cabin_ohe = OneHotEncoder(categories = cabin_zone_cats, drop = 'first')
    cabin_feature_arr = cabin_ohe.fit_transform(df[['CabinZone']]).toarray()
    # prepare the heading of the columns, cabin_zone one-hotted
    cabin_zone_cats[0].remove('Cabin-n')
    cabin_feature_labels = cabin_zone_cats[0]
    cabin_features = pd.DataFrame(cabin_feature_arr, columns=cabin_feature_labels)
    df = pd.concat([df, cabin_features], axis=1)

    df.drop('CabinZone', axis=1, inplace = True)

    # *** query should fill in all attributes, so no impute some missing values like Age and Embarked ***
    #     df['Age'] = df['Age'].fillna(df['Age'].mean())
    #     df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def preprocess_query(df):
    # 1. Clean data (or step 2 in training a model)
    df = clean_query(df)

    # 2. Extract feature and split data  (or step 3 in training a model)
    # ----------------------------------------------
    # 2.1 extract features
    df = extract_feat(df)
    # 2.2 encode all categorical data
    df = encode_data(df, DEBUG=True)
    return df