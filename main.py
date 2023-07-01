from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from category_encoders.one_hot import OneHotEncoder
from pickle import dump, load
import pandas as pd

# Function splits a df into features and a target
def split_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    X = df.drop(['satisfaction', 'id', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
    y = df['satisfaction']
    return X, y

# Function reads csv file and turn it into df
def open_data(path="clients.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Function preprocesses df for using it in model
def preprocess_data(df: pd.DataFrame, test=True) -> (pd.DataFrame, pd.Series):

    # Deleting rows with missing values ​​in all features, except Age and Gender
    df.dropna(subset=filter(lambda x: x not in ['Gender', 'Age', 'satisfaction'],
                            df.columns),
              inplace=True)
    
    # Encoding the values ​​of Gender, Customer Type, Type of Travel by 0 and 1
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    df['Customer Type'] = df['Customer Type'].apply(lambda x: 1 if x == 'Loyal Customer' else 0)
    df['Type of Travel'] = df['Type of Travel'].apply(lambda x: 1 if x == 'Business travel' else 0)

    # Replacing Age missing values by the average
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # Removing Age and Flight Distance outliers (with >80 and >4000 values resp.)
    df = df[df['Age'] < 80]
    df = df[df['Flight Distance'] < 4000]

    # Removing outliers from other categorical features and cast them to the appropriate type
    selected_cols = ['Inflight wifi service',
                     'Departure/Arrival time convenient',
                     'Ease of Online booking',
                     'Gate location',
                     'Food and drink',
                     'Online boarding',
                     'Seat comfort',
                     'Inflight entertainment',
                     'On-board service',
                     'Leg room service',
                     'Baggage handling',
                     'Checkin service',
                     'Inflight service',
                     'Cleanliness']
    
    for col in selected_cols:
        df = df[df[col] <= 5]
        df = df[df[col] >= 1]

    df[selected_cols] = df[selected_cols].astype("category")
    #df[list(filter(lambda x: x in selected_cols, df.columns))] = df[list(filter(lambda x: x in selected_cols, df.columns))].astype("category")

    if test:
        # Deleting rows with missing target values and replacing neutral or dissatisfied by 1 and satisfied by 0
        df = df[df['satisfaction'] != '-']
        df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'neutral or dissatisfied' else 0)
        X_df, y_df = split_data(df)
    else:
        X_df = df

    # Using OHE to encode Class column
    one_hot = pd.get_dummies(X_df['Class'])
    X_df = X_df.drop(columns=['Class'], axis=1)
    one_hot = one_hot.drop(one_hot.columns[0], axis=1)
    X_df = X_df.join(one_hot)

    # Scaling features by using MinMaxScaler
    ss = MinMaxScaler()
    ss.fit(X_df)
    X_df = pd.DataFrame(ss.transform(X_df), columns=X_df.columns)

    if test:
        return X_df, y_df
    else:
        return X_df

# Function creates Random Forest Classifier and saves this model in selected path
def fit_and_save_model(X_df: pd.DataFrame, y_df: pd.Series, path="model_weights.mw") -> None:
    model = RandomForestClassifier()
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

# Function loads model and make the prediction using ones
def load_model_and_predict(df: pd.DataFrame, path="model_weights.mw") -> (str, pd.DataFrame):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]

    encode_prediction_proba = {
        0: "Полет понравился с вероятностью",
        1: "Полет не понравился с вероятностью"
    }

    encode_prediction = {
        0: "Полет понравился",
        1: "Полет не понравился"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
