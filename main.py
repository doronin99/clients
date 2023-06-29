from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from category_encoders.one_hot import OneHotEncoder
from pickle import dump, load
import pandas as pd


def split_data(df: pd.DataFrame):
    X = df.drop(['satisfaction', 'id', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
    y = df['satisfaction']
    return X, y

def open_data(path="clients.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame, test=True):

    df.dropna(subset=filter(lambda x: x not in ['Gender', 'Age', 'satisfaction'],
                            df.columns),
              inplace=True) # Удаляем строки с пропущенными значениями во всех признаках, кроме Age и Gender

    # Удаляем строки с пропущенными значениями таргета и заменяем значение neutral or dissatisfied на 1, а satisfied - на 0
    df = df[df['satisfaction'] != '-']
    df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'neutral or dissatisfied' else 0)

    # Кодируем значение признаков Gender, Customer Type, Type of Travel нулем и единицей
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    df['Customer Type'] = df['Customer Type'].apply(lambda x: 1 if x == 'Loyal Customer' else 0)
    df['Type of Travel'] = df['Type of Travel'].apply(lambda x: 1 if x == 'Business travel' else 0)

    df['Age'].fillna(df['Age'].mean(), inplace=True) # Заменяем пропущенные значения в столбце Age средним значением

    # Удаляем строки с выбросами для признаков Age и Flight Distance, т.е. со значениями >80 и >4000 соответственно
    df = df[df['Age'] < 80]
    df = df[df['Flight Distance'] < 4000]

    # Удаляем выборсы из категориальных переменных и приводим их к соответствующему типу
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

    df[list(filter(lambda x: x in selected_cols, df.columns))] = df[list(filter(lambda x: x in selected_cols, df.columns))].astype("category")

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    #
    X_df = pd.concat([X_df.drop(['Class'], axis=1),
                   OneHotEncoder().fit_transform(X_df['Class']).drop(['Class_1'], axis=1)],
                  axis=1)
    X_df.head()

    ss = MinMaxScaler()
    ss.fit(X_df)
    X_df = pd.DataFrame(ss.transform(X_df), columns=X_df.columns)

    if test:
        return X_df, y_df
    else:
        return X_df

def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    model = RandomForestClassifier()
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Полет понравился в вероятностью",
        1: "Полет не понравился в вероятностью"
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