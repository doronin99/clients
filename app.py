import pandas as pd
import streamlit as st
from main import open_data, preprocess_data, fit_and_save_model, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()

def show_main_page():

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Clients",
    )

    st.write(
        """
        # Задача прогноза удовлетворенности клиента полетом
        Определяем, кто из пассажиров самолета удовлетворен полетом, а кто – нет.
        """
    )

def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, train_y_df = preprocess_data(train_df)
    fit_and_save_model(train_X_df, train_y_df, path="model_weights.mw")

    user_X_df = preprocess_data(user_input_df, test=False)
    write_user_data(user_X_df)

    #prediction, prediction_probas = load_model_and_predict(user_X_df)
    #write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    gender = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20,
                            step=1)
    customer_type = st.sidebar.selectbox("Лоялен ли клиент авиакомпании?", ("Лоялен", "Не лоялен"))
    type_of_travel = st.sidebar.selectbox("Тип поездки", ("Деловая", "Личная"))
    clas = st.sidebar.selectbox("Класс", ("Бизнес", "Эко", "Эко+"))
    flight_distance = st.sidebar.slider("Дальность перелета (в милях)", min_value=50, max_value=4000, value=500,
                            step=50)

    inflight_wifi_service = st.sidebar.slider("Качество работы Wi-Fi в полете", min_value=1, max_value=5, value=5, step=1)
    departure_arrival_time_convenient = st.sidebar.slider("Удобство времени отрпавления и прибытия", min_value=1, max_value=5, value=5, step=1)
    ease_of_online_booking = st.sidebar.slider("Удобство онлайн бронирования", min_value=1, max_value=5, value=5, step=1)
    gate_location = st.sidebar.slider("Удобство расположения выхода на посадку", min_value=1, max_value=5, value=5, step=1)
    food_and_drink = st.sidebar.slider("Качество еды и напитков", min_value=1, max_value=5, value=5, step=1)
    online_boarding = st.sidebar.slider("Удобство онлайн регистрации", min_value=1, max_value=5, value=5, step=1)
    seat_comfort = st.sidebar.slider("Удобство сидений", min_value=1, max_value=5, value=5, step=1)
    inflight_entertainment = st.sidebar.slider("Развлечения", min_value=1, max_value=5, value=5, step=1)
    onboard_service = st.sidebar.slider("Услуги на борту", min_value=1, max_value=5, value=5, step=1)
    leg_room_service = st.sidebar.slider("Удобство дистанции между креслами", min_value=1, max_value=5, value=5, step=1)
    baggage_handling = st.sidebar.slider("Работа с багажом", min_value=1, max_value=5, value=5, step=1)
    checkin_service = st.sidebar.slider("Услуги регистрации", min_value=1, max_value=5, value=5, step=1)
    inflight_service = st.sidebar.slider("Сервис в полете", min_value=1, max_value=5, value=5, step=1)
    cleanliness = st.sidebar.slider("Чистота", min_value=1, max_value=5, value=5, step=1)

    translatetion = {
        "Мужской": "Male",
        "Женский": "Female",
        "Лоялен": "Loyal Customer",
        "Не лоялен": "disloyal Customer",
        "Деловая": "Business travel",
        "Личная": "Personal Travel",
        "Бизнес": "Business",
        "Эко": "Eco",
        "Эко+": "Eco Plus",
    }

    data = {
        "Gender": translatetion[gender],
        "Age": age,
        "Customer Type": translatetion[customer_type],
        "Type of Travel": translatetion[type_of_travel],
        "Class": translatetion[clas],
        "Flight Distance": flight_distance,

        'Inflight wifi service': inflight_wifi_service,
        'Departure/Arrival time convenient': departure_arrival_time_convenient,
        'Ease of Online booking': ease_of_online_booking,
        'Gate location': gate_location,
        'Food and drink': food_and_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': inflight_entertainment,
        'On-board service': onboard_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()