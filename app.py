import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="EV Battery SOH Prediction",
    page_icon="🔋",
    layout="centered"
)

st.title("🔋 EV Battery SOH Prediction")
st.write("""
Bu arayüz, batarya verilerine göre **SOH (%)** tahmini yapar.

Bu sürümde **scikit-learn kullanılmaz**, bu yüzden DLL hatasına takılmaz.
Model mantığı: Linear Regression
""")


@st.cache_data
def load_data():
    df = pd.read_csv("EV_Battery_Dataset_1.csv")
    return df


def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Sıfıra bölme hatasını engellemek için
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


def train_linear_regression(X_train, y_train):
    # Bias kolonu eklenir
    ones = np.ones((X_train.shape[0], 1))
    X_bias = np.hstack([ones, X_train])

    # Linear Regression normal equation
    coefficients = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y_train

    return coefficients


def predict_linear_regression(X, coefficients):
    ones = np.ones((X.shape[0], 1))
    X_bias = np.hstack([ones, X])
    predictions = X_bias @ coefficients
    return predictions


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_total)

    return mae, rmse, r2


@st.cache_resource
def train_model(df):
    target = "SOH_pct"

    # Notebooktaki gibi Capacity_Ah modele dahil edilmiyor
    remove_cols = [target, "Capacity_Ah"]

    X_df = df.drop(remove_cols, axis=1)
    y_series = df[target]

    feature_names = X_df.columns.tolist()

    X = X_df.values.astype(float)
    y = y_series.values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train_scaled, X_test_scaled, mean, std = standardize_train_test(X_train, X_test)

    coefficients = train_linear_regression(X_train_scaled, y_train)

    y_pred = predict_linear_regression(X_test_scaled, coefficients)

    mae, rmse, r2 = calculate_metrics(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return coefficients, mean, std, feature_names, metrics


try:
    df = load_data()
    coefficients, mean, std, feature_names, metrics = train_model(df)

    st.subheader("📊 Model Performansı")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{metrics['MAE']:.4f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
    col3.metric("R²", f"{metrics['R2']:.4f}")

    st.divider()

    st.subheader("🧪 Batarya Değerlerini Gir")

    user_inputs = {}

    for feature in feature_names:
        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        mean_value = float(df[feature].mean())

        user_inputs[feature] = st.number_input(
            label=feature,
            min_value=min_value,
            max_value=max_value,
            value=mean_value,
            step=0.01
        )

    if st.button("SOH Tahmin Et"):
        input_array = np.array([[user_inputs[feature] for feature in feature_names]], dtype=float)

        input_scaled = (input_array - mean) / std

        prediction = predict_linear_regression(input_scaled, coefficients)[0]

        st.success(f"Tahmini SOH Değeri: %{prediction:.2f}")

        if prediction >= 90:
            st.info("Batarya sağlığı oldukça iyi görünüyor.")
        elif prediction >= 75:
            st.warning("Batarya sağlığı orta seviyede. Degradasyon başlamış olabilir.")
        else:
            st.error("Batarya sağlığı düşük görünüyor. Detaylı kontrol önerilir.")

    

except FileNotFoundError:
    st.error("""
    EV_Battery_Dataset_1.csv dosyası bulunamadı.

    Lütfen bu dosyayı app.py ile aynı klasöre koy:
    - app.py
    - EV_Battery_Dataset_1.csv
    """)

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
