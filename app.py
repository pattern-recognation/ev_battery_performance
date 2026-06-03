import streamlit as st
import pandas as pd
import joblib


# =========================
# SAYFA AYARLARI
# =========================

st.set_page_config(
    page_title="EV Battery SOH Prediction",
    page_icon="🔋",
    layout="centered"
)

st.title("🔋 EV Battery SOH Prediction")

st.write("""
Bu arayüz, Linear Regression modelini kullanarak
batarya **SOH (%)** tahmini yapar.
""")


# =========================
# MODELİ YÜKLE
# =========================

try:
    model_package = joblib.load("ev_battery_model.pkl")

    model = model_package["model"]
    scaler = model_package["scaler"]
    feature_names = model_package["feature_names"]
    target = model_package["target"]

    st.success("Linear Regression modeli başarıyla yüklendi.")

    st.caption(f"Hedef değişken: {target}")
    st.caption("Model dosyası: ev_battery_model.pkl")

    st.divider()


    # =========================
    # KULLANICI GİRİŞLERİ
    # =========================

    st.subheader("🧪 Batarya Değerlerini Gir")

    default_values = {
        "Cycle": 750.0,
        "Voltage_V": 4.08,
        "Current_A": 1.50,
        "Temperature_C": 28.7,
        "Time_s": 4219.0
    }

    user_inputs = {}

    for feature in feature_names:
        user_inputs[feature] = st.number_input(
            label=feature,
            value=float(default_values.get(feature, 0.0)),
            step=0.01
        )


    # =========================
    # TAHMİN
    # =========================

    if st.button("SOH Tahmin Et"):
        input_df = pd.DataFrame([user_inputs])

        # Eğitimde kullanılan kolon sırası korunur
        input_df = input_df[feature_names]

        # Notebookta StandardScaler kullanıldığı için aynı scaler uygulanır
        input_scaled = scaler.transform(input_df)

        # Linear Regression tahmini
        prediction = model.predict(input_scaled)[0]

        # SOH değeri 100'ün üstüne çıkarsa 100 olarak göster
        if prediction > 100:
            prediction = 100

        # İstersen negatif değerleri de 0'a sabitleyebiliriz
        if prediction < 0:
            prediction = 0

        st.success(f"Tahmini SOH Değeri: %{prediction:.2f}")

        if prediction >= 90:
            st.info("Batarya sağlığı oldukça iyi görünüyor.")
        elif prediction >= 75:
            st.warning("Batarya sağlığı orta seviyede. Degradasyon başlamış olabilir.")
        else:
            st.error("Batarya sağlığı düşük görünüyor.")



except FileNotFoundError:
    st.error("""
    ev_battery_model.pkl dosyası bulunamadı.

    Lütfen notebookta model kaydetme hücresini çalıştırdıktan sonra oluşan
    ev_battery_model.pkl dosyasını app.py ile aynı klasöre koy.
    """)

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")