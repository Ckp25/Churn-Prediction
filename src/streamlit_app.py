import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/logistic_model_fbeta.joblib")
scaler = joblib.load("models/preprocessor.joblib")

# Load preprocessed data with IDs
df_full = pd.read_csv("data/preprocessed_churn_with_id.csv")

df_lookup = df_full.set_index("customerID")

st.title(" Telco Customer Churn Predictor")
st.write("Enter a customer ID to predict their churn risk.")

customer_id = st.text_input("Customer ID")

if customer_id:
    if customer_id in df_lookup.index:
        row = df_lookup.loc[customer_id]
        X = row.drop("Churn").values.reshape(1, -1)
        y_true = row["Churn"]

        # Predict churn probability
        churn_prob = model.predict_proba(X)[0][1]
        churn_class = model.predict(X)[0]

        # Determine risk level and color
        if churn_prob >= 0.75:
            risk_label = "High Risk"
            risk_color = "#ff4d4d"  # red
        elif churn_prob >= 0.4:
            risk_label = "Medium Risk"
            risk_color = "#ffa500"  # orange
        else:
            risk_label = "Low Risk"
            risk_color = "#90ee90"  # green

        st.markdown(f"###  Prediction for Customer `{customer_id}`")
        st.markdown(
            f"<div style='padding: 10px; background-color: {risk_color}; border-radius: 5px;'>"
            f"<strong>Risk Level:</strong> {risk_label}<br>"
            f"<strong>Predicted Churn Probability:</strong> {churn_prob:.2%}"
            f"</div>", unsafe_allow_html=True
        )

        st.write(f"**Predicted Class:** {'Yes' if churn_class == 1 else 'No'}")
        st.write(f"**True Label:** {y_true}")

        # Show feature values (top 10 by magnitude)
        coef = model.coef_[0]
        features = row.drop("Churn").index
        contributions = pd.DataFrame({
            "Feature": features,
            "Value": X.flatten(),
            "Coefficient": coef,
            "Impact": X.flatten() * coef
        })
        contributions["AbsImpact"] = contributions["Impact"].abs()
        top_contrib = contributions.sort_values("AbsImpact", ascending=False).head(10)

        st.markdown("### Top Feature Contributions")
        st.dataframe(top_contrib[["Feature", "Value", "Coefficient", "Impact"]].round(3))

    else:
        st.error("Customer ID not found. Please check and try again.")
