import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib

@st.cache_data
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def preprocess_input(data, scaler):
    df = pd.DataFrame([data])
    features = scaler.feature_names_in_.tolist()
    result = pd.DataFrame(0, index=[0], columns=features)

    # Numeric fields
    for col in ['Age', 'Job', 'Credit amount', 'Duration']:
        if col in df and col in result:
            result[col] = df[col].values[0]

    # One-hot encodings
    one_hot_maps = {
        'Sex': {'male': 'Sex_male'},
        'Housing': {'own': 'Housing_own', 'rent': 'Housing_rent'},
        'Checking account': {
            'moderate': 'Checking account_moderate',
            'rich': 'Checking account_rich'
        },
        'Saving accounts': {
            'moderate': 'Saving accounts_moderate',
            'quite rich': 'Saving accounts_quite rich',
            'rich': 'Saving accounts_rich'
        },
        'Purpose': {
            'car': 'Purpose_car',
            'furniture/equipment': 'Purpose_furniture/equipment',
            'radio/TV': 'Purpose_radio/TV',
            'domestic appliance': 'Purpose_domestic appliances',
            'repairs': 'Purpose_repairs',
            'education': 'Purpose_education',
            'business': 'Purpose_vacation/others',
            'vacation/others': 'Purpose_vacation/others'
        }
    }

    for field, mapping in one_hot_maps.items():
        value = data.get(field)
        col_name = mapping.get(value)
        if col_name in result.columns:
            result[col_name] = 1

    result = result[features]  # Ensure correct order
    scaled = scaler.transform(result)
    return pd.DataFrame(scaled, columns=features)

def user_inputs():
    st.header("Enter Applicant Information")
    return {
        'Age': st.number_input('Age', 18, 100, 30),
        'Sex': st.selectbox('Sex', ['male', 'female']),
        'Job': st.number_input('Job Type (0-3)', 0, 3, 2),
        'Housing': st.selectbox('Housing', ['own', 'free', 'rent']),
        'Saving accounts': st.selectbox('Saving accounts', ['little', 'moderate', 'rich', 'quite rich']),
        'Checking account': st.selectbox('Checking account', ['little', 'moderate', 'rich', None]),
        'Credit amount': st.number_input('Credit amount', 100, 100000, 5000),
        'Duration': st.number_input('Duration (months)', 1, 120, 24),
        'Purpose': st.selectbox('Purpose', [
            'car', 'furniture/equipment', 'radio/TV', 'domestic appliance',
            'repairs', 'education', 'business', 'vacation/others'
        ])
    }

def main():
    st.title("🏦 German Credit Risk Prediction")
    st.markdown("Predict if an applicant is **low-risk** or **high-risk** using a trained XGBoost model.")

    model_path = r'C:\Users\pc\Downloads\analyzeabtestresults-2\AnalyzeABTestResults 2\Credit Risk Prediction Model for Fintech Startup\models\best_xgboost_model.pkl'
    scaler_path = r'C:\Users\pc\Downloads\analyzeabtestresults-2\AnalyzeABTestResults 2\Credit Risk Prediction Model for Fintech Startup\models\scaler.pkl'

    debug = st.sidebar.checkbox("Enable Debug Mode")

    try:
        model = load_pickle(model_path)
        scaler = load_pickle(scaler_path)
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler: {e}")
        return

    data = user_inputs()

    if st.button("Predict Credit Risk"):
        try:
            processed = preprocess_input(data, scaler)
            if processed.empty:
                st.error("Processed data is empty.")
                return

            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0]
            conf = max(prob) * 100

            if pred == 1:
                st.success(f"Low Risk ✅ - Confidence: {conf:.2f}%")
            else:
                st.error(f"High Risk ❌ - Confidence: {conf:.2f}%")

            if debug:
                st.subheader("🔍 Debug Info")
                st.write("Processed Features:", processed.columns.tolist())
                st.write("Expected by model:", getattr(model, 'feature_names_in_', []))

            st.subheader("📊 SHAP Explainability")
            try:
                matplotlib.use('Agg')
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(processed)

                readable = {
                    "Duration": "Loan Duration",
                    "Sex_male": "Sex: Male",
                    "Age": "Applicant Age",
                    "Housing_own": "Housing: Own",
                    "Checking account_moderate": "Checking: Moderate",
                    "Job": "Job",
                    "Purpose_radio/TV": "Purpose: Radio/TV",
                    "Saving accounts_rich": "Savings: Rich",
                    "Saving accounts_moderate": "Savings: Moderate"
                }
                renamed = [readable.get(col, col) for col in processed.columns]
                df_readable = pd.DataFrame(processed.values, columns=renamed)

                st.markdown("**Top Features Influencing Prediction:**")
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, df_readable, plot_type="bar", show=False)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(plt.gcf())

                st.markdown("**Detailed Waterfall Plot:**")
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=df_readable.iloc[0]
                ), show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            except Exception as shap_err:
                st.warning(f"SHAP Error: {shap_err}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            if debug:
                st.subheader("Error Debug Info")
                st.write("Raw Input:", data)
                try:
                    st.write("Processed Columns:", preprocess_input(data, scaler).columns.tolist())
                except Exception as pe:
                    st.write(f"Preprocessing Error: {pe}")

if __name__ == "__main__":
    main()
