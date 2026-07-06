# Healthcare Cost Predictive Analytics & AI Pipeline

## 1. Pipeline Architecture
* **Data Ingestion:** Synthetic generation of patient attributes (Age, BMI, Smoking status).
* **Preprocessing:** Feature scaling using Scikit-Learn's `StandardScaler`.
* **Model:** Gradient Boosting Regressor ($R^2 = 0.9875$).
* **API Layer:** FastAPI REST endpoints handling JSON validation via Pydantic.
* **UI Layer:** Streamlit interactive analytics dashboard.

## 2. How to Run Locally
1. Run `pip install -r requirements.txt`
2. Train model: `python train_model.py`
3. Launch API: `uvicorn app_backend:app --reload`
4. Launch UI: `streamlit run app_frontend.py`
