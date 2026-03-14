import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="Détection d'attaque DDoS", page_icon="🛡️", layout="wide")

DATA_PATH = Path("ddos_dataset_demo.csv")
MODEL_PATH = Path("rf_ddos_model.joblib")
PREPROCESSOR_PATH = Path("ddos_preprocessor.joblib")
META_PATH = Path("ddos_meta.joblib")

# Démo simple : à remplacer par une base de données ou un fichier sécurisé
USERS = {
    "admin": "admin123",
    "analyste": "soc2026",
}


# =========================
# AUTHENTIFICATION
# =========================
def init_session():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None


def login_page():
    st.title("🔐 Login")
    st.markdown("Connectez-vous pour accéder au dashboard de détection d'attaque.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Se connecter")

        if submitted:
            if username in USERS and USERS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Authentification réussie.")
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")


def logout_button():
    if st.sidebar.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()


# =========================
# DATA + MODELE
# =========================
@st.cache_data
def load_dataset(data_path: str):
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Le fichier {data_path} est introuvable. Placez votre dataset à côté de l'application."
        )
    return pd.read_csv(data_path)


@st.cache_resource
def train_and_prepare(df: pd.DataFrame):
    data = df.copy()

    if "label" not in data.columns:
        raise ValueError("Le dataset doit contenir une colonne cible nommée 'label'.")

    # Nettoyage
    data = data.drop_duplicates().copy()

    # Suppression colonne timestamp si inutilisable pour le modèle
    if "timestamp" in data.columns:
        data = data.drop(columns=["timestamp"])

    X = data.drop(columns=["label"])
    y = data["label"].copy()

    # Encodage cible si nécessaire
    label_encoder = None
    if y.dtype == object:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_columns = [col for col in X.columns if col not in numeric_columns]

    if non_numeric_columns:
        raise ValueError(
            "Toutes les variables d'entrée doivent être numériques pour cette application. "
            f"Colonnes non numériques détectées : {non_numeric_columns}"
        )

    # Imputation
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Équilibrage
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resampled, y_resampled, test_size=0.30, random_state=42, stratify=y_resampled
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Modèle
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    meta = {
        "feature_names": X.columns.tolist(),
        "label_encoder": label_encoder,
        "class_names": infer_class_names(label_encoder),
    }

    # Sauvegarde facultative des objets
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"imputer": imputer, "scaler": scaler}, PREPROCESSOR_PATH)
    joblib.dump(meta, META_PATH)

    return {
        "clean_df": data,
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "metrics": metrics,
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
        "feature_importances": feature_importances,
        "feature_names": X.columns.tolist(),
        "meta": meta,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
    }


def infer_class_names(label_encoder):
    if label_encoder is not None:
        classes = list(label_encoder.classes_)
        if len(classes) == 2:
            return classes
        return [str(c) for c in classes]
    return ["Normal", "Attaque"]


# =========================
# PREDICTION
# =========================
def predict_single_input(model_objects, input_df: pd.DataFrame):
    imputer = model_objects["imputer"]
    scaler = model_objects["scaler"]
    model = model_objects["model"]
    meta = model_objects["meta"]

    X_imputed = imputer.transform(input_df)
    X_scaled = scaler.transform(X_imputed)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    label_encoder = meta["label_encoder"]
    if label_encoder is not None:
        predicted_label = label_encoder.inverse_transform([pred])[0]
        positive_class_name = label_encoder.classes_[1] if len(label_encoder.classes_) > 1 else str(predicted_label)
    else:
        predicted_label = "Attaque" if pred == 1 else "Normal"
        positive_class_name = "Attaque"

    attack_probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

    return predicted_label, attack_probability, positive_class_name


# =========================
# UI
# =========================
def dashboard_page(df: pd.DataFrame, model_objects: dict):
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller vers", ["Dashboard", "Test de prédiction"])
    st.sidebar.success(f"Connecté : {st.session_state.username}")
    logout_button()

    if page == "Dashboard":
        render_dashboard(df, model_objects)
    else:
        render_prediction_form(model_objects)


def render_dashboard(df: pd.DataFrame, model_objects: dict):
    st.title("🛡️ Dashboard - Détection d'attaque")

    clean_df = model_objects["clean_df"]
    metrics = model_objects["metrics"]

    st.subheader("Statistiques du dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre de lignes", f"{len(clean_df):,}")
    c2.metric("Nombre de colonnes", clean_df.shape[1])
    c3.metric("Training set", model_objects["train_shape"][0])
    c4.metric("Test set", model_objects["test_shape"][0])

    with st.expander("Afficher un aperçu du dataset"):
        st.dataframe(clean_df.head(20), use_container_width=True)

    st.subheader("Distribution attaque / normal")
    if "label" in clean_df.columns:
        distribution = clean_df["label"].value_counts().reset_index()
        distribution.columns = ["Classe", "Nombre"]
        st.dataframe(distribution, use_container_width=True)

        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        ax_dist.bar(distribution["Classe"].astype(str), distribution["Nombre"])
        ax_dist.set_title("Distribution du trafic réseau")
        ax_dist.set_xlabel("Classe")
        ax_dist.set_ylabel("Nombre")
        st.pyplot(fig_dist)

    st.subheader("Performances du modèle")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m2.metric("Precision", f"{metrics['precision']:.4f}")
    m3.metric("Recall", f"{metrics['recall']:.4f}")
    m4.metric("F1-score", f"{metrics['f1']:.4f}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Courbe ROC**")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(model_objects["fpr"], model_objects["tpr"])
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        st.pyplot(fig_roc)

    with col2:
        st.markdown("**Importance des variables**")
        top_features = model_objects["feature_importances"].head(10).sort_values()
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        ax_imp.barh(top_features.index.astype(str), top_features.values)
        ax_imp.set_title("Top 10 features")
        ax_imp.set_xlabel("Importance")
        st.pyplot(fig_imp)


def render_prediction_form(model_objects: dict):
    st.title("🧪 Test de prédiction")
    st.write("Saisissez les paramètres réseau pour prédire si le trafic est **Normal** ou **Attaque**.")

    feature_names = model_objects["feature_names"]

    with st.form("prediction_form"):
        user_input = {}
        cols = st.columns(2)

        for idx, feature in enumerate(feature_names):
            with cols[idx % 2]:
                user_input[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    step=0.1,
                    format="%.4f"
                )

        submitted = st.form_submit_button("Prédire")

    if submitted:
        input_df = pd.DataFrame([user_input], columns=feature_names)
        predicted_label, attack_probability, positive_class_name = predict_single_input(model_objects, input_df)

        st.subheader("Résultat")
        if str(predicted_label).lower() in ["1", "attack", "attaque", "ddos", positive_class_name.lower()]:
            st.error(f"Prédiction : {predicted_label}")
        else:
            st.success(f"Prédiction : {predicted_label}")

        st.write(f"Probabilité de la classe positive ({positive_class_name}) : **{attack_probability:.2%}**")
        st.dataframe(input_df, use_container_width=True)


# =========================
# MAIN
# =========================
def main():
    init_session()

    if not st.session_state.authenticated:
        login_page()
        return

    try:
        df = load_dataset(str(DATA_PATH))
        model_objects = train_and_prepare(df)
        dashboard_page(df, model_objects)
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.info(
            "Vérifiez que le fichier 'ddos_dataset_demo.csv' existe dans le même dossier que l'application, "
            "et qu'il contient une colonne 'label'."
        )


if __name__ == "__main__":
    main()
