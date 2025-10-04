import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

class keblerdata():
    def _init_(self):
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []

    def cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop_duplicates(inplace=True)
        drop_cols = ['rowid','kepid','kepoi_name','koi_pdisposition','kepler_name','koi_score','koi_model_dof','koi_model_chisq','koi_sage','koi_ingress','koi_longp','koi_datalink_dvr','koi_datalink_dvs','koi_comment']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        for col in df.columns:
            if (df[col].isnull().sum()/len(df))<=0.2:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna('unknown')
                else:
                    df[col] = df[col].fillna(df[col].median())
        # Define numeric and categorical columns
        self.numeric_columns = df.select_dtypes(include=['int64','float64']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        # Separate date columns
        self.date_columns = [col for col in self.categorical_columns if 'date' in col.lower()]
        self.categorical_columns = [col for col in self.categorical_columns if col not in self.date_columns]
        le=LabelEncoder()
        for col in self.categorical_columns:
            if df[col].nunique() ==2:
                df[col] = le.fit_transform(df[col])
        self.categorical_columns = [col for col in self.categorical_columns if df[col].nunique() > 2]
        return df

    def feature_engineering(self, user_input) -> pd.DataFrame:
        # تحويل input لـ DataFrame مهما كان نوعه
        if isinstance(user_input, dict):
            df = pd.DataFrame([user_input])
        elif isinstance(user_input, pd.Series):
            df = user_input.to_frame().T
        else:
            df = user_input.copy()
        G = 6.67430e-11
        M_sun = 1.989e30
        day_to_sec = 86400
        R_sun = 6.957e8

        # koi_sma
        if "koi_sma" not in df.columns and "koi_period" in df.columns and "koi_smass" in df.columns:
            P_sec = df["koi_period"] * day_to_sec
            M_kg = df["koi_smass"] * M_sun
            df["koi_sma"] = ((G * M_kg * (P_sec*2)) / (4 * np.pi2))*(1/3) / R_sun

        # koi_teq
        if "koi_teq" not in df.columns and all(col in df.columns for col in ["koi_steff","koi_srad","koi_sma"]):
            df["koi_teq"] = df["koi_steff"] * np.sqrt(df["koi_srad"] / (2 * df["koi_sma"]))

        # koi_insol
        if "koi_insol" not in df.columns and all(col in df.columns for col in ["koi_steff","koi_srad","koi_sma"]):
            df["koi_insol"] = (df["koi_srad"]*2 * df["koi_steff"]*4) / (df["koi_sma"]*2)

        # koi_dor
        if "koi_dor" not in df.columns and all(col in df.columns for col in ["koi_sma","koi_srad"]):
            df["koi_dor"] = df["koi_sma"] / df["koi_srad"]

        # koi_depth
        if "koi_depth" not in df.columns and "koi_ror" in df.columns:
            df["koi_depth"] = df["koi_ror"]**2

        # koi_duration
        if "koi_duration" not in df.columns and all(col in df.columns for col in ["koi_period","koi_srad","koi_sma"]):
            df["koi_duration"] = (df["koi_period"]/np.pi) * (df["koi_srad"]/df["koi_sma"])

        # koi_srho
        if "koi_srho" not in df.columns and all(col in df.columns for col in ["koi_smass","koi_srad"]):
            df["koi_srho"] = df["koi_smass"] / (df["koi_srad"]**3)

        # Color indices
        if "color_g_r" not in df.columns and all(col in df.columns for col in ["koi_gmag","koi_rmag"]):
            df["color_g_r"] = df["koi_gmag"] - df["koi_rmag"]
        if "color_r_i" not in df.columns and all(col in df.columns for col in ["koi_rmag","koi_imag"]):
            df["color_r_i"] = df["koi_rmag"] - df["koi_imag"]

        # Fill any missing columns required by model
        required_cols = ['koi_period','koi_smass','koi_srad','koi_steff','koi_ror','koi_gmag','koi_rmag','koi_imag','koi_sma','koi_teq','koi_insol','koi_dor','koi_depth','koi_duration','koi_prad','koi_num_transits']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[required_cols]  

        df = df.fillna(0)

        return df

    def plot_feature_importance(self, rf_model, numeric_cols, categorical_columns):
        rf = rf_model.named_steps["classifier"]
        feature_names = numeric_cols.copy()
        if categorical_columns:
            ohe = rf_model.named_steps["preprocessor"].named_transformers_["cat"]
            ohe_features = ohe.get_feature_names_out(categorical_columns)
            feature_names.extend(ohe_features)
        importance = rf.feature_importances_
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": (importance / importance.sum()) * 100
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(12, len(feature_importance) * 0.3))
        sns.barplot(data=feature_importance, x="Importance", y="Feature", palette="viridis")
        plt.title("Feature Importances (%) - Random Forest", fontsize=14)
        plt.xlabel("Importance (%)", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        for i, (imp, feat) in enumerate(zip(feature_importance["Importance"], feature_importance["Feature"])):
            plt.text(imp + 0.1, i, f"{imp:.2f}%", va="center")
        plt.tight_layout()
        plt.show()
        return feature_importance

    def train_xgb_model(self, df: pd.DataFrame, target_col: str, **xgb_params):
        import xgboost as xgb
        X = df[['koi_period','koi_smass','koi_srad','koi_steff','koi_ror','koi_gmag','koi_rmag','koi_imag','koi_sma','koi_teq','koi_insol','koi_dor','koi_depth','koi_prad','koi_num_transits']]
        y = df[target_col]
        categorical_columns = [col for col in X.columns if X[col].dtype=='object']
        numeric_cols = [col for col in X.columns if col not in categorical_columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num","passthrough",numeric_cols),
                ("cat",OneHotEncoder(handle_unknown='ignore',drop="first"),categorical_columns)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **xgb_params)
        pipeline = Pipeline(steps=[("preprocessor",preprocessor),("classifier",xgb_clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"✅ Model Trained with params: {xgb_params}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
        return pipeline, acc, report, cm

    def train_rf_model(self, df: pd.DataFrame, target_col: str, **rf_params):
        X = df[['koi_period','koi_smass','koi_srad','koi_steff','koi_ror','koi_gmag','koi_rmag','koi_imag','koi_sma','koi_teq','koi_insol','koi_dor','koi_depth','koi_duration','koi_prad','koi_num_transits']]
        y = df[target_col]
        categorical_columns = [col for col in X.columns if X[col].dtype=='object']
        numeric_cols = [col for col in X.columns if col not in categorical_columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num","passthrough",numeric_cols),
                ("cat",OneHotEncoder(handle_unknown='ignore',drop="first"),categorical_columns)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(random_state=42,class_weight="balanced",**rf_params)
        pipeline = Pipeline(steps=[("preprocessor",preprocessor),("classifier",rf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"✅ Model Trained with params: {rf_params}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)

df=pd.read_csv('app/data/kepler.csv')
kebler = keblerdata()
df = kebler.cleaning(df)
# Features & target
df.drop(df[df['koi_disposition']=='FALSE POSITIVE'].index,inplace=True)
df['koi_disposition']=df['koi_disposition'].map({'CONFIRMED':0,'CANDIDATE':1}).astype(int)
xgb_params = {
    "objective": "multi:softmax",
    "num_class": 3,
    "learning_rate": 0.1,
    "max_depth": 6,
    "alpha": 10,
    "n_estimators": 100
}
pipeline, acc, report, cm = kebler.train_xgb_model(df, target_col="koi_disposition", **xgb_params)

import joblib
joblib.dump(pipeline, 'kepler_xgb_model.pkl')