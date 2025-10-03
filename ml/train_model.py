#!/usr/bin/env python3
"""
Cosmic Hunter ML Model Training Script
Trains XGBoost binary classifier for exoplanet detection
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Required features for the model
REQUIRED_FEATURES = [
    'koi_ror',      # Planet-to-star radius ratio
    'koi_impact',   # Impact parameter
    'koi_depth',    # Transit depth
    'koi_prad',     # Planetary radius
    'koi_teq',      # Equilibrium temperature
    'koi_duration', # Transit duration
    'koi_insol',    # Insolation flux
    'koi_steff'     # Stellar effective temperature
]

def load_and_merge_datasets():
    """Load and merge the three Kepler datasets"""
    print("Loading datasets...")
    
    # Load cumulative dataset (main dataset)
    cumulative_df = pd.read_csv('../cumulative_2025.10.01_15.38.26.csv')
    print(f"Cumulative dataset: {cumulative_df.shape}")
    
    # Load K2 dataset
    k2_df = pd.read_csv('../k2pandc_2025.10.01_15.39.58.csv')
    print(f"K2 dataset: {k2_df.shape}")
    
    # Load TOI dataset
    toi_df = pd.read_csv('../TOI_2025.10.01_15.39.37.csv')
    print(f"TOI dataset: {toi_df.shape}")
    
    # For now, focus on cumulative dataset as it has the most complete data
    # and the required features. We can extend this later to merge datasets.
    return cumulative_df

def clean_data(df):
    """Clean and preprocess the data"""
    print("Cleaning data...")
    
    # Create binary target: CONFIRMED = 1, others = 0
    df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
    
    # Filter for required features + target
    feature_cols = REQUIRED_FEATURES + ['target', 'koi_disposition']
    df_clean = df[feature_cols].copy()
    
    # Remove rows with missing values in required features
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=REQUIRED_FEATURES)
    final_rows = len(df_clean)
    
    print(f"Removed {initial_rows - final_rows} rows with missing values")
    print(f"Final dataset shape: {df_clean.shape}")
    
    # Check class distribution
    print("\nClass distribution:")
    print(df_clean['target'].value_counts())
    print(f"Class balance: {df_clean['target'].mean():.3f}")
    
    return df_clean

def train_model(df):
    """Train XGBoost model with optimized parameters"""
    print("\nTraining XGBoost model...")
    
    # Prepare features and target
    X = df[REQUIRED_FEATURES]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost parameters optimized for recall and AUC
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'scale_pos_weight': 2.0,  # Handle class imbalance
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    # Train model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    print("\nModel Evaluation:")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': REQUIRED_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, scaler, feature_importance, X_test_scaled, y_test, y_pred_proba

def save_model_and_artifacts(model, scaler, feature_importance):
    """Save trained model and preprocessing artifacts"""
    print("\nSaving model and artifacts...")
    
    # Save model
    joblib.dump(model, 'exoplanet_model.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    # Save feature names for API
    with open('feature_names.txt', 'w') as f:
        for feature in REQUIRED_FEATURES:
            f.write(f"{feature}\n")
    
    print("Model and artifacts saved successfully!")

def plot_results(feature_importance, X_test, y_test, y_pred_proba):
    """Create visualization plots"""
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cosmic Hunter Model Results', fontsize=16, color='white')
    
    # Feature importance plot
    ax1 = axes[0, 0]
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax1, palette='viridis')
    ax1.set_title('Feature Importance', color='white')
    ax1.set_xlabel('Importance', color='white')
    ax1.set_ylabel('Features', color='white')
    ax1.tick_params(colors='white')
    
    # ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, color='cyan', linewidth=2, label=f'AUC = {auc_score:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate', color='white')
    ax2.set_ylabel('True Positive Rate', color='white')
    ax2.set_title('ROC Curve', color='white')
    ax2.legend()
    ax2.tick_params(colors='white')
    
    # Prediction distribution
    ax3 = axes[1, 0]
    confirmed_probs = y_pred_proba[y_test == 1]
    not_confirmed_probs = y_pred_proba[y_test == 0]
    ax3.hist(not_confirmed_probs, bins=30, alpha=0.7, label='Not Confirmed', color='red')
    ax3.hist(confirmed_probs, bins=30, alpha=0.7, label='Confirmed', color='green')
    ax3.set_xlabel('Predicted Probability', color='white')
    ax3.set_ylabel('Frequency', color='white')
    ax3.set_title('Prediction Distribution', color='white')
    ax3.legend()
    ax3.tick_params(colors='white')
    
    # Confusion Matrix
    ax4 = axes[1, 1]
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
    ax4.set_title('Confusion Matrix', color='white')
    ax4.set_xlabel('Predicted', color='white')
    ax4.set_ylabel('Actual', color='white')
    ax4.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('model_results.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()

def main():
    """Main training pipeline"""
    print("🚀 Cosmic Hunter ML Model Training")
    print("=" * 50)
    
    # Load and merge datasets
    df = load_and_merge_datasets()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Train model
    model, scaler, feature_importance, X_test, y_test, y_pred_proba = train_model(df_clean)
    
    # Save model and artifacts
    save_model_and_artifacts(model, scaler, feature_importance)
    
    # Create visualizations
    plot_results(feature_importance, X_test, y_test, y_pred_proba)
    
    print("\n✅ Training completed successfully!")
    print("Model files created:")
    print("- exoplanet_model.pkl")
    print("- scaler.pkl") 
    print("- feature_importance.csv")
    print("- feature_names.txt")
    print("- model_results.png")

if __name__ == "__main__":
    main()

