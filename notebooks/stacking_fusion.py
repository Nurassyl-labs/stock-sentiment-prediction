#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Movement Prediction - Perfected Stacking Fusion (Part E5)
Fuses:
  1. Price Technical Indicators (E1) -> XGBoost
  2. GloVe Twitter Embeddings (E2) -> MLP Classifier
  3. FinBERT Sentiment Scores (E4) -> Random Forest
Meta-Model:
  NonNegativeLogisticRegression using Out-of-Fold (OOF) cross-validation predictions.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)

# --- CUSTOM NON-NEGATIVE LOGISTIC REGRESSION ---
class NonNegativeLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression meta-learner with non-negative constraints on coefficients.
    Prevents the meta-model from overfitting to noisy out-of-fold predictions by 
    arbitrarily inverting the sign of weak classifiers.
    """
    def __init__(self, L2_reg=1e-5):
        self.L2_reg = L2_reg
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            
        def loss_func(params):
            w = params[:-1]
            b = params[-1]
            z = np.dot(X, w) + b
            probs = sigmoid(z)
            # Binary cross entropy with numerical stability epsilon
            loss = -np.mean(y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15))
            # Minimal L2 weight decay regularization
            return loss + self.L2_reg * np.sum(w**2)
            
        # Initial guess: w = [1.0, 1.0, 1.0], b = 0.0
        init_params = np.zeros(n_features + 1)
        init_params[:-1] = 1.0
        
        # Bounds: weights >= 0.0 (non-negative), intercept can be negative
        bounds = [(0.0, None)] * n_features + [(None, None)]
        
        res = minimize(loss_func, init_params, bounds=bounds, method='L-BFGS-B')
        self.coef_ = np.array([res.x[:-1]])
        self.intercept_ = np.array([res.x[-1]])
        return self
        
    def predict_proba(self, X):
        z = np.dot(X, self.coef_[0]) + self.intercept_[0]
        prob_1 = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return np.column_stack([1 - prob_1, prob_1])
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

# --- DETERMINISTIC WORD EMBEDDINGS FALLBACK ---
class DeterministicWordEmbedder:
    """Fallback word embedder in case GloVe weights are missing."""
    def __init__(self, dim=300):
        self.dim = dim
        self.cache = {}
        
    def get_vector(self, word):
        word = word.lower().strip()
        if not word:
            return np.zeros(self.dim)
        if word in self.cache:
            return self.cache[word]
        h = 0
        for char in word:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        rng = np.random.RandomState(h)
        vec = rng.normal(0, 0.1, size=self.dim)
        self.cache[word] = vec
        return vec

    def embed_tweet(self, tokens):
        if not tokens or len(tokens) == 0:
            return np.zeros(self.dim)
        vectors = [self.get_vector(w) for w in tokens if w]
        if len(vectors) == 0:
            return np.zeros(self.dim)
        return np.mean(vectors, axis=0)

# --- LOAD OR COMPUTE GLOVE EMBEDDINGS ---
def load_glove_embeddings(glove_path):
    """Loads GloVe embeddings from file."""
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Successfully loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

def get_daily_glove_embeddings(dataset_path, cache_path, glove_path, stocks=['AAPL', 'AMZN', 'BABA', 'GOOG']):
    """Retrieves daily mean-pooled GloVe embeddings from cache or computes them from scratch."""
    if os.path.exists(cache_path):
        print(f"Loading daily GloVe embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    print("Daily GloVe embeddings cache not found. Computing from raw tweets...")
    
    # Check if GloVe file exists
    has_glove = os.path.exists(glove_path)
    if has_glove:
        glove_index = load_glove_embeddings(glove_path)
        dim = 300
    else:
        print("⚠️ GloVe weights file not found! Falling back to deterministic word vectors for demo.")
        embedder = DeterministicWordEmbedder(dim=300)
        dim = 300
        
    daily_glove = {}
    
    for stock in stocks:
        stock_tweet_dir = os.path.join(dataset_path, 'tweet', 'preprocessed', stock)
        daily_glove[stock] = {}
        
        if not os.path.exists(stock_tweet_dir):
            print(f"⚠️ Tweet directory for {stock} not found at {stock_tweet_dir}")
            continue
            
        print(f"Processing tweets for {stock}...")
        for filename in os.listdir(stock_tweet_dir):
            file_path = os.path.join(stock_tweet_dir, filename)
            if not os.path.isfile(file_path):
                continue
                
            date_str = filename
            try:
                dt = pd.to_datetime(date_str)
                tweet_vectors = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        tweet = json.loads(line)
                        tokens = tweet.get('text', [])
                        
                        if has_glove:
                            # Mean-pool words in tweet using GloVe
                            vecs = [glove_index[w] for w in tokens if w in glove_index]
                            vec = np.mean(vecs, axis=0) if vecs else np.zeros(dim)
                        else:
                            vec = embedder.embed_tweet(tokens)
                            
                        tweet_vectors.append(vec)
                        
                if tweet_vectors:
                    daily_glove[stock][dt] = np.mean(tweet_vectors, axis=0)
            except Exception as e:
                pass
                
    # Cache the computed embeddings
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(daily_glove, f)
    print(f"Saved daily GloVe embeddings to cache: {cache_path}")
    return daily_glove

# --- FEATURE ENGINEERING ---
def compute_technical_indicators(df):
    """Computes E1-consistent 10 technical indicators per stock."""
    df = df.copy()
    close = df['close'].values.astype(float)
    volume = df['volume'].values.astype(float)
    
    # MAs
    df['MA5'] = pd.Series(close).rolling(5, min_periods=5).mean().values
    df['MA20'] = pd.Series(close).rolling(20, min_periods=20).mean().values
    df['MA5_MA20_ratio'] = np.where(np.abs(df['MA20'].values) > 1e-10,
                                   df['MA5'].values / df['MA20'].values, np.nan)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = (100 - 100 / (1 + rs)).values
    
    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    df['MACD'] = (dif - dea).values
    
    # Bollinger Bands
    bb_mid = pd.Series(close).rolling(20, min_periods=20).mean()
    bb_std = pd.Series(close).rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    df['BB_position'] = np.where(np.abs(bb_range.values) > 1e-10,
                                (close - bb_lower.values) / bb_range.values, np.nan)
    
    # Volume and Returns
    vol_prev = pd.Series(volume).shift(1)
    df['vol_change'] = np.where(np.abs(vol_prev.values) > 1e-10,
                               (volume - vol_prev.values) / vol_prev.values, np.nan)
    df['ret_1d'] = pd.Series(close).pct_change(1).values
    df['ret_3d'] = pd.Series(close).pct_change(3).values
    df['ret_5d'] = pd.Series(close).pct_change(5).values
    
    return df

# --- PIPELINE CLASS ---
class PerfectStackingFusionPipeline:
    def __init__(self, dataset_path, cache_path, glove_path, figures_path, stocks=['AAPL', 'AMZN', 'BABA', 'GOOG']):
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.glove_path = glove_path
        self.figures_path = figures_path
        self.stocks = stocks
        
        # Column names
        self.tech_cols = ['MA5', 'MA20', 'MA5_MA20_ratio', 'RSI_14', 'MACD',
                          'BB_position', 'vol_change', 'ret_1d', 'ret_3d', 'ret_5d']
        self.sent_cols = ['positive_mean', 'positive_median', 'positive_min', 'positive_max',
                          'negative_mean', 'negative_median', 'negative_min', 'negative_max',
                          'neutral_mean', 'neutral_median', 'neutral_min', 'neutral_max',
                          'tweet_count', 'rel_volume']
        
        # Base Models (Optimized and regularized to prevent overfitting)
        # XGBoost on technical indicators
        self.price_model = XGBClassifier(
            n_estimators=100, 
            max_depth=2,          # Shallow tree to prevent overfitting
            learning_rate=0.03, 
            reg_alpha=1.0,        # L1 regularization
            reg_lambda=5.0,       # L2 regularization
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
        # MLP on GloVe text embeddings
        self.text_model = MLPClassifier(
            hidden_layer_sizes=(32, 16), # Smaller capacity to reduce overfitting
            activation='relu',
            alpha=3.0,                  # Strong weight decay regularization
            max_iter=1000, 
            early_stopping=True,        # Early stopping validation
            random_state=42
        )
        # Random Forest on FinBERT sentiment statistics
        self.sent_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=3,                # Very shallow trees
            min_samples_split=20, 
            min_samples_leaf=15, 
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1
        )
        
        # Meta-Model (NonNegativeLogisticRegression to prevent sign inversion on noise)
        self.meta_model = NonNegativeLogisticRegression(L2_reg=1e-5)
        
        # Scalers
        self.price_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        self.sent_scaler = StandardScaler()
        
    def prepare_data(self):
        print("\n--- Preparing Stacking Dataset ---")
        # Load daily GloVe embeddings (cached or computed)
        glove_cache_path = os.path.join(self.cache_path, 'daily_glove.pkl')
        daily_glove = get_daily_glove_embeddings(self.dataset_path, glove_cache_path, self.glove_path, self.stocks)
        
        # Load FinBERT sentiment
        finbert_cache_path = os.path.join(self.cache_path, 'daily_sentiment.pkl')
        if not os.path.exists(finbert_cache_path):
            raise FileNotFoundError(f"Missing precomputed FinBERT cache at {finbert_cache_path}")
        sent_df = pd.read_pickle(finbert_cache_path)
        sent_df['date'] = pd.to_datetime(sent_df['date'])
        
        all_stocks_data = []
        
        for stock in self.stocks:
            # Load price data
            price_file = os.path.join(self.dataset_path, 'price', 'preprocessed', f'{stock}.txt')
            cols = ['date', 'movement_pct', 'open', 'high', 'low', 'close', 'volume']
            df_price = pd.read_csv(price_file, sep='\t', header=None, names=cols)
            df_price['date'] = pd.to_datetime(df_price['date'])
            df_price['stock'] = stock
            df_price = df_price.sort_values('date').reset_index(drop=True)
            
            # Compute 10 technical indicators (E1)
            df_features = compute_technical_indicators(df_price)
            df_features[self.tech_cols] = df_features[self.tech_cols].replace([np.inf, -np.inf], np.nan)
            
            # Merge FinBERT daily sentiment statistics (E4)
            stock_sent = sent_df[sent_df['stock'] == stock]
            df_merged = df_features.merge(stock_sent[['date'] + self.sent_cols], on='date', how='left')
            
            # Impute missing sentiment values on trading days with zero tweets
            defaults = {c: 1/3 for c in self.sent_cols if c.startswith(('positive', 'negative', 'neutral'))}
            defaults['tweet_count'] = 0
            defaults['rel_volume'] = 1.0
            df_merged[self.sent_cols] = df_merged[self.sent_cols].fillna(defaults)
            
            # Create GloVe 3-day history sequence (E2)
            glove_stock = daily_glove.get(stock, {})
            
            glove_features = []
            for i, date in enumerate(df_merged['date']):
                # Current day embedding (if missing, fill with zeros)
                emb_t = glove_stock.get(date, np.zeros(300))
                
                # Previous 2 trading days
                emb_t1 = glove_stock.get(df_merged.loc[i-1, 'date'], np.zeros(300)) if i >= 1 else emb_t
                emb_t2 = glove_stock.get(df_merged.loc[i-2, 'date'], np.zeros(300)) if i >= 2 else emb_t1
                
                # Concatenate 3 days -> 900 dimensions
                combined_emb = np.concatenate([emb_t2, emb_t1, emb_t])
                glove_features.append(combined_emb)
                
            glove_matrix = np.array(glove_features)
            glove_cols = [f'glove_{j}' for j in range(900)]
            df_glove = pd.DataFrame(glove_matrix, columns=glove_cols)
            
            df_stock_final = pd.concat([df_merged, df_glove], axis=1)
            
            # Set Label: Next-day close > Today's close (上涨=1, 否则=0)
            df_stock_final['target'] = (df_stock_final['movement_pct'].shift(-1) > 0).astype(int)
            
            # Remove last row because we don't have next-day price movement
            df_stock_final = df_stock_final.iloc[:-1].reset_index(drop=True)
            all_stocks_data.append(df_stock_final)
            
        # Combine all stock datasets
        full_df = pd.concat(all_stocks_data, ignore_index=True)
        
        # Filter for 2014-2016 period (Standard experimental range)
        full_df = full_df[(full_df['date'] >= '2014-01-01') & (full_df['date'] <= '2015-12-31')]
        
        # Drop rows with NaNs (which includes the rolling window warm-up of technical indicators)
        full_df = full_df.dropna(subset=self.tech_cols + ['target']).reset_index(drop=True)
        full_df['target'] = full_df['target'].astype(int)
        
        # Train-Test Split (Chronological splitting at 2015-07-01)
        split_date = pd.to_datetime('2015-07-01')
        train_mask = full_df['date'] < split_date
        test_mask = full_df['date'] >= split_date
        
        self.train_df = full_df[train_mask].reset_index(drop=True)
        self.test_df = full_df[test_mask].reset_index(drop=True)
        
        # Segment features
        glove_cols = [f'glove_{j}' for j in range(900)]
        
        self.X_price_train = self.train_df[self.tech_cols].values
        self.X_text_train = self.train_df[glove_cols].values
        self.X_sent_train = self.train_df[self.sent_cols].values
        self.y_train = self.train_df['target'].values
        
        self.X_price_test = self.test_df[self.tech_cols].values
        self.X_text_test = self.test_df[glove_cols].values
        self.X_sent_test = self.test_df[self.sent_cols].values
        self.y_test = self.test_df['target'].values
        
        print(f"Data prepared successfully!")
        print(f"Training set: {len(self.y_train)} samples (Before 2015-07-01)")
        print(f"Testing set: {len(self.y_test)} samples (On/After 2015-07-01)")
        
        # Standard Scaling on training data, apply to test data
        self.X_price_train_scaled = self.price_scaler.fit_transform(self.X_price_train)
        self.X_price_test_scaled = self.price_scaler.transform(self.X_price_test)
        
        self.X_text_train_scaled = self.text_scaler.fit_transform(self.X_text_train)
        self.X_text_test_scaled = self.text_scaler.transform(self.X_text_test)
        
        self.X_sent_train_scaled = self.sent_scaler.fit_transform(self.X_sent_train)
        self.X_sent_test_scaled = self.sent_scaler.transform(self.X_sent_test)
        
    def train_stacking(self):
        """Performs 5-Fold Cross Validation on train set to create OOF probabilities and trains Meta-Model."""
        print("\n--- Training Stacking Meta-Model via 5-Fold Cross Validation (Out-of-Fold) ---")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Probability array containers
        oof_price = np.zeros(len(self.y_train))
        oof_text = np.zeros(len(self.y_train))
        oof_sent = np.zeros(len(self.y_train))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_price_train_scaled)):
            X_p_tr, X_p_val = self.X_price_train_scaled[train_idx], self.X_price_train_scaled[val_idx]
            X_t_tr, X_t_val = self.X_text_train_scaled[train_idx], self.X_text_train_scaled[val_idx]
            X_s_tr, X_s_val = self.X_sent_train_scaled[train_idx], self.X_sent_train_scaled[val_idx]
            y_tr = self.y_train[train_idx]
            
            # Clone object level regularized models
            fold_price = clone(self.price_model).fit(X_p_tr, y_tr)
            fold_text = clone(self.text_model).fit(X_t_tr, y_tr)
            fold_sent = clone(self.sent_model).fit(X_s_tr, y_tr)
            
            # Predict validation fold OOF probabilities
            oof_price[val_idx] = fold_price.predict_proba(X_p_val)[:, 1]
            oof_text[val_idx] = fold_text.predict_proba(X_t_val)[:, 1]
            oof_sent[val_idx] = fold_sent.predict_proba(X_s_val)[:, 1]
            
            print(f"Fold {fold+1}/5 finished.")
            
        # Combine OOF features -> shape (N_train, 3)
        self.X_meta_train = np.column_stack([oof_price, oof_text, oof_sent])
        
        # Train Meta-Model on OOF probabilities
        print("Training Logistic Regression meta-model...")
        self.meta_model.fit(self.X_meta_train, self.y_train)
        print(f"Meta intercept: {self.meta_model.intercept_[0]:.4f}")
        print(f"Meta coefficients -> Price (XGB): {self.meta_model.coef_[0][0]:.4f}, Text (MLP): {self.meta_model.coef_[0][1]:.4f}, Sentiment (RF): {self.meta_model.coef_[0][2]:.4f}")
        
        # Retrain base models on ENTIRE training set for final test inference
        print("Fitting final base models on full training set...")
        self.price_model.fit(self.X_price_train_scaled, self.y_train)
        self.text_model.fit(self.X_text_train_scaled, self.y_train)
        self.sent_model.fit(self.X_sent_train_scaled, self.y_train)
        
    def evaluate(self):
        """Evaluates price, text, sentiment base models and stacking model on the test set."""
        print("\n--- Evaluating Models on Test Set ---")
        
        # Base Model Predictions
        prob_price = self.price_model.predict_proba(self.X_price_test_scaled)[:, 1]
        pred_price = self.price_model.predict(self.X_price_test_scaled)
        
        prob_text = self.text_model.predict_proba(self.X_text_test_scaled)[:, 1]
        pred_text = self.text_model.predict(self.X_text_test_scaled)
        
        prob_sent = self.sent_model.predict_proba(self.X_sent_test_scaled)[:, 1]
        pred_sent = self.sent_model.predict(self.X_sent_test_scaled)
        
        # Meta-Model Stacking Predictions
        self.X_meta_test = np.column_stack([prob_price, prob_text, prob_sent])
        prob_stack = self.meta_model.predict_proba(self.X_meta_test)[:, 1]
        pred_stack = self.meta_model.predict(self.X_meta_test)
        
        models_dict = {
            'E1: Price Technical (XGB)': (pred_price, prob_price),
            'E2: Text GloVe (MLP)': (pred_text, prob_text),
            'E4: FinBERT Sentiment (RF)': (pred_sent, prob_sent),
            'E5: Stacking Fusion (LR)': (pred_stack, prob_stack)
        }
        
        results = []
        for name, (pred, prob) in models_dict.items():
            acc = accuracy_score(self.y_test, pred)
            prec = precision_score(self.y_test, pred, zero_division=0)
            rec = recall_score(self.y_test, pred, zero_division=0)
            f1 = f1_score(self.y_test, pred, zero_division=0)
            mcc = matthews_corrcoef(self.y_test, pred)
            auc = roc_auc_score(self.y_test, prob)
            
            results.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'MCC': mcc,
                'ROC-AUC': auc
            })
            
        self.df_results = pd.DataFrame(results)
        print("\nTest Set Prediction Results:")
        print(self.df_results.to_string(index=False))
        return results

    def plot_and_save_results(self):
        """Generates premium diagnostic plots and saves them to E5_Stacking_Fusion/figures/."""
        os.makedirs(self.figures_path, exist_ok=True)
        sns.set_theme(style='darkgrid')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.meta_model.predict(self.X_meta_test))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Down (0)', 'Up (1)'],
                    yticklabels=['Down (0)', 'Up (1)'])
        plt.title('E5 Stacking Fusion Confusion Matrix', fontsize=14, pad=15, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 2. Meta-Model Coefficients
        coefs = self.meta_model.coef_[0]
        plt.figure(figsize=(7, 4.5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        sns.barplot(x=['Price Technical (XGB)', 'Text GloVe (MLP)', 'FinBERT Sentiment (RF)'], y=coefs, palette=colors)
        plt.title('Meta-Model Weights (Relative Feature Block Influence)', fontsize=13, pad=15, fontweight='bold')
        plt.ylabel('Logistic Regression Coefficient Value', fontsize=12)
        for i, v in enumerate(coefs):
            plt.text(i, v + (0.02 if v >= 0 else -0.08), f"{v:.4f}", ha='center', fontweight='bold', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'meta_coefficients.png'), dpi=300)
        plt.close()
        
        # 3. ROC Curves
        plt.figure(figsize=(8, 6.5))
        for name in ['E1: Price Technical (XGB)', 'E2: Text GloVe (MLP)', 'E4: FinBERT Sentiment (RF)', 'E5: Stacking Fusion (LR)']:
            prob = self.X_meta_test[:, 0] if 'Price' in name else \
                   self.X_meta_test[:, 1] if 'Text' in name else \
                   self.X_meta_test[:, 2] if 'Sentiment' in name else \
                   self.meta_model.predict_proba(self.X_meta_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, prob)
            auc_score = roc_auc_score(self.y_test, prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linewidth=2)
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Performance Comparison', fontsize=14, pad=15, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'roc_comparison.png'), dpi=300)
        plt.close()
        
        # 4. Model Accuracy Comparison Bar Chart
        plt.figure(figsize=(8.5, 4.5))
        sns.barplot(data=self.df_results, x='Model', y='Accuracy', palette='viridis')
        plt.ylim(0.4, 0.7)
        plt.title('Accuracy Performance Comparison on Test Set', fontsize=14, pad=15, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('', fontsize=12)
        for i, row in self.df_results.iterrows():
            plt.text(i, row['Accuracy'] + 0.005, f"{row['Accuracy']:.2%}", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'accuracy_comparison.png'), dpi=300)
        plt.close()
        print(f"Saved diagnostic plots to: {self.figures_path}")

if __name__ == '__main__':
    # Define paths relative to the execution folder (notebooks)
    dataset_dir = "../data"
    cache_dir = "../cache"
    glove_txt = "../../E1245/glove/glove.6B.300d.txt"
    fig_dir = "./figures"
    
    pipeline = PerfectStackingFusionPipeline(
        dataset_path=dataset_dir,
        cache_path=cache_dir,
        glove_path=glove_txt,
        figures_path=fig_dir
    )
    pipeline.prepare_data()
    pipeline.train_stacking()
    pipeline.evaluate()
    pipeline.plot_and_save_results()
