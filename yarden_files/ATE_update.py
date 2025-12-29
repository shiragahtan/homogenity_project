import json
import numpy as np
import pandas as pd
from scipy import stats
from linear_model_unlearning import CertifiableUnlearningLogisticRegression, BaseLinearRegression
from sklearn.linear_model import LogisticRegression
from numpy.linalg import LinAlgError

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

TREATMENT_COL = config['TREATMENT_COL']


def calculate_ate_safe(df, treatment_col, outcome_col, delta=None, ret_obj=False):
    """
    Safely calculates ATE.

    Args:
        df: DataFrame containing the data
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        delta: (Optional) Threshold for sample size calculation.
               If None or 0, strict size checks are relaxed.
        ret_obj: Whether to return the ATE object instead of just the value.
    """
    try:
        # 1. SAMPLE SIZE FILTER
        counts = df[treatment_col].value_counts()

        # LOGIC CHANGE:
        # If delta is provided (and > 0), use the strict project logic.
        # If delta is None or 0, use a minimal safety floor (e.g. 5 samples) to allow execution.
        if delta and delta > 0:
            min_samples_per_group = max(30, delta / 20.0)
        else:
            min_samples_per_group = 5  # Minimal constant to ensure regression doesn't crash

        if len(counts) < 2 or counts.min() < min_samples_per_group:
            return np.nan

        # Prepare covariates
        exclude_cols = [treatment_col, TREATMENT_COL, outcome_col]
        features_cols = [c for c in df.columns if c not in exclude_cols]
        features_cols = [c for c in features_cols if df[c].nunique() > 1]

        if not features_cols:
            return np.nan

        n_params = 2 + len(features_cols)
        if len(df) <= n_params + 5:
            return np.nan

        try:
            ate_obj = ATEUpdateLinear(
                df[features_cols],
                df[treatment_col],
                df[outcome_col]
            )

            cate_value = ate_obj.get_original_ate()

            if not np.isfinite(cate_value):
                return np.nan

            # If ATE is > $10,000,000, it is a math error, not a real salary difference.
            if abs(cate_value) > 10_000_000:
                return np.nan
            # -----------------------------------------------------

            # 3. STANDARD ERROR FILTER
            y_pred = ate_obj.X_matrix @ ate_obj.original_model.beta
            residuals = ate_obj.Y_matrix - y_pred
            rss = np.sum(residuals ** 2)
            df_resid = ate_obj.n_samples - ate_obj.n_features
            mse = rss / df_resid

            xtx_inv = ate_obj.original_model.XTX_inv
            var_beta_treatment = mse * xtx_inv[1, 1]

            # If variance is negative, the matrix inversion failed numerically.
            if var_beta_treatment <= 0:
                return np.nan

            se_beta_treatment = np.sqrt(var_beta_treatment)
            # -------------------------------------------------------------

            # THRESHOLD: Filter if SE is > 50% of the mean outcome
            outcome_mean = abs(df[outcome_col].mean())
            if outcome_mean > 0 and se_beta_treatment > (outcome_mean * 0.5):
                return np.nan

            return cate_value if not ret_obj else ate_obj

        except LinAlgError:
            return np.nan if not ret_obj else None

    except Exception:
        return np.nan


class ATEUpdateLinear:
    def __init__(self, X, T, Y, find_confounders=False):
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T, index=self.X.index)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y, index=self.X.index)
        intercept = pd.Series(1, index=self.X.index, name='intercept')
        
        self.design_matrix = pd.concat([intercept, self.T, self.X], axis=1)
        self.design_matrix.columns = ['intercept', 'treatment'] + self.X.columns.tolist()
        self.X_matrix = self.design_matrix.values
        self.Y_matrix = self.Y.values.reshape(-1, 1)
        self.n_samples = self.X_matrix.shape[0]
        self.n_features = self.X_matrix.shape[1]
        
        self.original_model = BaseLinearRegression(self.X_matrix, self.Y_matrix)
        self.original_ate = float(self.original_model.beta[1].item())

    def get_original_ate(self):
        return self.original_ate

    def calculate_p_value(self):
        """
        Calculate p-value for treatment effect using standard OLS inference.
        Uses t-test with (n - k) degrees of freedom where k is number of parameters.
        
        Returns:
            p-value for the treatment coefficient (β₁)
        """
        try:
            y_pred = self.X_matrix @ self.original_model.beta
            residuals = self.Y_matrix - y_pred
            rss = np.sum(residuals ** 2)
            
            df_resid = self.n_samples - self.n_features
            # Require minimum degrees of freedom for reliable inference
            # Standard practice: at least 5-10 df, but we use 3 as minimum
            if df_resid < 3:
                return 1.0
                
            mse = rss / df_resid
            
            if hasattr(self.original_model, 'XTX_inv'):
                xtx_inv = self.original_model.XTX_inv
            else:
                return 1.0 
            
            # Variance-covariance matrix: Var(β) = σ² * (X^T X)^(-1)
            # For standard errors, we need diagonal elements: Var(βᵢ) = σ² * (X^T X)ᵢᵢ^(-1)
            var_beta = mse * np.diag(xtx_inv)
            
            # Handle numerical issues: clip negative variances to zero (shouldn't happen in theory)
            # This can occur due to floating point errors in matrix inversion
            var_beta = np.clip(var_beta, 0, None)
            
            # Compute standard errors (sqrt of variance)
            with np.errstate(invalid='ignore'):
                se_beta = np.sqrt(var_beta)
            
            # Set any invalid (NaN/Inf) or very small SEs to infinity to make p-value = 1
            se_beta = np.where((se_beta < 1e-10) | ~np.isfinite(se_beta), np.inf, se_beta)
            
            t_stat = self.original_model.beta.flatten() / se_beta
            
            # Two-tailed t-test: P(|T| > |t|)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df_resid))
            
            # Return p-value for treatment coefficient (index 1: intercept=0, treatment=1)
            return p_values[1] 
        except Exception:
            return 1.0

    def get_ate_difference(self, removed_indices, approx=False, update=True):
        if not removed_indices:
            return 0.0
        
        current_ate = self.original_ate
        
        if isinstance(removed_indices, int):
            removed_indices = [removed_indices]
        X_remove = self.design_matrix.loc[removed_indices].values
        Y_remove = self.Y.loc[removed_indices].values.reshape(-1, 1)
        
        if approx:
            XTX_inv_updated = self.original_model.neumann_update(X_remove)
        else:
            XTX_inv_updated = self.original_model.woodbury_update(X_remove)
        beta_updated = XTX_inv_updated @ (self.X_matrix.T @ self.Y_matrix - X_remove.T @ Y_remove)
        
        new_ate = float(beta_updated[1].item())
        if update:
            self.original_model.XTX_inv = XTX_inv_updated
            self.original_model.beta = beta_updated
            self.original_ate = new_ate
        
            keep_indices = [i for i in self.X.index if i not in removed_indices]
            self.X = self.X.loc[keep_indices]
            self.T = self.T.loc[keep_indices]
            self.Y = self.Y.loc[keep_indices]
            
            if hasattr(self, 'design_matrix'):
                self.design_matrix = self.design_matrix.loc[keep_indices]
            
            self.X_matrix = self.design_matrix.values
            self.Y_matrix = self.Y.values.reshape(-1, 1)
            self.n_samples = self.X_matrix.shape[0]
        
        return new_ate - current_ate

    def calculate_updated_ATE(self, removed_indices, approx=False):
        if not removed_indices:
            return self.original_ate
        
        self.get_ate_difference(removed_indices, approx=approx, update=True)
        return self.original_ate
    

class ATEUpdateLogistic:
    def __init__(self, X, T, Y, lambda_reg=0.1, max_iter=1000):
        # Store the original dataset
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y)
        
        # Model parameters
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        
        # Initialize and train the original model
        self.original_model = CertifiableUnlearningLogisticRegression(lambda_reg=lambda_reg, max_iter=max_iter)
        # self.original_model = LogisticRegression(C=1 / self.lambda_reg, max_iter=self.max_iter)
        self.original_model.fit(self.X.values, self.T.values)
        
        # Compute the original ATE
        self.original_ate = self._compute_ate_ipw_unlearning(self.T, self.Y, self.X, model=self.original_model)
        # print(f"Original ATE: {self.original_ate}")
        
        # Store all available indices
        self.available_indices = list(range(len(self.X)))
    
    def _compute_ate_ipw_unlearning(self, T, Y, X, model=None, removed_index=None):
        if model is None:
            # Train a new model
            model = CertifiableUnlearningLogisticRegression(lambda_reg=self.lambda_reg, max_iter=self.max_iter)
            # model = LogisticRegression(C=1 / self.lambda_reg, max_iter=self.max_iter)
            model.fit(X.values, T.values)
        elif removed_index is not None:
            # Create copies of data without the removed points for unlearning
            if isinstance(removed_index, int):
                removed_index = [removed_index]
                
            X_remove = X.iloc[removed_index].values
            T_remove = T.iloc[removed_index].values
            
            # Keep data
            keep_indices = [i for i in range(len(X)) if i not in removed_index]
            X_keep = X.iloc[keep_indices].values
            T_keep = T.iloc[keep_indices].values

            # Apply incremental mini-batch unlearning
            model.fit_incremental_mini_batch(
                X_keep, T_keep, X_remove, T_remove, sigma=0, batch_size=len(X_remove)
            )
        
        # Compute propensity scores
        propensity_scores = []
        for i in range(len(X)):
            prob = model.sigmoid(X.iloc[i:i+1].values @ model.theta)
            propensity_scores.append(prob[0])
        propensity_scores = np.array(propensity_scores)
        # propensity_scores = model.predict_proba(X.values)[:, 1]
        
        # If we're removing indices, exclude them from ATE calculation
        if removed_index is not None:
            if isinstance(removed_index, int):
                removed_index = [removed_index]
                
            # Create mask for indices to include
            include_mask = np.array([i not in removed_index for i in range(len(X))])
            
            # Apply mask
            T_filtered = T[include_mask]
            Y_filtered = Y[include_mask]
            ps_filtered = propensity_scores[include_mask]
            
            treated_mask = (T_filtered == 1)
            control_mask = (T_filtered == 0)
            
            # For treated units
            weighted_sum_treated = np.sum(Y_filtered[treated_mask] / ps_filtered[treated_mask])
            weight_total_treated = np.sum(1 / ps_filtered[treated_mask])
            weighted_mean_treated = weighted_sum_treated / weight_total_treated
    
            # For control units
            weighted_sum_control = np.sum(Y_filtered[control_mask] / (1 - ps_filtered[control_mask]))
            weight_total_control = np.sum(1 / (1 - ps_filtered[control_mask]))
            weighted_mean_control = weighted_sum_control / weight_total_control
        else:
            # min_propensity = 0.01
            # max_propensity = 0.99
            # propensity_scores = np.clip(propensity_scores, min_propensity, max_propensity)
            
            treated_mask = (T == 1)
            control_mask = (T == 0)
            
            # For treated units
            weighted_sum_treated = np.sum(Y[treated_mask] / propensity_scores[treated_mask])
            weight_total_treated = np.sum(1 / propensity_scores[treated_mask])
            weighted_mean_treated = weighted_sum_treated / weight_total_treated
    
            # For control units
            weighted_sum_control = np.sum(Y[control_mask] / (1 - propensity_scores[control_mask]))
            weight_total_control = np.sum(1 / (1 - propensity_scores[control_mask]))
            weighted_mean_control = weighted_sum_control / weight_total_control

        # ATE estimate
        return weighted_mean_treated - weighted_mean_control
    
    def get_ate_difference(self, removed_indices, method='unlearning'):
        """
        Compute the difference in ATE after removing specified indices.
        Permanently updates the model and dataset.
        
        Parameters:
        removed_indices (int or list): Index or indices of data points to remove
        method (str): Method to use - 'unlearning' or 'retrain'
        
        Returns:
        float: Difference between updated ATE and original ATE
        """
        if not removed_indices:
            return 0.0
            
        # Store the current ATE
        current_ate = self.original_ate
        
        if isinstance(removed_indices, int):
            removed_indices = [removed_indices]
        
        if method == 'unlearning':
            # Extract rows to be removed
            X_remove = self.X.iloc[removed_indices].values
            T_remove = self.T.iloc[removed_indices].values
            
            # Keep data
            keep_indices = [i for i in range(len(self.X)) if i not in removed_indices]
            X_keep = self.X.iloc[keep_indices].values
            T_keep = self.T.iloc[keep_indices].values
            
            # Update the model using unlearning
            self.original_model.fit_incremental_mini_batch(
                X_keep, T_keep, X_remove, T_remove, sigma=0, batch_size=len(X_remove)
            )
        
        elif method == 'retrain':
            # Keep data
            keep_indices = [i for i in range(len(self.X)) if i not in removed_indices]
            X_keep = self.X.iloc[keep_indices].values
            T_keep = self.T.iloc[keep_indices].values
            
            # Retrain the model from scratch
            self.original_model = CertifiableUnlearningLogisticRegression(lambda_reg=self.lambda_reg, max_iter=self.max_iter)
            self.original_model.fit(X_keep, T_keep)
        
        else:
            raise ValueError("Method must be either 'unlearning' or 'retrain'")
        
        # Update the dataset
        keep_indices = [i for i in range(len(self.X)) if i not in removed_indices]
        # self.X = self.X.iloc[keep_indices].reset_index(drop=True)
        # self.T = self.T.iloc[keep_indices].reset_index(drop=True)
        # self.Y = self.Y.iloc[keep_indices].reset_index(drop=True)
        self.X = self.X.iloc[keep_indices]
        self.T = self.T.iloc[keep_indices]
        self.Y = self.Y.iloc[keep_indices]
        
        # Update available indices
        self.available_indices = [idx for i, idx in enumerate(self.available_indices) if i not in removed_indices]
        
        # Update the ATE
        self.original_ate = self._compute_ate_ipw_unlearning(self.T, self.Y, self.X, model=self.original_model)
        
        return self.original_ate - current_ate

    def get_original_ate(self):
        """
        Get the current ATE (treatment effect).
        
        Returns:
        float: Current ATE
        """
        return self.original_ate

    def calculate_updated_ate(self, removed_indices, method='unlearning'):
        """
        Calculate updated ATE after removing specified indices.
        Permanently updates the model and dataset.
        
        Parameters:
        removed_indices (int or list): Index or indices of data points to remove
        method (str): Method to use - 'unlearning' or 'retrain'
        
        Returns:
        float: Updated ATE after removal
        """
        # Get the old ATE for returning the difference (backwards compatibility)
        old_ate = self.original_ate
        
        # Update model and dataset, and return new ATE
        self.get_ate_difference(removed_indices, method=method)
        return self.original_ate
    
    
    def _identify_confounders(self):
        """
        Use DoWhy to identify confounders.
        
        Returns:
        --------
        list
            List of column names identified as confounders
        """
        try:
            import dowhy
            from dowhy import CausalModel
            import warnings
            warnings.filterwarnings('ignore')  # Suppress DoWhy warnings
            
            # Prepare data
            data = self.X.copy()
            data['treatment'] = self.T.values
            data['outcome'] = self.Y.values
            
            # Create causal graph
            feature_names = self.X.columns.tolist()
            edges = []
            for feat in feature_names:
                edges.append(f"{feat} -> treatment")
                edges.append(f"{feat} -> outcome")
            edges.append("treatment -> outcome")
            
            graph = "digraph {" + "; ".join(edges) + "}"
            
            # Create causal model
            model = CausalModel(
                data=data,
                treatment='treatment',
                outcome='outcome',
                graph=graph,
                approach="backdoor"
            )
            
            # Identify effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Extract confounders
            if hasattr(identified_estimand, 'backdoor_variables') and identified_estimand.backdoor_variables:
                return identified_estimand.backdoor_variables
            else:
                return self.X.columns.tolist()
                
        except ImportError:
            print("DoWhy not installed. Using all variables as potential confounders.")
            return self.X.columns.tolist()
        except Exception as e:
            print(f"Error in confounder identification: {e}. Using all variables.")
            return self.X.columns.tolist()
    