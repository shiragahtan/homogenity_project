import json
import numpy as np
import pandas as pd
from linear_model_unlearning import CertifiableUnlearningLogisticRegression, BaseLinearRegression
from sklearn.linear_model import LogisticRegression
from numpy.linalg import LinAlgError

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

TREATMENT_COL = config['TREATMENT_COL']

def calculate_ate_safe(df, treatment_col, outcome_col, ret_obj=False):
    """
    Calculate ATE safely with error handling, similar to naive_DFS_algorithm.py
    """
    try:
        if df.empty or df[treatment_col].nunique() < 2:
            return 0.0
        
        # Get feature columns excluding treatment and outcome
        features_cols = [col for col in df.columns if col not in [treatment_col, TREATMENT_COL, outcome_col]]
        
        # Drop every column that is constant in this slice
        features_cols = [c for c in features_cols if df[c].nunique() > 1]
        if not features_cols:  # nothing varies → skip slice
            return 0.0
        
        try:
            ate_obj = ATEUpdateLinear(
                df[features_cols],
                df[TREATMENT_COL],
                df[outcome_col]
            )
            cate_value = ate_obj.get_original_ate()
            return cate_value if not ret_obj else ate_obj
        except LinAlgError:  # XᵀX still singular
            return 0.0 if not ret_obj else None
    except Exception as e:
        import ipdb; ipdb.set_trace()


class ATEUpdateLinear:
    def __init__(self, X, T, Y, find_confounders=False):
        """
        Initialize with dataset and identify confounders using DoWhy.
        The design matrix is created once and preserves the original index
        to allow for efficient slicing for subgroup analysis.
        
        Parameters:
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Covariates/features
        T : pandas.Series or numpy.ndarray
            Treatment indicator (0 or 1)
        Y : pandas.Series or numpy.ndarray
            Outcome variable
        """
        # Ensure inputs are DataFrames/Series with a consistent index
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X,
                                                                           columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T, index=self.X.index)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y, index=self.X.index)

        # Create the intercept Series, preserving the original index
        intercept = pd.Series(1, index=self.X.index, name='intercept')
        
        if find_confounders:
            self.confounders = self._identify_confounders()
            self.confounders = self.confounders if isinstance(self.confounders, list) else self.confounders.get(
                'backdoor')
            X_confounders = self.X[self.confounders] if self.confounders else self.X
            # Create design matrix, ensuring index alignment
            self.design_matrix = pd.concat([intercept, self.T, X_confounders], axis=1)
            column_names = ['intercept', 'treatment'] + (
                self.confounders if self.confounders else self.X.columns.tolist())
            self.design_matrix.columns = column_names
        else:
            # Use all features as confounders
            self.design_matrix = pd.concat([intercept, self.T, self.X], axis=1)
            self.design_matrix.columns = ['intercept', 'treatment'] + self.X.columns.tolist()

        # Convert to numpy for faster computation
        self.X_matrix = self.design_matrix.values
        self.Y_matrix = self.Y.values.reshape(-1, 1)

        # Store dimensions
        self.n_samples = self.X_matrix.shape[0]
        self.n_features = self.X_matrix.shape[1]
        
        # Compute initial linear regression
        self.original_model = BaseLinearRegression(self.X_matrix, self.Y_matrix)
        
        # Store original ATE (treatment effect)
        self.original_ate = float(self.original_model.beta[1].item())
    
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
    
    def get_ate_difference(self, removed_indices, approx=False, update=True):
        """
        Compute the difference in ATE after removing specified data points.
        Permanently updates the model and dataset.
        
        Parameters:
        -----------
        removed_indices : int or list
            Index or indices of data points to remove
        
        Returns:
        --------
        float
            Difference between updated ATE and original ATE
        """
        if not removed_indices:
            return 0.0
        
        # Store current ATE before update
        current_ate = self.original_ate
        
        # Update the model and dataset
        if isinstance(removed_indices, int):
            removed_indices = [removed_indices]

        # Extract rows to be removed
        X_remove = self.design_matrix.loc[removed_indices].values
        Y_remove = self.Y.loc[removed_indices].values.reshape(-1, 1)
        
        if approx:
            # Update inverse using Neumann series
            XTX_inv_updated = self.original_model.neumann_update(X_remove)
        else:
            # Update inverse using Woodbury formula
            XTX_inv_updated = self.original_model.woodbury_update(X_remove)

        beta_updated = XTX_inv_updated @ (self.X_matrix.T @ self.Y_matrix - X_remove.T @ Y_remove)
        
        # Update the ATE
        new_ate = float(beta_updated[1].item())

        if update:
            self.original_model.XTX_inv = XTX_inv_updated
            self.original_model.beta = beta_updated
            self.original_ate = new_ate
        
            keep_indices = [i for i in self.X.index if i not in removed_indices]

            self.X = self.X.loc[keep_indices]
            self.T = self.T.loc[keep_indices]
            self.Y = self.Y.loc[keep_indices]
            
            # Update design matrix if it exists
            if hasattr(self, 'design_matrix'):
                # self.design_matrix = self.design_matrix.iloc[keep_indices].reset_index(drop=True)
                self.design_matrix = self.design_matrix.loc[keep_indices]
            
            self.X_matrix = self.design_matrix.values
            self.Y_matrix = self.Y.values.reshape(-1, 1)
            self.n_samples = self.X_matrix.shape[0]
        
        return new_ate - current_ate

    def get_original_ate(self):
        """
        Get the current ATE (treatment effect).
        
        Returns:
        --------
        float
            Current ATE
        """
        return self.original_ate

    def calculate_updated_ATE(self, removed_indices, approx=False):
        """
        Calculate updated ATE after removing specified data points using the Woodbury method.
        Permanently updates the model and dataset.
        
        Parameters:
        -----------
        indices_to_remove : list
            Index or indices of data points to remove
        
        Returns:
        --------
        float
            Updated ATE after removing specified data points
        """
        if not removed_indices:
            return self.original_ate
        
        # Return the ATE difference to maintain backwards compatibility
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
    