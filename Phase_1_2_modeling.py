import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

print("FASE 1: EDA & Preprocessing")
# Load data
print("Loading dataset")
df = pd.read_csv("fremotor1prem0304a.csv")

# 1. Drop kolom tidak relevan
drop_cols = [
    'IDpol', 'MaritalStatus', 'JobCode', 
    'PremWindscreen', 'PremDamAll', 'PremFire', 'PremAcc1', 
    'PremAcc2', 'PremLegal', 'PremTPLM', 'PremTPLV', 'PremServ', 'PremTheft'
]
df = df.drop(columns=drop_cols)


df = df[df['PremTot'] > 0].copy()

# Sensitive attribute DrivGender
df['D_gender_binary'] = df['DrivGender'].map({'F': 1, 'M': 0})

# Base feature
X_base = df.drop(columns=['PremTot', 'DrivGender', 'D_gender_binary'])

# One hot encoding 
print("Encoding variables")
X_encoded = pd.get_dummies(X_base, drop_first=True, dtype=float)

df_final = pd.concat([df[['PremTot', 'D_gender_binary', 'DrivGender']], X_encoded], axis=1)

# Split 70:30
print("Splitting dataset")
train_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final['D_gender_binary'])

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")


def calc_dir(y_pred, d_binary):
    mean_f = np.mean(y_pred[d_binary == 1])
    mean_m = np.mean(y_pred[d_binary == 0])
    return mean_f / mean_m if mean_m > 0 else 0

baseline_dir = calc_dir(test_df['PremTot'].values, test_df['D_gender_binary'].values)
print(f"Baseline DIR (actual data): {baseline_dir:.4f} (Ideal: 1.0)")

print("FASE 2: Training Base Models ")

X_cols = X_encoded.columns.tolist()
D_train = train_df['D_gender_binary'].values
Y_train = train_df['PremTot'].values
X_train_df = train_df[X_cols].copy()

D_test = test_df['D_gender_binary'].values
Y_test = test_df['PremTot'].values
X_test_df = test_df[X_cols].copy()

# -- Model 1: MO (Orthogonal) --
print("Training Model 1: MO (Orthogonal)")
X_train_mo = X_train_df.copy()
X_test_mo = X_test_df.copy()

for col in X_cols:
    lr = LinearRegression()
    lr.fit(D_train.reshape(-1, 1), X_train_df[col])
    X_train_mo[col] = X_train_df[col] - lr.predict(D_train.reshape(-1, 1))
    X_test_mo[col] = X_test_df[col] - lr.predict(D_test.reshape(-1, 1))

# Train Gamma GLM for MO
mo_model = sm.GLM(Y_train, sm.add_constant(X_train_mo), family=sm.families.Gamma(link=sm.families.links.Log())).fit()
preds_mo = mo_model.predict(sm.add_constant(X_test_mo, has_constant='add'))
rmse_mo = np.sqrt(mean_squared_error(Y_test, preds_mo))
dir_mo = calc_dir(preds_mo, D_test)
print(f"MO   -> RMSE: {rmse_mo:.2f}, DIR: {dir_mo:.4f}")

# -- Model 2: MDF (Discrimination-Free) --
print("Training Model 2: MDF (Discrimination-Free)")
X_train_mdf = X_train_df.copy()
X_train_mdf['D_gender'] = D_train

# Train Gamma GLM for MDF
mdf_model = sm.GLM(Y_train, sm.add_constant(X_train_mdf), family=sm.families.Gamma(link=sm.families.links.Log())).fit()

# Predictions for MDF: average over empirical distribution
p_F = np.mean(D_train == 1)
p_M = 1 - p_F

X_test_mdf_F = X_test_df.copy()
X_test_mdf_F['D_gender'] = 1
preds_mdf_F = mdf_model.predict(sm.add_constant(X_test_mdf_F, has_constant='add'))

X_test_mdf_M = X_test_df.copy()
X_test_mdf_M['D_gender'] = 0
preds_mdf_M = mdf_model.predict(sm.add_constant(X_test_mdf_M, has_constant='add'))

preds_mdf = p_F * preds_mdf_F + p_M * preds_mdf_M
rmse_mdf = np.sqrt(mean_squared_error(Y_test, preds_mdf))
dir_mdf = calc_dir(preds_mdf, D_test)
print(f"MDF  -> RMSE: {rmse_mdf:.2f}, DIR: {dir_mdf:.4f}")

# Model 3: MSCM (Synthetic Control)
print("Training Model 3: MSCM (Synthetic Control)")
# 1. Train RF to get feature importance
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train_df, Y_train)
W = rf.feature_importances_

# 2. Compute weighted distance & find nearest neighbor from opposite gender
def find_counterfactual_y(X_data, Y_data, D_data, W):
    X_F = X_data[D_data == 1].values * np.sqrt(W)
    Y_F = Y_data[D_data == 1]
    
    X_M = X_data[D_data == 0].values * np.sqrt(W)
    Y_M = Y_data[D_data == 0]
    
    Y_cf = np.zeros(len(Y_data))
    
    nn_F = NearestNeighbors(n_neighbors=1).fit(X_F)
    nn_M = NearestNeighbors(n_neighbors=1).fit(X_M)
    
    # For Females, find NN in Males
    _, idx_m = nn_M.kneighbors(X_data[D_data == 1].values * np.sqrt(W))
    Y_cf[D_data == 1] = Y_M[idx_m.flatten()]
    
    # For Males, find NN in Females
    _, idx_f = nn_F.kneighbors(X_data[D_data == 0].values * np.sqrt(W))
    Y_cf[D_data == 0] = Y_F[idx_f.flatten()]
    
    return Y_cf

Y_cf_train = find_counterfactual_y(X_train_df, Y_train, D_train, W)
Y_train_mscm = (Y_train + Y_cf_train) / 2.0

X_train_mscm = X_train_df.copy()
X_train_mscm['D_gender'] = D_train
mscm_model = sm.GLM(Y_train_mscm, sm.add_constant(X_train_mscm), family=sm.families.Gamma(link=sm.families.links.Log())).fit()

X_test_mscm = X_test_df.copy()
X_test_mscm['D_gender'] = D_test
preds_mscm = mscm_model.predict(sm.add_constant(X_test_mscm, has_constant='add'))
rmse_mscm = np.sqrt(mean_squared_error(Y_test, preds_mscm))
dir_mscm = calc_dir(preds_mscm, D_test)
print(f"MSCM -> RMSE: {rmse_mscm:.2f}, DIR: {dir_mscm:.4f}")


# Model 4: MU (Model Unawareness) 
print("Training Model 4: MU (Model Unawareness)")
X_train_mu = X_train_df.copy()
mu_model = sm.GLM(Y_train, sm.add_constant(X_train_mu), family=sm.families.Gamma(link=sm.families.links.Log())).fit()
preds_mu = mu_model.predict(sm.add_constant(X_test_df, has_constant='add'))
rmse_mu = np.sqrt(mean_squared_error(Y_test, preds_mu))
dir_mu = calc_dir(preds_mu, D_test)
print(f"MU   -> RMSE: {rmse_mu:.2f}, DIR: {dir_mu:.4f}")


print("Summary")
df_res = pd.DataFrame({
    'Model': ['MO', 'MDF', 'MSCM', 'MU'],
    'RMSE': [rmse_mo, rmse_mdf, rmse_mscm, rmse_mu],
    'DIR': [dir_mo, dir_mdf, dir_mscm, dir_mu],
    '|1 - DIR|': [abs(1-dir_mo), abs(1-dir_mdf), abs(1-dir_mscm), abs(1-dir_mu)]
})
print(df_res)

print(" Exporting Data for Phase 3")
# Save the test set predictions, true values, and features
export_df = X_test_df.copy()
export_df['D_gender_binary'] = D_test
export_df['PremTot_actual'] = Y_test
export_df['pred_MO'] = preds_mo
export_df['pred_MDF'] = preds_mdf
export_df['pred_MSCM'] = preds_mscm
export_df['pred_MU'] = preds_mu

export_df.to_csv("phase12_predictions.csv", index=False)
print("Saved predictions to phase12_predictions.csv")
