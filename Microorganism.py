import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Dataset
# ============================================================
## Dataset dimension = (5, 29, 8, 4, 8, 8, 60) = (Operating voltages, variations, subarray index 1, subarray index 2, individual transistor index 1, individual transistor index 2, time)
## Note that our sensor consists of 8*4 subarrays and each subarrays have 8*8 transistors (See Main Fig.1)
## Microorganism was exposed to the device around t=40 (min)

path = os.getcwd()
t_measurement = np.load(os.path.join(path, "Sample_Dataset", "Microorganism_dataset_time.npy"))

# Load subarray-i split files and concat
dataset_parts = []
for i in range(8):
    fname = os.path.join(path, "Sample_Dataset", f"Microorganism_dataset_subI{i}.npy")
    dataset_parts.append(np.load(fname))

# Reconstruct original dataset
dataset = np.stack(dataset_parts, axis=2)
# -> (5,29,8,4,8,8,60)

t_gas = [0, 35, 60]

# ============================================================
# Gas injection → measurement index
# ============================================================
## This is to find index when the gas is injected
def index_measurement_gas_injection(t_measurement, t_gas):
    index_measurement = []
    for i in range(len(t_gas)):
        for index in range(len(t_measurement)):
            if t_measurement[index] > t_gas[i]:
                index_measurement.append(index - 1)
                break
    if t_gas[-1] > t_measurement[-1]:
        index_measurement.append(len(t_measurement) - 1)
    return index_measurement


# ============================================================
# Count NaN per subarray
# ============================================================
## This is to chcek the number of nan values in a subarray. Because subarrays potentially have NaN values due to the faulty transistors,
## we need to check the number of NaN values. If the number of NaN values in a subarray is too high, we can exclude the subarray.

def check_nan(feature_one_subarray, t_measurement, t_gas):
    nan_check = np.zeros((8, 4))
    t = index_measurement_gas_injection(t_measurement, t_gas)[1]

    for col in range(8):
        for row in range(8):
            for i in range(8):
                for j in range(4):
                    if np.isnan(feature_one_subarray[i][j][col][row][t]):
                        nan_check[i][j] += 1
    return nan_check


def idx_high_nan(feature_one_voltage, chip_index, t_measurement, t_gas, lim_nan):
    nan = []
    for i in range(len(chip_index)):
        nan.append(check_nan(feature_one_voltage[chip_index[i]],
                             t_measurement[chip_index[i]],
                             t_gas))

    idx_high = np.zeros((8, 4))
    for k in range(len(nan)):
        for i in range(8):
            for j in range(4):
                if nan[k][i][j] > lim_nan:
                    idx_high[i][j] += 1
    return nan, idx_high


# ============================================================
# Build fictitious arrays
# ============================================================
def fict_array(feature_one_subarray, idx_high, t_measurement, t_gas):
    feature_fict_array = [[] for _ in range(64)]
    feature_fict_array_nonref = [[] for _ in range(64)]

    t = index_measurement_gas_injection(t_measurement, t_gas)[1]

    ref_array = [(0,1),(0,3),(1,0),(1,2),(2,1),(2,3),(3,0),(3,2),
                 (4,1),(4,3),(5,0),(5,2),(6,1),(6,3),(7,0),(7,2)]

    ref_value = []
    ref_value_subarray = [[[] for _ in range(4)] for _ in range(8)]

    n = 0
    for col in range(8):
        for row in range(8):
            for i in range(8):
                for j in range(4):
                    if idx_high[i][j]==0:                      ## exclude if the number of NaN values is too high   
                        
                        if np.isnan(feature_one_subarray[i][j][col][row][t]):
                            feature_fict_array[n].append(np.zeros(60))
                        elif feature_one_subarray[i][j][col][row][t]!=0:
                            feature_fict_array[n].append(feature_one_subarray[i][j][col][row][t]/feature_one_subarray[i][j][col][row]-1)                                        
                        else:
                            feature_fict_array[n].append(np.array(feature_one_subarray[i][j][col][row])-np.array(feature_one_subarray[i][j][col][row]))                        
                        
                        if (i,j) in ref_array:
                            ref_value.append(feature_one_subarray[i][j][col][row][t]/feature_one_subarray[i][j][col][row]-1)
                            ref_value_subarray[i][j].append(feature_one_subarray[i][j][col][row][t]/feature_one_subarray[i][j][col][row]-1)                            
                        elif (i,j) not in ref_array:
                            if np.isnan(feature_one_subarray[i][j][col][row][t]):
                                feature_fict_array_nonref[n].append(np.zeros(60))
                            elif feature_one_subarray[i][j][col][row][t]!=0:
                                feature_fict_array_nonref[n].append(feature_one_subarray[i][j][col][row][t]/feature_one_subarray[i][j][col][row]-1)
                            else:
                                feature_fict_array_nonref[n].append(np.array(feature_one_subarray[i][j][col][row])-np.array(feature_one_subarray[i][j][col][row]))                                       
            n=n+1   

    return feature_fict_array, feature_fict_array_nonref, ref_value, ref_value_subarray


# ============================================================
# Chip labels
# ============================================================
## class 0 : Week 1, EC
## class 1 : Week 1, PSA
## class 2 : Week 1, CA
## class 3 : Week 2, EC
## class 4 : Week 2, PSA
## class 5 : Week 2, CA
## class 6 : Week 3, EC
## class 7 : Week 3, PSA
## class 8 : Week 3, CA

chip_index = np.arange(29)
chip_class = [0,0,0,0,1,1,2,2,2,3,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8]

# ============================================================
# Build dataset
# ============================================================
feature_fict_array_all = []
feature_fict_array_nonref_all = []
y = []
ref = []

lim_nan = 64  ## if lim_nan is 64, we don't exclude any subarrays. The lower the lim_nan is, the more the number of subarrays excluded is.
nan, idx_high = idx_high_nan(dataset[2], chip_index, t_measurement, t_gas, lim_nan)

for i in range(len(chip_index)):
    temp, temp_nonref, temp_ref, _ = fict_array(dataset[2][chip_index[i]], idx_high, t_measurement[chip_index[i]], t_gas)

    ref.append(temp_ref)
    for j in range(64):
        feature_fict_array_all.append(temp[j])
        feature_fict_array_nonref_all.append(temp_nonref[j])
        y.append(chip_class[i])

ref_mean = np.nanmean(ref, axis=1)

# ============================================================
# TSNE-style preprocessing
# ============================================================
## Even though the starting concentration of microorganisms are nominally the same, stochastic growth causes randomness in their final density.
## To address it, we use not only raw dataset, bu also normalized dataset to enhance the stability of our sensing system.

TSNE_points_t = [42, 46, 50]

def TSNE_preprocessing(feature_fict_array_all, feature_fict_array_nonref, TSNE_points_t):
    TSNE_x = [[] for _ in range(len(feature_fict_array_all))]

    # IMPORTANT: use a fixed channel count defined by the first sample (your original behavior)
    n_ch = len(feature_fict_array_nonref[0])

    for i in range(len(feature_fict_array_all)):
        for j in range(n_ch):
            for k in range(len(TSNE_points_t)):
                time = index_measurement_gas_injection(
                    t_measurement[chip_index[int(i/64)]], TSNE_points_t
                )[k]

                v = feature_fict_array_all[i][j][time]
                TSNE_x[i].append(v)

                denom = ref_mean[int(i/64)][time]
                if denom != 0 and not np.isnan(denom):
                    TSNE_x[i].append(v / denom)
                else:
                    TSNE_x[i].append(0.0)

    return np.array(TSNE_x)


TSNE_X = TSNE_preprocessing(feature_fict_array_all,
                           feature_fict_array_nonref_all,
                           TSNE_points_t)

TSNE_X = np.nan_to_num(TSNE_X, nan=0.0, posinf=0.0, neginf=0.0)
X = TSNE_X

# ============================================================
# LDA
# ============================================================
y_num = np.array(y)
X_tr, X_te, y_tr_con, y_te_con = train_test_split(X, y_num, test_size=0.5, random_state=42, stratify=y_num)

y_tr = y_tr_con % 3
y_te = y_te_con % 3

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
])
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
print("Accuracy:", accuracy_score(y_te, y_pred))

# ============================================================
# 2D LDA embedding
# ============================================================
embed = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(n_components=2))
])
Z_tr = embed.fit_transform(X_tr, y_tr)
Z_te = embed.transform(X_te)
    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

vis_clf = LinearDiscriminantAnalysis()
vis_clf.fit(Z_tr, y_tr)   # y_tr: (EC/PSA/CA) 3-class


x_min, x_max = -8, 8
y_min, y_max = -5, 5
h = 0.02

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]
Z_pred = vis_clf.predict(grid).reshape(xx.shape)

# --- plot ---
colors = np.array(['#98223b',  '#8ec0cd', '#5c8c27',
                   '#c94d6d', '#4f7e88', '#a1d35c',
                   '#6a1b2d', '#b3e6f2', '#2f5d13'])

y_gas = ['EC, Week 1','PSA, Week 1','CA, Week 1',
         'EC, Week 2','PSA, Week 2','CA, Week 2',
         'EC, Week 3','PSA, Week 3','CA, Week 3']

plt.figure(figsize=(4,4), dpi=400)

classes = np.unique(y_tr_con)   # 0..8 (9 classes)

num = 0
for c in classes:
    mtr = (y_tr_con == c)
    mte = (y_te_con == c)

    plt.scatter(Z_tr[mtr,0], Z_tr[mtr,1],
                s=40, marker='o', alpha=0.5,
                color=colors[num], label=f"{y_gas[c]} (train)")

    plt.scatter(Z_te[mte,0], Z_te[mte,1],
                s=40, marker='x',
                color=colors[num], label=f"{y_gas[c]} (test)")
    num += 1

plt.contour(xx, yy, Z_pred, colors='grey', linestyles='--', linewidths=1.0, alpha=1.0, zorder=10)

plt.xlabel("LDA axis 1", fontsize=14)
plt.ylabel("LDA axis 2", fontsize=14, labelpad=-2)

plt.tick_params(axis="x", direction='in', length=4, labelsize=13)
plt.tick_params(axis="y", direction='in', length=4, labelsize=13)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.legend(frameon=False, fontsize=11, bbox_to_anchor=(1,1.2), loc='upper left')

#plt.tight_layout()
plt.show()

