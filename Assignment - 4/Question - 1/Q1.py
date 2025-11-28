import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def generate_data(n_samples, r_minus=2, r_plus=4, sigma=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.random.choice([-1, +1], size=n_samples)
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    r = np.where(y == -1, r_minus, r_plus)
    noise = np.random.normal(0, sigma, (n_samples, 2))
    
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + noise
    return X, y

def svm_cross_validation(X, y, k=5):
    C_list = [0.1, 1, 10, 50, 100]
    gamma_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    print("SVM: Performing 5-fold Cross-Validation")
    print(f"C values: {C_list}")
    print(f"Gamma values: {gamma_list}\n")

    acc_mat = np.zeros((len(gamma_list), len(C_list)))
    best_acc, best_C, best_gamma = -1.0, None, None

    for gi, gamma in enumerate(gamma_list):
        for ci, C in enumerate(C_list):
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")
            acc = scores.mean()
            acc_mat[gi, ci] = acc

            if acc > best_acc:
                best_acc, best_C, best_gamma = acc, C, gamma
            
            print(f"  C={C:>5}, gamma={gamma}: CV Accuracy = {acc:.4f}")

    print(f"\nBest: C={best_C}, gamma={best_gamma}, CV Accuracy={best_acc:.4f}")
    
    best_idx = (gamma_list.index(best_gamma), C_list.index(best_C))
    return best_C, best_gamma, acc_mat, C_list, gamma_list, best_idx

def plot_svm_cv(acc_mat, C_list, gamma_list, best_idx):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(acc_mat, cmap="YlOrRd", aspect='auto')
    plt.title("SVM K-Fold Cross-Validation Accuracy")
    plt.xlabel("C (Box Constraint)")
    plt.ylabel("Gamma (Kernel Width)")
    plt.xticks(range(len(C_list)), C_list)
    plt.yticks(range(len(gamma_list)), gamma_list)

    for i in range(len(gamma_list)):
        for j in range(len(C_list)):
            plt.text(j, i, f"{acc_mat[i, j]:.3f}", ha="center", va="center", fontsize=9)

    bi, bj = best_idx
    plt.scatter(bj, bi, marker="*", s=300, edgecolor="black", facecolor="lime")
    plt.colorbar(im, label="CV Accuracy")
    plt.tight_layout()
    plt.savefig('svm_cv_results.png', dpi=150)
    plt.show()

def mlp_cross_validation(X, y, k=5):
    hidden_sizes = [5, 10, 20, 50, 100]

    print("MLP: Performing 5-fold Cross-Validation")
    print(f"Hidden layer sizes: {hidden_sizes}\n")

    acc_list = []
    best_acc, best_h = -1.0, None

    for h in hidden_sizes:
        model = MLPClassifier(
            hidden_layer_sizes=(h,),
            activation="relu",
            max_iter=1000,
            random_state=42
        )
        scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")
        acc = scores.mean()
        acc_list.append(acc)

        if acc > best_acc:
            best_acc, best_h = acc, h
        
        print(f"  Hidden size={h:>3}: CV Accuracy = {acc:.4f}")

    print(f"\nBest: Hidden size={best_h}, CV Accuracy={best_acc:.4f}")
    
    return best_h, acc_list, hidden_sizes


def plot_mlp_cv(acc_list, hidden_sizes, best_h):
    plt.figure(figsize=(8, 5))
    plt.plot(hidden_sizes, acc_list, 'bo-', linewidth=2, markersize=8)
    
    best_idx = hidden_sizes.index(best_h)
    plt.scatter([best_h], [acc_list[best_idx]], marker="*", s=300, 
                color="red", zorder=5, label=f"Best: {best_h} neurons")
    
    plt.xlabel("Number of Perceptrons in Hidden Layer")
    plt.ylabel("CV Accuracy")
    plt.title("MLP K-Fold Cross-Validation Accuracy")
    plt.xticks(hidden_sizes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mlp_cv_results.png', dpi=150)
    plt.show()

def plot_decision_boundary(model, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.3, levels=[-1.5, 0, 1.5])
    plt.contour(xx, yy, Z, colors='black', linewidths=1.5, levels=[0])    
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=5, alpha=0.5, label='Class -1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=5, alpha=0.5, label='Class +1')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# Generate data
print("DATA GENERATION")
X_train, y_train = generate_data(1000, seed=42)
X_test, y_test = generate_data(10000, seed=123)
print(f"Training samples: 1000, Test samples: 10000")
print(f"r_-1 = 2, r_+1 = 4, sigma = 1\n")

# -------------------- SVM --------------------
print("SUPPORT VECTOR MACHINE (Gaussian/RBF Kernel)")

best_C, best_gamma, acc_mat, C_list, gamma_list, best_idx = \
    svm_cross_validation(X_train, y_train, k=5)

plot_svm_cv(acc_mat, C_list, gamma_list, best_idx)

# Train final SVM
print("\nTraining final SVM with optimal hyperparameters...")
svm_final = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
svm_final.fit(X_train, y_train)

# Test SVM
y_pred_svm = svm_final.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
err_svm = 1 - acc_svm

print(f"\nSVM Test Results:")
print(f"  Probability of Error: {err_svm:.4f}")

plot_decision_boundary(svm_final, X_test, y_test,
    f"SVM Decision Boundary (C={best_C}, gamma={best_gamma})\nP(error) = {err_svm:.4f}",
    'svm_decision_boundary.png')

# -------------------- MLP --------------------
print("MULTI-LAYER PERCEPTRON (Single Hidden Layer, Softmax Output)")

best_h, acc_list, hidden_sizes = mlp_cross_validation(X_train, y_train, k=5)

plot_mlp_cv(acc_list, hidden_sizes, best_h)

# Train final MLP
print("\nTraining final MLP with optimal hyperparameters...")
mlp_final = MLPClassifier(
    hidden_layer_sizes=(best_h,),
    activation="relu",
    max_iter=1000,
    random_state=42
)
mlp_final.fit(X_train, y_train)

# Test MLP
y_pred_mlp = mlp_final.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
err_mlp = 1 - acc_mlp

print(f"\nMLP Test Results:")
print(f"  Probability of Error: {err_mlp:.4f}")

plot_decision_boundary(mlp_final, X_test, y_test,
    f"MLP Decision Boundary ({best_h} neurons)\nP(error) = {err_mlp:.4f}",
    'mlp_decision_boundary.png')

# -------------------- SUMMARY --------------------
print("SUMMARY")
print(f"\nSVM (Gaussian Kernel):")
print(f"  Best C = {best_C}, Best gamma = {best_gamma}")
print(f"  Test P(error) = {err_svm:.4f}")

print(f"\nMLP (Single Hidden Layer):")
print(f"  Best hidden layer size = {best_h}")
print(f"  Test P(error) = {err_mlp:.4f}")
