import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


def compute_averages(df):
    avg_stats = {}
    for col in ["Gene", "precision", "recall", "f1", "auc"]:
        avg_stats[f"avg_{col}"] = df[col].mean() if col in df.columns else None
    for col in ["test_target_count", "test_nontarget_count"]:
        avg_stats[f"total_{col}"] = df[col].sum() if col in df.columns else None
    return avg_stats


def save_with_avg(df, avg_stats, path):
    df_out = df.copy()
    avg_row = pd.Series(avg_stats, name="AVERAGE")
    df_out = pd.concat([df_out, avg_row.to_frame().T])
    df_out.to_csv(path)


def run_classifier(embedding_path, embed_dim, global_pca):
    with open(os.path.join(embedding_path, "train_embeddings.pkl"), "rb") as f:
        train_real_embeddings = pickle.load(f)
    with open(os.path.join(embedding_path, "test_embeddings.pkl"), "rb") as f:
        test_real_embeddings = pickle.load(f)

    train_embeddings, test_embeddings = [], []
    train_genes, test_genes = [], []
    for emb in train_real_embeddings:
        if emb["metadata"]["Gene"][0] in ["DGKE", "GAS7"]:
            continue
        if emb["metadata"]["Time"][0] != "D28":
            continue
        train_embeddings.append(emb["embedding"].reshape(1, -1))
        train_genes.append(emb["metadata"]["Gene"])
    for emb in test_real_embeddings:
        if emb["metadata"]["Gene"][0] in ["DGKE", "GAS7"]:
            continue
        if emb["metadata"]["Time"][0] != "D28":
            continue
        test_embeddings.append(emb["embedding"].reshape(1, -1))
        test_genes.append(emb["metadata"]["Gene"])

    train_embeddings = np.array(train_embeddings).squeeze()
    test_embeddings = np.array(test_embeddings).squeeze()

    pca_embeddings_train = None
    pca_embeddings_test = None
    if global_pca:
        for chan_num in range(14):
            channel_embeddings = train_embeddings[
                :, chan_num * embed_dim : (chan_num + 1) * embed_dim
            ]
            pca = PCA(n_components=20)
            pca_model = pca.fit(channel_embeddings)
            train_channel_pca = pca_model.transform(channel_embeddings)
            test_channel_pca = pca_model.transform(
                test_embeddings[:, chan_num * embed_dim : (chan_num + 1) * embed_dim]
            )
            if pca_embeddings_train is None:
                pca_embeddings_train = train_channel_pca
                pca_embeddings_test = test_channel_pca
            else:
                pca_embeddings_train = np.concatenate(
                    (pca_embeddings_train, train_channel_pca), axis=1
                )
                pca_embeddings_test = np.concatenate(
                    (pca_embeddings_test, test_channel_pca), axis=1
                )
    else: 
        pca = PCA(n_components=embed_dim)
        pca_model = pca.fit(train_embeddings)
        
        pca_embeddings_train = pca_model.transform(train_embeddings)
        pca_embeddings_test = pca_model.transform(test_embeddings)
    

    train_labels = [gene[0] for gene in train_genes]
    test_labels = [gene[0] for gene in test_genes]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(pca_embeddings_train)
    X_test_scaled = scaler.transform(pca_embeddings_test)

    unique_genes = list(set(train_labels + test_labels))
    unique_genes = [gene for gene in unique_genes if gene != "non-target"]

    lr_results = {}
    for target_gene in unique_genes:
        train_binary_labels, train_binary_indices = [], []
        for i, label in enumerate(train_labels):
            if label == target_gene:
                train_binary_labels.append("target")
                train_binary_indices.append(i)
            elif label == "non-target":
                train_binary_labels.append("non-target")
                train_binary_indices.append(i)
        test_binary_labels, test_binary_indices = [], []
        for i, label in enumerate(test_labels):
            if label == target_gene:
                test_binary_labels.append("target")
                test_binary_indices.append(i)
            elif label == "non-target":
                test_binary_labels.append("non-target")
                test_binary_indices.append(i)
        X_train_binary = X_train_scaled[train_binary_indices]
        X_test_binary = X_test_scaled[test_binary_indices]

        test_target_count = sum(1 for label in test_binary_labels if label == "target")
        test_nontarget_count = sum(
            1 for label in test_binary_labels if label == "non-target"
        )
        if test_target_count < 5:
            continue
        binary_model = LogisticRegression(
            solver="saga", max_iter=1000, class_weight="balanced", random_state=42
        )
        binary_model.fit(X_train_binary, train_binary_labels)
        test_pred = binary_model.predict(X_test_binary)
        test_proba = binary_model.predict_proba(X_test_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_binary_labels, test_pred, average="binary", pos_label="target"
        )
        try:
            target_proba = (
                test_proba[:, 1]
                if binary_model.classes_[1] == "target"
                else test_proba[:, 0]
            )
            auc = roc_auc_score(
                [1 if label == "target" else 0 for label in test_binary_labels],
                target_proba,
            )
        except:
            auc = 0.0
        lr_results[target_gene] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "test_target_count": test_target_count,
            "test_nontarget_count": test_nontarget_count,
        }

    rf_results = {}
    for target_gene in unique_genes:
        train_binary_labels, train_binary_indices = [], []
        for i, label in enumerate(train_labels):
            if label == target_gene:
                train_binary_labels.append("target")
                train_binary_indices.append(i)
            elif label == "non-target":
                train_binary_labels.append("non-target")
                train_binary_indices.append(i)
        test_binary_labels, test_binary_indices = [], []
        for i, label in enumerate(test_labels):
            if label == target_gene:
                test_binary_labels.append("target")
                test_binary_indices.append(i)
            elif label == "non-target":
                test_binary_labels.append("non-target")
                test_binary_indices.append(i)
        X_train_binary = X_train_scaled[train_binary_indices]
        X_test_binary = X_test_scaled[test_binary_indices]

        test_target_count = sum(1 for label in test_binary_labels if label == "target")
        test_nontarget_count = sum(
            1 for label in test_binary_labels if label == "non-target"
        )
        if test_target_count < 5:
            continue
        binary_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        binary_model.fit(X_train_binary, train_binary_labels)
        test_pred = binary_model.predict(X_test_binary)
        test_proba = binary_model.predict_proba(X_test_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_binary_labels, test_pred, average="binary", pos_label="target"
        )
        try:
            target_proba = (
                test_proba[:, 1]
                if binary_model.classes_[1] == "target"
                else test_proba[:, 0]
            )
            auc = roc_auc_score(
                [1 if label == "target" else 0 for label in test_binary_labels],
                target_proba,
            )
        except:
            auc = 0.0
        rf_results[target_gene] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "test_target_count": test_target_count,
            "test_nontarget_count": test_nontarget_count,
        }

    os.makedirs(embedding_path, exist_ok=True)
    lr_df = pd.DataFrame.from_dict(lr_results, orient="index")
    rf_df = pd.DataFrame.from_dict(rf_results, orient="index")
    lr_avg = compute_averages(lr_df)
    rf_avg = compute_averages(rf_df)
    save_with_avg(
        lr_df,
        lr_avg,
        os.path.join(
            embedding_path, "binary_classification_logistic_regression_results.csv"
        ),
    )
    save_with_avg(
        rf_df,
        rf_avg,
        os.path.join(embedding_path, "binary_classification_random_forest_results.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neuron feature classifier.")
    parser.add_argument(
        "--embedding_path", type=str, required=True, help="Path to embedding directory"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=384, help="Dimension of embeddings"
    )
    parser.add_argument(
        "--global_pca", action='store_true', help="When true, computes PCA on global embeds, when False assumes embed_dim * 14 (channels) embed dim and computes PCA per channel and concats"
    )
    args = parser.parse_args()
    run_classifier(args.embedding_path, args.embed_dim, args.global_pca)
