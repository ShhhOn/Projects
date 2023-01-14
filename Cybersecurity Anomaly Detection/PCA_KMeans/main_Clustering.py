import os
import time
import numpy as np
import pandas as pd
from joblib.numpy_pickle_utils import xrange
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import seaborn as sns
import random

random.seed(10)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

startTime = time.time()


def elbow_plot_PCA(X_hat, principalComponents, pc, case_label, save_folder,selected_k):
    n = range(1, 10)
    inertia = []
    for i in n:
        mod = KMeans(n_clusters=i)

        mod.fit(principalComponents[:, :pc])
        inertia.append(mod.inertia_)

    plt.plot(n, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method: Finding Optimal K Value')

    plt.plot((selected_k), inertia[(selected_k-1)], 'rx')

    plt.savefig(save_folder + case_label + "_elbowplot.png")
    plt.close()
    return


def PCA_transform(X_hat, case_label, execute_plots):
    pca = PCA()
    principalComponents = pca.fit_transform(X_hat)
    explained_variane = pca.explained_variance_ratio_
    components = abs(pca.components_)

    if execute_plots:
        features = range(pca.n_components_)
        plt.figure(figsize=(10, 8))
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('Variance %')
        plt.xticks(features)
        plt.title("Variance % vs PCA Features")
        plt.savefig((save_folder + case_label + "_pca_variance.png"))
        plt.close()

    PCA_components = pd.DataFrame(principalComponents)

    if execute_plots:
        plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title("PCA 1 vs. PCA 2")
        plt.savefig((save_folder + case_label + "_pca1_pca2.png"))
        plt.close()
    return principalComponents, components, explained_variane


def kmeans_missing(X, n_clusters, max_iter, rs_var):
    # Source for Methodology:https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data
    for i in xrange(max_iter):
        # create idx for missing values
        missing = ~np.isfinite(X)
        mu = np.nanmean(X, 0, keepdims=1)
        X_hat = np.where(missing, mu, X)
        cls = KMeans(n_clusters, random_state=rs)
        # fit with filled in data
        fitted_model = cls.fit(X_hat)
        labels = cls.predict(X_hat)
        centroids = cls.cluster_centers_
        # replace missing idx with new labels
        X_hat[missing] = centroids[labels][missing]

        # if labels stop changing, clustering is complete
        if i > 0 and np.all(labels == prev_labels):
            break

            prev_labels = labels
            prev_centroids = cls.cluster_centers_

    return labels, centroids, X_hat, fitted_model


def cluster_analysis(mod, principalComponents, pc_components, clusters_n, case_label):
    clusterDistance = mod.transform(principalComponents[:, :pc_components])
    labels = mod.predict(principalComponents[:, :pc_components])
    sns.scatterplot(x = mod.cluster_centers_[:, 0], y = mod.cluster_centers_[:, 1],
                    marker='+',
                    color='black',
                    s=200)
    sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1], hue=labels,
                    palette=sns.color_palette("Set1", n_colors=clusters_n))

    plt.savefig(save_folder + case_label + '_xNy.png')

    sns.scatterplot(x = mod.cluster_centers_[:, 1], y = mod.cluster_centers_[:, 2],
                    marker='+',
                    color='black',
                    s=200)
    sns.scatterplot(x=principalComponents[:, 1], y=principalComponents[:, 2], hue=labels,
                    palette=sns.color_palette("Set1", n_colors=clusters_n))

    plt.savefig(save_folder + case_label + "_yNz.png")

    return


def outlier_label(mod, principalComponents, pc_components, clusters_n, original, threshold):
    outlier_set = original
    outlier_set['outlier'] = np.zeros((len(original), 1)).astype(bool)
    clusterDistance = mod.transform(principalComponents[:, :pc_components])
    labels = mod.predict(principalComponents[:, :pc_components])
    clusterDistance = clusterDistance.min(axis=1)

    for i in range(clusters_n):

        current_cluster = original[original['clusters'] == i]
        idx = np.where(original['clusters'] == i)
        current_distances = clusterDistance[original['clusters'] == i]
        csd = np.std(current_distances)
        cavg = np.mean(current_distances)
        mcsd = csd * threshold

        thres = mcsd + cavg

        outlier_idx = current_distances > thres

        for i2 in range(len(outlier_idx)):
            c_idx = idx[0][i2]
            outlier_set.iloc[c_idx, -1] = outlier_idx[i2]

    return outlier_set


def plot_clusters_loop(X_hat, labels, centroids, case_label,original):
    df = pd.DataFrame(data=X_hat)

    lbl_values = np.unique(labels)
    # # Creating figure
    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    legend_lst = []
    clst_lbl_lst = []
    for i in lbl_values:
        label_n = df[labels == i]
        a = ax.scatter3D(label_n.iloc[:, 0], label_n.iloc[:, 1], label_n.iloc[:, 2])
        ax.scatter3D(centroids[i, 0], centroids[i, 1], centroids[i, 2], color='red', s=100)

        tmp_lbl_lgd = "Cluster " + str(i)
        legend_lst.append(a)
        clst_lbl_lst.append(tmp_lbl_lgd)

    ax.set_xlabel('Component 1', fontweight='bold')
    ax.set_ylabel('Component 2', fontweight='bold')
    ax.set_zlabel('Component 3', fontweight='bold')
    plt.title("Clusters by PCA Component")
    plt.legend(handles = legend_lst, labels = clst_lbl_lst)
    if original:
        plt.savefig(save_folder + case_label + '_original_plotted.png')
    else:
        plt.savefig(save_folder + case_label + '_clusters_plotted.png')


    plt.close()
    debug = 1
    return


def grade_outlier(outlier_set):
    normal_state = outlier_set['Normal_Attack'] == 0
    attack_state = outlier_set['Normal_Attack'] == 1
    model_attack = outlier_set['outlier'] == 1
    model_normal = outlier_set['outlier'] == 0

    normal_overall_cnt = sum(normal_state)
    attack_overall_cnt = sum(attack_state)

    model_attack_cnt = sum(model_attack)
    model_normal_cnt = sum(model_normal)

    true_positive = (sum(normal_state & model_normal))
    true_negative = (sum(attack_state & model_attack))

    false_positive = (sum(normal_state & model_attack))
    false_negative = (sum(attack_state & model_normal))
    accuracy = (true_positive + true_negative) / (normal_overall_cnt + attack_overall_cnt)

    data = np.array((true_positive, true_negative, false_positive, false_negative, accuracy)).reshape(1, 5)
    df = pd.DataFrame(data, columns=['True_Positive', 'True_negative', 'False_Positive', 'False_negative', 'Accuracy'])

    # print("True Positives: " + str(true_positive))
    # print("True Negatives: " + str(true_negative))
    # print("False Positives: " + str(false_positive))
    # print("False Negatives: " + str(false_negative))
    # print("Accurracy: " + str(accuracy))
    debug = 1

    return df


def execute_all(file_dir, options, variables, save_folder, case_num):
    df = pd.read_csv(file_dir)
    original = df

    df = df.drop(['Normal_Attack'], axis=1)
    df = df.iloc[:, 1:]

    ## produce plots
    execeute_transform_plots = options[0]
    execute_elbow_plot = options[1]
    execute_cluster_analysis = options[2]
    execute_plot_clusters = options[3]
    execute_plot_clusters_original = options[4]
    execute_Grade_Outlier = options[5]
    execute_save_cluster_file = options[6]
    case_label = "Case_" + str(case_num)
    clusters_n = int(variables['clusters'])
    pc_components = int(variables['PCA_components'])
    threshold = int(variables['threshold'])

    scale = StandardScaler()
    scale.fit(df)
    StandardScaler(with_mean=True, with_std=True, copy=True)
    X_norm = scale.transform(df)

    [principalComponents, components, explained_variane] = PCA_transform(X_norm, case_label, execeute_transform_plots)

    if execute_elbow_plot:
        elbow_plot_PCA(principalComponents, X_norm, pc_components, case_label, save_folder,clusters_n)

    mod = KMeans(n_clusters=clusters_n, random_state=rs)
    mod.fit(principalComponents[:, :pc_components])
    labels = mod.predict(principalComponents[:, :pc_components])

    centroids = mod.cluster_centers_
    original['clusters'] = labels

    if execute_cluster_analysis:
        cluster_analysis(mod, principalComponents, pc_components, clusters_n, case_label)

    if execute_plot_clusters:
        plot_clusters_loop(principalComponents[:, :pc_components], labels, centroids, case_label,False)

    if execute_plot_clusters_original:
        attk_labels = original['Normal_Attack']
        plot_clusters_loop(principalComponents[:, :pc_components], attk_labels, centroids, case_label,True)

    outlier_set = outlier_label(mod, principalComponents, pc_components, clusters_n, original, threshold)

    if execute_Grade_Outlier:
        results = grade_outlier(outlier_set)

    if execute_save_cluster_file:
        original.to_csv((save_folder + "/" + case_label + "_SWaT_clusters.csv"))

    return results


if __name__ == "__main__":
    rs = 42
    dirname = os.path.dirname(__file__)
    sub_folder = "/SWAt2019/P6/"
    file_name = "SWaT_Dataset_2019_P6.csv"
    file_dir = dirname + sub_folder + file_name

    save_folder = dirname + sub_folder
    trial = True

    if trial:
        options = [False, False, False, False, False, True, False]
        test_cases_str = "/SWAt2019/P6/test_case_P6.csv"
    else:
        options = [True, True, True, True, True, True, True, True]
        test_cases_str = "/SWAt2019/P6/best_case_P6.csv"

    test_Cases = dirname + test_cases_str
    test_cases = pd.read_csv(test_Cases)
    df = pd.DataFrame(columns=['True_Positive', 'True_negative', 'False_Positive', 'False_negative', 'Accuracy'])
    for i in range(len(test_cases.index)):
        variables = test_cases.iloc[i, :]
        df2 = execute_all(file_dir, options, variables, save_folder, i)

        df = pd.concat([df, df2], axis=0)

    df = df.reset_index()
    df = pd.concat([test_cases, df], axis=1)
    df.to_csv(save_folder + "/Results.csv")
    debug = 1
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
