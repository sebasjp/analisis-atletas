import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


def get_correlation_circle_data(X, Xpca):

    ccircle = []
    eucl_dist = []
    for i,j in enumerate(X.T):
        corr1 = np.corrcoef(j, Xpca[:,0])[0,1]
        corr2 = np.corrcoef(j, Xpca[:,1])[0,1]
        ccircle.append((corr1, corr2))
        eucl_dist.append(np.sqrt(corr1**2 + corr2**2))

    return ccircle, eucl_dist


def plot_correlation_circle(X, Xpca, feature_names: list):

    ccircle, eucl_dist = get_correlation_circle_data(X, Xpca)
    
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        fig, axs = plt.subplots(figsize=(6, 6))
        for i,j in enumerate(eucl_dist):
            arrow_col = plt.cm.cividis((eucl_dist[i] - np.array(eucl_dist).min())/\
                                    (np.array(eucl_dist).max() - np.array(eucl_dist).min()) )
            axs.arrow(0,0, # Arrows start at the origin
                    ccircle[i][0],  #0 for PC1
                    ccircle[i][1],  #1 for PC2
                    lw=1, # line width
                    length_includes_head=True, 
                    color=arrow_col,
                    fc = arrow_col,
                    head_width=0.05,
                    head_length=0.05)
            axs.text(
                ccircle[i][0], 
                ccircle[i][1], 
                feature_names[i], 
                horizontalalignment="center", 
                verticalalignment="center"
            )
        # Draw the unit circle, for clarity
        circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
        axs.add_patch(circle)
        axs.set_xlabel("PCA 1")
        axs.set_ylabel("PCA 2")
    plt.tight_layout()
    plt.show()


def plot_individuals(Xpca, color: list=None):

    # array to DataFrame
    Xpca_df = pd.DataFrame(
        Xpca, 
        columns=[f"PC{i+1}" for i in range(Xpca.shape[1])]
    )

    # Create a scatter plot to visualize the observations in the 2D PCA space
    plt.figure(figsize=(10, 6))
    if color is not None:
        Xpca_df["color"] = color
        sns.scatterplot(
            x="PC1", y="PC2",
            hue="color",
            data=Xpca_df
        )
    else:
        sns.scatterplot(
            x="PC1", y="PC2",
            data=Xpca_df
        )
    # plt.scatter(
    #     Xpca[:, 0], # position on the first principal component of the observations
    #     Xpca[:, 1], # position on the second principal component of the observations
    #     alpha=0.7,
    #     c=color,
    # )
    # Add title and axis label
    plt.title('Scatter Plot of Observations in 2D PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # (optionally) Add labels to each point based on their index in the original dataframe
    for i in range(Xpca.shape[0]):
        # plt.annotate(df_sit.index[i][0], (Xpca[i, 0], Xpca[i, 1]), fontsize=8)
        plt.annotate(i, (Xpca[i, 0], Xpca[i, 1]), fontsize=8)
        # This might be useful when doing outlier detection

    # Add grid in the background
    plt.grid(True)
    plt.show()


def biplot(X, Xpca, feature_names: list,  color: list=None):
    
    # array to DataFrame
    Xpca_df = pd.DataFrame(
        Xpca, 
        columns=[f"PC{i+1}" for i in range(Xpca.shape[1])]
    )
    # factor to scale individuals loadings in order to plot with correlation circle
    scale_factor = 1. / (Xpca_df.max() - Xpca_df.min())
    Xpca_scaled_df = Xpca_df.copy()
    for compcol in Xpca_scaled_df.columns:
        Xpca_scaled_df[compcol] *= scale_factor[compcol]
    
    # correlation circle coords
    ccircle, eucl_dist = get_correlation_circle_data(X, Xpca)

    # individuals plot
    fig, axs = plt.subplots(figsize=(6, 6))

    if color is not None:
        Xpca_df["color"] = color
        sns.scatterplot(
            x="PC1", y="PC2",
            hue="color",
            data=Xpca_scaled_df, 
            ax=axs
        )
    else:
        sns.scatterplot(
            x="PC1", y="PC2",
            data=Xpca_scaled_df, 
            ax=axs
        )
    for i in range(Xpca_scaled_df.shape[0]):
        plt.annotate(i, (Xpca_scaled_df.iloc[i, 0], Xpca_scaled_df.iloc[i, 1]), fontsize=8)

    # add correlation circle coords
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        
        for i,j in enumerate(eucl_dist):
            # arrow colors
            arrow_col = plt.cm.cividis(
                (eucl_dist[i] - np.array(eucl_dist).min())/\
                (np.array(eucl_dist).max() - np.array(eucl_dist).min()) 
            )
            axs.arrow(0,0, # Arrows start at the origin
                ccircle[i][0],  #0 for PC1
                ccircle[i][1],  #1 for PC2
                lw=1, # line width
                length_includes_head=True, 
                color=arrow_col,
                fc = arrow_col,
                head_width=0.05,
                head_length=0.05
            )
            axs.text(
                ccircle[i][0], 
                ccircle[i][1], 
                feature_names[i], 
                horizontalalignment="center", 
                verticalalignment="center"
            )
    plt.axis('off')
    plt.show()


class MyPCA:
    
    def __init__(self, n_components, method):
        self.n_components = n_components
        self.method = method
        
    def fit(self, X):
        # Standardize data 
        X = X.copy()
        self.mean = np.mean(X, axis = 0)
        self.scale = np.std(X, axis = 0)
        X_std = (X - self.mean) / self.scale
        
        # Eigendecomposition of covariance matrix       
        cov_mat = pd.DataFrame(X_std).corr(method=self.method)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat) 
        
        # Adjusting the eigenvectors that are largest in absolute value to be positive    
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs*signs[np.newaxis,:]
        eig_vecs = eig_vecs.T
       
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i,:]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])
        
        self.components = eig_vecs_sorted[:self.n_components,:]
        
        # Explained variance ratio
        self.explained_variance_ratio = [i/np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)
        
        return X_proj