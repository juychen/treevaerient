# %% [markdown]
# # TreeVAE

# %% [markdown]
# ### Table Of Contents
# 1. [Data Loading](#section_1)
# 2. [Generations](#section_2)
# 3. [Reconstructions](#section_3)
# 4. [Tree and Representation Analysis](#section_4)
# 5. [CelebA Attributes](#section_5)
# 
# This is the notebook for analyzing and visualizing the trees learnt by TreeVAE. 
# 
# Trees can be learnt by running main.py and stored by setting the option save_model to True in the config file.
# 
# 

# %% [markdown]
# ## <a name="section_1"></a> 1. Data Loading

# %% [markdown]
# Always execute this section first. This section loads the data and model and computes the NMI to ensure that the model was loaded correctly. Make sure to set the path in the second cell to the specific model that you want to analyze.

# %%
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from models.model import TreeVAE
import scipy
import os
import yaml
import gc
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from pathlib import Path
from utils.utils import reset_random_seeds, display_image
from utils.data_utils import get_data, get_gen
from utils.training_utils import compute_leaves, predict, move_to
from train.validate_tree import compute_likelihood
from models.model_smalltree import SmallTreeVAE
from models.losses import loss_reconstruction_binary, loss_reconstruction_mse, loss_reconstruction_cov_mse_eval
from utils.model_utils import Node, construct_tree_fromnpy, return_list_tree, construct_data_tree, construct_tree_fromnpy
from utils.plotting_utils import plot_tree_graph, get_node_embeddings, draw_tree_with_scatter_plots

# %%
path = 'models/experiments/'
ex_path = 'waldvarient/20240506-230310_d26be' # INSERT YOUR PATH HERE
checkpoint_path = path+ex_path
with open(checkpoint_path + "/config.yaml", 'r') as stream:
    configs = yaml.load(stream,Loader=yaml.Loader)
print(configs)

# %%
# Load Data
trainset, trainset_eval, testset = get_data(configs)
gen_train = get_gen(trainset, configs, validation=False, shuffle=False)
gen_train_eval = get_gen(trainset_eval, configs, validation=True, shuffle=False)
gen_test = get_gen(testset, configs, validation=True, shuffle=False)
gen_train_eval_iter = iter(gen_train_eval)
gen_test_iter = iter(gen_test)
y_train = trainset_eval.dataset.targets[trainset_eval.indices].numpy()
y_test = testset.dataset.targets[testset.indices].numpy()

# Load Model
n_d = configs['training']['num_clusters_tree']
model = TreeVAE(**configs['training'])
data_tree = np.load(checkpoint_path+'/data_tree.npy', allow_pickle=True)
model = construct_tree_fromnpy(model, data_tree, configs)
if not configs['globals']['eager_mode']:
    model = torch.compile(model)
model.load_state_dict(torch.load(checkpoint_path+'/model_weights.pt'),strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

plot_tree_graph(data_tree)

# %%
data_tree

# %%
# Compute Train NMI
prob_leaves = predict(gen_train_eval, model, device,'prob_leaves')
y = np.squeeze(np.argmax(prob_leaves, axis=-1))
print('Train NMI:',normalized_mutual_info_score(y, np.squeeze(y_train)))

tot_counts = []
print("                                  Leaf", np.arange(10))
for i in np.unique(y_test):
    list_y_hat, counts = np.unique(y[np.squeeze(y_train)==i], return_counts=True)
    for j in range(n_d):
        if j not in list_y_hat:
            list_y_hat = np.insert(list_y_hat, j, j)
            counts = np.insert(counts, j, 0)
    tot_counts.append(counts)
    print(f"Class {i:<10}", list_y_hat, counts)

# %%
# Compute Test NMI
prob_leaves = predict(gen_test, model, device,'prob_leaves')
y = np.squeeze(np.argmax(prob_leaves, axis=-1))
print('Test NMI:', normalized_mutual_info_score(y, np.squeeze(y_test)))

tot_counts = []
print("                                  Leaf", np.arange(10))
for i in np.unique(y_test):
    list_y_hat, counts = np.unique(y[np.squeeze(y_test)==i], return_counts=True)
    for j in range(n_d):
        if j not in list_y_hat:
            list_y_hat = np.insert(list_y_hat, j, j)
            counts = np.insert(counts, j, 0)
    tot_counts.append(counts)
    print(f"Class {i:<10}", list_y_hat, counts)

# %% [markdown]
# ## <a name="section_2"></a> 2. Generations

# %% [markdown]
# This section is concerned with unconditionally generating new samples as opposed to reconstructing existing data points.

# %% [markdown]
# ### Clusterwise generations

# %% [markdown]
# Here, given one unconditional random sampling from the root, we visualize the generations for each leaf. That is, each row corresponds to one sample and each column corresponds to one leaf. Above each generation, we provide the probability of falling into the respective leaf for this sample. 
# 
# This way of visualization can provide insights on the characteristics each leaf is associated with. Observe that the generations differ across the leaves, as each leaf decodes the sample in the style of the cluster that it learnt. It is likely that cluster-differences are observed more strongly than in the reconstructions' section, as here, we have no guiding information from the bottom-up.

# %%
n_imgs = 15
with torch.no_grad():
    reconstructions, p_c_z = model.generate_images(n_imgs, device)
reconstructions = move_to(reconstructions, 'cpu')
for i in range(n_imgs):
    fig, axs = plt.subplots(1, n_d, figsize=(15, 15))
    for c in range(n_d):
        axs[c].imshow(display_image(reconstructions[c][i]), cmap=plt.get_cmap('gray'))
        axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(p_c_z[i][c],))
        axs[c].axis('off')
    plt.show()

# %% [markdown]
# ### Generate new images according to cluster assignment

# %% [markdown]
# In this subsection, given a leaf, we store the first 100 generations, for which this leaf is their most likely cluster assignment. This allows us to gain insights on the cluster and characteristics that each leaf learnt.

# %%
# Here, we store generations for each leaf simultaneously until
# every leaf has n_imgs associated generations, or we iterated through max_iter batches.
n_imgs = configs['training']['batch_size']
max_iter = 200

with torch.no_grad():
    reconstructions, p_c_z = model.generate_images(n_imgs, device)
reconstructions = move_to(reconstructions, 'cpu')
clusterwise_reconst = [torch.zeros_like(reconstructions[0][0:2]) for i in range(len(reconstructions))]
n_iter=0
while min([clusterwise_reconst[leaf_ind].shape[0] for leaf_ind in range(len(reconstructions))]) < n_imgs+2 and n_iter < max_iter:
    for i in range(n_imgs):
        leaf_ind = torch.argmax(p_c_z[i])
        if clusterwise_reconst[leaf_ind].shape[0] < n_imgs+2:
            clusterwise_reconst[leaf_ind] = torch.vstack([clusterwise_reconst[leaf_ind], reconstructions[leaf_ind][i].unsqueeze(0)])
    with torch.no_grad():
        reconstructions, p_c_z = model.generate_images(n_imgs, device)
    reconstructions = move_to(reconstructions, 'cpu')
    n_iter += 1
    if n_iter %10 == 0:
        print(n_iter)
for i in range(len(reconstructions)):
    clusterwise_reconst[i] = clusterwise_reconst[i][2:,:]

# %%
len(clusterwise_reconst)

# %%
# For each leaf, we visualize n_grid x n_grid generations, 
# which have highest probability of being assigned to this cluster
n_leaves = len(clusterwise_reconst)
n_grid = min(5,int((clusterwise_reconst[leaf_ind].shape[0])**.5))

k=0
for l in range(n_leaves):
        fig, axs = plt.subplots(n_grid, n_grid, figsize=(4,4))
        i=0
        for a in range(n_grid):
            for b in range(n_grid):
                try:
                        axs[a,b].set_axis_off()
                        axs[a,b].imshow(display_image(clusterwise_reconst[k][i]), cmap=plt.get_cmap('gray'))
                        i+=1
                except Exception as e:
                    print(e)
                    pass
        fig.suptitle(f"Leaf {k} samples",fontsize=25)
        fig.tight_layout()
        fig.subplots_adjust(top=0.87)
        k+=1
plt.show()

# %%
# For closer inspection, one can select a specific leaf by leaf_ind and investigate more generations.
leaf_ind = 0
    
n_grid = int((clusterwise_reconst[leaf_ind].shape[0])**.5)
fig, axs = plt.subplots(n_grid, n_grid, figsize=(15,15))

i=0
for a in range(n_grid):
    for b in range(n_grid):
        axs[a,b].set_axis_off()
        axs[a,b].imshow(display_image(clusterwise_reconst[leaf_ind][i]), cmap=plt.get_cmap('gray'))
        i+=1
fig.suptitle(f"Leaf {leaf_ind} samples",fontsize=25)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

# %% [markdown]
# ## <a name="section_3"></a> 3. Reconstructions

# %% [markdown]
# This section is concerned with computing reconstructions of input samples.

# %% [markdown]
# ### Clusterwise reconstructions

# %% [markdown]
# Here, given one input image, we visualize the reconstructions for each leaf. That is, each row corresponds to one input image and each column corresponds to one leaf. Above each reconstruction, we provide the probability of falling into the respective leaf for this sample. 
# 
# This way of visualization can provide insights on the characteristics each leaf is associated with. Observe that the reconstructions differ across the leaves, as each leaf reconstructs the image in the style of the cluster that it learnt.

# %%
# Training Set
gen_train_eval_iter = iter(gen_train_eval)
inputs, labels = next(gen_train_eval_iter)


inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
with torch.no_grad():
    reconstructions_gpu, node_leaves_gpu = model.compute_reconstruction(inputs_gpu)
reconstructions = move_to(reconstructions_gpu, 'cpu')
node_leaves = move_to(node_leaves_gpu, 'cpu')


for i in range(10):
    print("Class:", labels[i].item())
    fig, axs = plt.subplots(1, n_d+1, figsize=(15, 15))
    axs[n_d].imshow(display_image(inputs[i]), cmap=plt.get_cmap('gray'))
    axs[n_d].set_title("Original")
    axs[n_d].axis('off')
    for c in range(n_d):
        axs[c].imshow(display_image(reconstructions[c][i]), cmap=plt.get_cmap('gray'))
        axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(node_leaves[c]['prob'][i]))
        axs[c].axis('off')

    plt.show()

# %%
# Test Set
gen_test_iter = iter(gen_test)
inputs, labels = next(gen_test_iter)
inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
with torch.no_grad():
    reconstructions_gpu, node_leaves_gpu = model.compute_reconstruction(inputs_gpu)
reconstructions = move_to(reconstructions_gpu, 'cpu')
node_leaves = move_to(node_leaves_gpu, 'cpu')


for i in range(10):
    print("Class:", labels[i].item())
    fig, axs = plt.subplots(1, n_d+1, figsize=(15, 15))
    axs[n_d].imshow(display_image(inputs[i]), cmap=plt.get_cmap('gray'))
    axs[n_d].set_title("Original")
    axs[n_d].axis('off')
    for c in range(n_d):
        axs[c].imshow(display_image(reconstructions[c][i]), cmap=plt.get_cmap('gray'))
        axs[c].set_title(f"L{c}: " + f"p=%.2f" % torch.round(node_leaves[c]['prob'][i]))
        axs[c].axis('off')

    plt.show()

# %% [markdown]
# ### Group reconstructions according to cluster assignment

# %% [markdown]
# In this subsection, given a leaf, we store the reconstructions of the first 100 samples, for which this leaf is their most likely cluster assignment. This allows us to visualize for each leaf, which samples fall into it, in order to gain insights on the cluster that each leaf learnt.

# %%
# Test Set
# Here, we store samples for each leaf simultaneously by iterating through the training set until
# every leaf has n_imgs associated samples, or we iterated through max_iter batches.
max_iter = 100
n_imgs = configs['training']['batch_size']

n_iter=0
gen_test_iter = iter(gen_test)
inputs, labels = next(gen_test_iter)
inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
with torch.no_grad():
    reconstructions_gpu, node_leaves_gpu = model.compute_reconstruction(inputs_gpu)
reconstructions = move_to(reconstructions_gpu, 'cpu')
node_leaves = move_to(node_leaves_gpu, 'cpu')
p_c_z = torch.stack([node_leaves[i]['prob'] for i in range(len(node_leaves))],1)
clusterwise_reconst = [torch.zeros_like(reconstructions[0][0:2]) for i in range(len(reconstructions))]
while min([clusterwise_reconst[leaf_ind].shape[0] for leaf_ind in range(len(reconstructions))]) < n_imgs+2 and n_iter < max_iter:
    n_iter += 1
    if n_iter %10 == 0:
        print(n_iter)
    for i in range(n_imgs):
        leaf_ind = p_c_z[i].numpy().argmax()
        if clusterwise_reconst[leaf_ind].shape[0] < n_imgs+2:
            clusterwise_reconst[leaf_ind] = torch.vstack([clusterwise_reconst[leaf_ind], reconstructions[leaf_ind][i].unsqueeze(0)])
    inputs, labels = next(gen_test_iter)
    inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
    with torch.no_grad():
        reconstructions_gpu, node_leaves_gpu = model.compute_reconstruction(inputs_gpu)
    reconstructions = move_to(reconstructions_gpu, 'cpu')
    node_leaves = move_to(node_leaves_gpu, 'cpu')
    p_c_z = torch.stack([node_leaves[i]['prob'] for i in range(len(node_leaves))],1)
for i in range(len(reconstructions)):
    clusterwise_reconst[i] = clusterwise_reconst[i][2:,:]

# %%
# For each leaf, we visualize n_grid x n_grid reconstructions of samples, 
# which have highest probability of being assigned to this cluster
n_leaves = len(clusterwise_reconst)
n_grid = min(5,int((clusterwise_reconst[leaf_ind].shape[0])**.5))

k=0
for l in range(n_leaves):
        fig, axs = plt.subplots(n_grid, n_grid, figsize=(4,4))
        i=0
        for a in range(n_grid):
            for b in range(n_grid):
                axs[a,b].set_axis_off()
                axs[a,b].imshow(display_image(clusterwise_reconst[k][i]), cmap=plt.get_cmap('gray'))
                i+=1
        fig.suptitle(f"Leaf {k} samples",fontsize=25)
        fig.tight_layout()
        fig.subplots_adjust(top=0.87)
        k+=1
plt.show()

# %%
# For closer inspection, one can select a specific leaf by leaf_ind and investigate more reconstructions.
leaf_ind = 0
    
n_grid = int((clusterwise_reconst[leaf_ind].shape[0])**.5)
fig, axs = plt.subplots(n_grid, n_grid, figsize=(15,15))

i=0
for a in range(n_grid):
    for b in range(n_grid):
        axs[a,b].set_axis_off()
        axs[a,b].imshow(display_image(clusterwise_reconst[leaf_ind][i]), cmap=plt.get_cmap('gray'))
        i+=1
fig.suptitle(f"Leaf {leaf_ind} samples",fontsize=25)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

# %% [markdown]
# ## <a name="section_4"></a> 4. Tree and Representation Analysis 

# %% [markdown]
# In this section we explore the structure of the learnt tree as well as the representations

# %% [markdown]
# ### Tree embeddings

# %% [markdown]
# Below, we visualize the learnt embeddings by performing PCA on each node. Set use_pca to False if you want to directly see the first two dimensions without dimensionality reduction.

# %%
# Do you want to look at pca embeddings or learnt representations
use_pca = True

# pick data loader
data_loader = gen_test


# each entry in node_embeddings is a dictionary with keys 'prob' and 'z_sample' for each leaf
nb_nodes = len(data_tree)
node_embeddings = [{'prob': [], 'z_sample': []} for _ in range(nb_nodes)]
label_list = []

# iterate over test data points
for inputs, labels in tqdm(data_loader):
    inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)

    label_list.append(labels)

    with torch.no_grad():
        node_info = get_node_embeddings(model, inputs_gpu)
    node_info = move_to(node_info, 'cpu')

    # for each node, append the probability and z_sample to the list

    k = 0 # need this variable to skip "no digits" nodes
    for i in range(nb_nodes):
        j = i - k
        if data_tree[i][1] == 'no digits':
            k += 1
            continue

        node_embeddings[i]['prob'].append(node_info[j]['prob'].numpy())
        node_embeddings[i]['z_sample'].append(node_info[j]['z_sample'].numpy())

# flatten the lists
k = 0
for i in range(nb_nodes):
    if data_tree[i][1] == 'no digits':
        node_embeddings[i]['prob'] = []
        node_embeddings[i]['z_sample'] = []
        continue
    
    node_embeddings[i]['prob'] = np.concatenate(node_embeddings[i]['prob'])
    node_embeddings[i]['z_sample'] = np.concatenate(node_embeddings[i]['z_sample'])

label_list = np.concatenate(label_list)

# Draw the tree graph with scatter plots as nodes and arrows for edges
draw_tree_with_scatter_plots(data_tree, node_embeddings, label_list, pca = use_pca)

# %% [markdown]
# ### Leaf embeddings

# %% [markdown]
# Below, we visualize the learnt leaf embeddings after performing PCA. This allows for a closer inspection of the leaf embeddings, which are also visualized in the tree above.

# %%
# get leaf embeddings for each test data point
gen_test_iter = iter(gen_test)
inputs, labels = next(gen_test_iter)
inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)
with torch.no_grad():
    reconstructions_gpu, node_leaves_gpu = model.compute_reconstruction(inputs_gpu)
reconstructions = move_to(reconstructions_gpu, 'cpu')
node_leaves = move_to(node_leaves_gpu, 'cpu')

# each entry in node_leaves is a dictionary with keys 'prob' and 'z_sample' for each leaf
node_leaves = [{'prob': [], 'z_sample': []} for _ in range(n_d)]
label_list = []

for inputs, labels in tqdm(gen_test):
    inputs_gpu, labels_gpu = inputs.to(device), labels.to(device)

    label_list.append(labels)

    with torch.no_grad():
        _, node_leaves_gpu = model.compute_reconstruction(inputs_gpu)
        node_leaves_cpu = move_to(node_leaves_gpu, 'cpu')
        
    # for each leaf, append the probability and z_sample to the list
    for i in range(n_d):
        node_leaves[i]['prob'].append(node_leaves_cpu[i]['prob'].numpy())
        node_leaves[i]['z_sample'].append(node_leaves_cpu[i]['z_sample'].numpy())

# flatten the lists
for i in range(n_d):
    node_leaves[i]['prob'] = np.concatenate(node_leaves[i]['prob'])
    node_leaves[i]['z_sample'] = np.concatenate(node_leaves[i]['z_sample'])

label_list = np.concatenate(label_list)

# %%
# visualize z_sample for each leaf, do PCA and plot in 2D
from sklearn.decomposition import PCA

# PCA on node_leaves['z_sample']
colors = label_list
plt.figure(figsize=(20, 10))

for i in range(n_d):
    z_sample = node_leaves[i]['z_sample']
    weights = node_leaves[i]['prob']

    pca = PCA(n_components=2)
    z_sample_pca = pca.fit_transform(z_sample)

    plt.subplot(2, -(-len(node_leaves)//2), i+1)
    plt.scatter(z_sample_pca[:, 0], z_sample_pca[:, 1], c=colors, cmap='tab10', alpha=weights)
    plt.title(f"Leaf {i}")
    plt.colorbar()
    plt.xlabel("PC1")
    plt.ylabel("PC2")

plt.tight_layout()

# %% [markdown]
# ## <a name="section_5"></a> 5. CelebA attributes

# %% [markdown]
# This section is designated for analyzing the learnt splits and clusters of datasets without ground truth cluster labels, but various attributes. It is designed with a focus on CelebA.

# %%
assert configs['data']['data_name'] == 'celeba'
import pandas as pd
data_dir = './data/celeba/'
attr = pd.read_csv(data_dir+'/list_attr_celeba.txt', sep="\s+", skiprows=1)
y_test = attr[182637:]
y_train = attr[:162770]

# %% [markdown]
# ### Calculate cluster-matching attributes

# %% [markdown]
# Preprocessing step where we store for every node, the indeces of the test samples, whose most likely path went through said node

# %%
# Change to leafwise view of samples
prob_leaves = predict(gen_test, model, device,'prob_leaves')
y = np.squeeze(np.argmax(prob_leaves, axis=-1))
sample_ind = []
for i in range(len(np.unique(y))):
    sample_ind.append([])
for i in np.unique(y):
    sample_ind[i] = np.where(y==i)[0]
    
# Fill all internal nodes and create datatree with corresponding samples
data_tree_ids = []
for i in range(len(data_tree)):
    data_tree_ids.append([i,[]])
for listnode in reversed(data_tree_ids):
    i = listnode[0]
    if data_tree[i][3] == 1:
        # If leaf, just copy samples from above
        data_tree_ids[i][1] = sample_ind[i-(len(data_tree_ids)-len(sample_ind))]
    else:
        # If internal node, take samples from children
        children = []
        for j in range(len(data_tree)):
            if data_tree[j][2] == i:
                children.append(j)
        assert len(children)==2
        data_tree_ids[i][1] = np.sort(np.concatenate((data_tree_ids[children[0]][1],data_tree_ids[children[1]][1])))
        
        
        
# Final ID-tree, where for each node, we store which test sample went through it
data_tree_ids

# %% [markdown]
# For each split, we additionally store the five attribute that correlate most highly with the the split. This gives an intuition on what attributes the split is based on, i.e. which characteristics the split differentiates between.
# 
# Note that for CelebA, the "ground truth" attributes are in our opinion not the most descriptive ones regarding overall image&cluster impression and focus sometimes on details, on which we don't pick up on.

# %%
# Highest correlated features per split
data_tree_new = data_tree.copy()
for i in range(len(data_tree_ids)):
    in_leaf = False
    node_ind = data_tree_ids[i][1]
    # Samples in node before split
    node_samples = y_test.iloc[node_ind]
    # Split of samples
    node_split = np.zeros(len(y_test))
    children = []
    for j in range(len(data_tree)):
        if data_tree[j][2] == i:
            children.append(j)
        if children == []:
            in_leaf = True
        else:
            in_leaf = False
    if not in_leaf: 
        child_left = children[0]
        node_split[data_tree_ids[child_left][1]] = 1
        node_split = node_split[node_ind]
        # Store corr coefficients
        corr = np.corrcoef(np.concatenate((np.array(node_samples),np.expand_dims(node_split,1)),1).T)[len(y_test.columns),0:len(y_test.columns)]
        data_tree_ids[i].append(corr)
        
        # Store 5 strongest correlations
        ind = np.abs(corr).argsort()[-5:][::-1]
        features = y_test.columns[ind].tolist()
        for k in range(len(ind)):
            if corr[ind[k]] < 0:
                features[k] = 'not ' + features[k]
            features[k] = features[k] + ' ({})'.format(round(corr[ind[k]], 2))
        data_tree_ids[i].append(features)
        
data_tree_ids

# %% [markdown]
# As a summary, for each attribute, we print the split that has the highest correlation with it. This gives an intuition on what internal node differentiates the most according to a given attribute.

# %%
# Attributewise node with highest correlation (i.e. internal node that was splitting attribute the most)
attr_maxnode = y_test.columns.tolist()
for i in range(len(y_test.columns)):
    attrcorr = []
    for node in range(len(data_tree_ids)):
        if len(data_tree_ids[node])==len(data_tree_ids[0]):
            attrcorr.append(data_tree_ids[node][2][i])
    attrcorr = np.array(attrcorr)
    if len(np.argwhere(np.isnan(attrcorr)).squeeze(1))>0:
        attrcorr[np.argwhere(np.isnan(attrcorr)).squeeze(1)] = 0
    ind = np.argmax(np.abs(attrcorr))
    attr_maxnode[i] = attr_maxnode[i] + ": " + f'{ind}' ' ({})'.format(round(attrcorr[ind], 2))
attr_maxnode

# %% [markdown]
# ### Evaluation of clusters

# %% [markdown]
# We can analyze the clustering quality according to certain attributes. To do this, in the second cell, pick the indeces of the attributes, whose intersections you want to determine as ground truth clusterings. Then, the NMI is calculated for treating the selected attributes' intersections as ground-truth clusters.

# %%
y_test.columns

# %%
# Pick labels here
label_ind = [2,20,39]
print([attr_maxnode[i] for i in label_ind])

# %%
if len(label_ind)==2:
    label_dict = {
        (-1, -1): 0,
        (-1, 1): 1,
        (1, -1): 2,
        (1, 1): 3
    }
else:
    label_dict = {
        (-1, -1, -1): 0,
        (-1, -1, 1): 1,
        (-1, 1, -1): 2,
        (-1, 1, 1): 3,
        (1, -1, -1): 4,
        (1, -1, 1): 5,
        (1, 1, -1): 6,
        (1, 1, 1): 7
    }
selected_classes = np.array(y_test.iloc[:, label_ind])
selected_classes = [tuple([x for x in a]) for a in selected_classes]
label_true = [label_dict[sample_labels] for sample_labels in selected_classes]

# %%
print('NMI:')
normalized_mutual_info_score(y, label_true)

# %% [markdown]
# ### Create attribute-wise percentage table for leaves

# %% [markdown]
# This subsection presents the frequency of the attributes for each leaf. The numbers indicate the percentage of samples assigned to a given leaf, that contain a certain attribute. For example: 67% of all people assigned to leaf 3 are blonde.

# %%
leaf_attr = []
n_leaves = len(np.unique(y))
for i in range(1,1+n_leaves):
    data_tree_ids[-i].append((y_test.iloc[data_tree_ids[-i][1]] == 1).mean())
    leaf_attr.append((y_test.iloc[data_tree_ids[-i][1]] == 1).mean())

# %%
leaf_attr_table = pd.DataFrame(np.stack(leaf_attr)[::-1])
leaf_attr_table.columns = y_test.columns

pd.set_option('display.max_columns', None)
leaf_attr_table.round(3)*100

# %% [markdown]
# Here, one can create new attributes by combining previous attributes

# %%
new_vars =[]
for i in range(1,1+n_leaves):
    temp = y_test.iloc[data_tree_ids[-i][1]] == 1
    temp['Hair_Loss'] = np.clip(temp['Bald'] + temp['Receding_Hairline'],0,1)
    temp['Dark_Hair'] = np.clip(temp['Brown_Hair'] + temp['Black_Hair'],0,1)
    temp['Happy'] = np.clip(temp['Smiling'] + temp['Mouth_Slightly_Open'],0,1)
    temp['Light_Hair'] = np.clip(temp['Blond_Hair'] + temp['Gray_Hair'],0,1)
    temp['Beard'] = np.clip(temp['5_o_Clock_Shadow'] + 1-temp['No_Beard'],0,1)

    new_vars.append([temp['Hair_Loss'].mean(),temp['Dark_Hair'].mean(),temp['Happy'].mean(),temp['Light_Hair'].mean(),temp['Beard'].mean()])
    
new_vars_table = pd.DataFrame(np.stack(new_vars)[::-1])
new_vars_table.columns = ['Hair_Loss','Dark_Hair','Happy','Light_Hair','Beard']
new_vars_table.round(3)*100

# %%



