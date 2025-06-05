from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sequence_preprocess import cgr, fcgr
import numpy as np

class CGRDataset(Dataset):
    def __init__(self, df, img_size=32):
        self.df = df
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]["sequence"]
        label = self.df.iloc[idx]["label"]
        img = cgr(sequence, self.img_size)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return img, label
    

class FCGRDataset(Dataset):
    def __init__(self, df, k):
        self.df = df
        self.k = k
        self.img_size = 2 ** k
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]["sequence"]
        label = self.df.iloc[idx]["label"]
        img = fcgr(sequence, self.k)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return img, label
    
class FCGRPreCompDataset(Dataset):
    def __init__(self, df):
        """
        Args:
            df: DataFrame with columns ['fcgr', 'label', 'genome_id']
                'fcgr' contains numpy arrays of shape (dim, dim)
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fcgr_img = self.df.iloc[idx]['fcgr'] 
        label = self.df.iloc[idx]['label']
        
        img_tensor = torch.tensor(fcgr_img, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor


class CIFAR10Net(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10Net, self).__init__()
        kernel_size = 7
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class CIFAR10NetFCGR(nn.Module):
    def __init__(self, num_classes, k):
        super(CIFAR10NetFCGR, self).__init__()
        kernel_size = 5
        padding = 1
        pool_kernel = 2
        pool_stride = 2
        num_layers = 3
        
        final_size = compute_final_size(
            k=k,
            kernel_size=kernel_size,
            padding=padding,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            num_layers=num_layers
        )
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_size * final_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def compute_final_size(k, kernel_size=5, padding=1, pool_kernel=2, pool_stride=2, num_layers=3):
    size = 2 ** k
    for _ in range(num_layers):
        size = size + 2*padding - kernel_size + 1 
        size = size // pool_stride  
    return size

def train(model, dataloader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) # type: ignore

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def weighted_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted', zero_division=0)

def weighted_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)

def df_to_fcgr(df, k):
    df_fcgr = df.copy()
    df_fcgr['fcgr'] = df_fcgr['sequence'].apply(lambda seq: fcgr(seq, k))
    return df_fcgr[['fcgr', 'label', 'genome_id']]

def skf_grouped(df, group, k=-1, n_splits = 5, batch_size = 32, max_epochs = 30, lr = 0.001):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch device", device)
    best_precision=(0.0,"")
    best_recall=(0,"")
    best_f1=(0,"")

    
    df = df_to_fcgr(df, k)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"], groups=df[group])):
        print(f"Fold {fold + 1}")

        num_classes = df["label"].nunique()
        num_worker = 4
        train_subset = df.iloc[train_idx]
        val_subset = df.iloc[val_idx]

        if k >= 1:
            train_dataset = FCGRPreCompDataset(train_subset)
            val_dataset = FCGRPreCompDataset(val_subset)
            # train_dataset = FCGRDataset(train_subset, k)
            # val_dataset = FCGRDataset(val_subset, k)
            model = CIFAR10NetFCGR(df.label.nunique(), k).to(device)
        else:
            train_dataset = CGRDataset(train_subset)
            val_dataset = CGRDataset(val_subset)
            model = CIFAR10Net(num_classes=num_classes).to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

        # model = CIFAR10Net(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # type: ignore
        criterion = torch.nn.CrossEntropyLoss()

        best_metric = -np.inf
        patience = 8
        counter = 0

        # Training loop
        for epoch in range(max_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            model.eval()
            correct, total = 0, 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)

                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

            acc = 100.0 * correct / total
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)




            str_output = f"Epoch {epoch+1}/{max_epochs}, Val Accuracy: {acc:.2f}%, "f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
            
            if precision > best_precision[0]:
                best_precision = (precision, str_output)
            if recall > best_recall[0]:
                best_recall = (recall, str_output)
            if f1 > best_f1[0]:
                best_f1 = (f1, str_output)
            
            print(str_output)
            
           
        permutation_test(model, val_loader, weighted_precision)
        permutation_test(model, val_loader, weighted_recall)
        permutation_test(model, val_loader, weighted_f1)

    print("Best Precision:", best_precision[1])
    print("Best Recall", best_recall[1])
    print("Best F1:", best_f1[1])


def permutation_test(model, val_loader, metric_fn, n_permutations=1000):
    """
    Perform permutation test on a trained model.
    
    Args:
        model: trained PyTorch model
        val_loader: DataLoader for validation data
        metric_fn: function to compute metric (e.g., accuracy_score, f1_score)
        n_permutations: number of permutations
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Original predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
    
    observed_score = metric_fn(all_targets, all_preds)

    # Permutation scores
    permuted_scores = []
    for _ in range(n_permutations):
        shuffled_targets = np.random.permutation(all_preds)
        score = metric_fn(all_targets, shuffled_targets)
        permuted_scores.append(score)

    # p-value = proportion of permuted scores ≥ observed score
    p_value = np.mean([score >= observed_score for score in permuted_scores])

    print(f"Observed {metric_fn.__name__}: {observed_score:.4f}")
    print(f"Permutation test p-value: {p_value:.4f}")

    return observed_score, p_value, permuted_scores



def gkf_grouped(df, group, k=-1, n_splits=5, batch_size=32, max_epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch device", device)
    best_precision = (0.0, "")
    best_recall = (0.0, "")
    best_f1 = (0.0, "")

    df = df_to_fcgr(df, k)
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df[group])):
        print(f"Fold {fold + 1}")
        num_classes = df["label"].nunique()
        num_worker = 4

        train_subset = df.iloc[train_idx]
        val_subset = df.iloc[val_idx]

        if k >= 1:
            train_dataset = FCGRPreCompDataset(train_subset)
            val_dataset = FCGRPreCompDataset(val_subset)
            model = CIFAR10NetFCGR(num_classes, k).to(device)
        else:
            train_dataset = CGRDataset(train_subset)
            val_dataset = CGRDataset(val_subset)
            model = CIFAR10Net(num_classes=num_classes).to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_metric = -np.inf
        patience = 8
        counter = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct, total = 0, 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

            acc = 100.0 * correct / total
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

            str_output = f"Epoch {epoch+1}/{max_epochs}, Val Accuracy: {acc:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
            if precision > best_precision[0]:
                best_precision = (precision, str_output)
            if recall > best_recall[0]:
                best_recall = (recall, str_output)
            if f1 > best_f1[0]:
                best_f1 = (f1, str_output)

            print(str_output)

            val_metric = acc
            if counter >= patience:
                print("Early stopping triggered")
                break
            if val_metric > best_metric:
                best_metric = val_metric
                counter = 0
                print("Reset Patience")
            else:
                counter += 1

        permutation_test(model, val_loader, weighted_precision)
        permutation_test(model, val_loader, weighted_recall)
        permutation_test(model, val_loader, weighted_f1)

    print("Best Precision:", best_precision[1])
    print("Best Recall:", best_recall[1])
    print("Best F1:", best_f1[1])
