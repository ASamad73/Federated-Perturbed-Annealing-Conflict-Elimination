import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from datasets import load_dataset
from utils import set_global_seed

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

train_transform_cifar = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std)
])
test_transform_cifar = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std)
])


def dirichlet_partition_noniid(dataset_targets, num_clients, alpha, min_size=10, rng=None):
    if rng is None:
        rng = np.random.RandomState(1234)
    labels = np.array(dataset_targets)
    num_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
    indices_per_client = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet([alpha] * num_clients)
        rng.shuffle(idx_c)
        counts = (proportions * len(idx_c)).astype(int)
        diff = len(idx_c) - np.sum(counts)
        while diff > 0:
            frac = proportions * len(idx_c) - counts
            idx = int(np.argmax(frac))
            counts[idx] += 1
            diff -= 1
        pointer = 0
        for k in range(num_clients):
            cnt = counts[k]
            if cnt > 0:
                portion = idx_c[pointer:pointer+cnt]
                indices_per_client[k].extend(portion)
                pointer += cnt

    for k in range(num_clients):
        if len(indices_per_client[k]) < min_size:
            donors = [j for j in range(num_clients) if len(indices_per_client[j]) > min_size]
            for d in donors:
                if len(indices_per_client[k]) >= min_size:
                    break
                indices_per_client[k].append(indices_per_client[d].pop())

    return indices_per_client


def build_cifar_clients(data_root, num_clients=10, alpha=0.1, batch_size=32, test_batch_size=256, num_workers=2, seed=0):
    set_global_seed(seed)
    
    cifar_train = CIFAR10(root=data_root, train=True, download=True, transform=train_transform_cifar)
    cifar_test = CIFAR10(root=data_root, train=False, download=True, transform=test_transform_cifar)
    
    train_targets = [int(x) for x in cifar_train.targets]
    
    indices_per_client = dirichlet_partition_noniid(train_targets, num_clients, alpha, rng=np.random.RandomState(seed))
    
    clients = []
    for k in range(num_clients):
        idxs = indices_per_client[k]
        subset = Subset(cifar_train, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        clients.append((f"client_{k}", loader))
    
    test_loader = DataLoader(cifar_test, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    
    return clients, test_loader, indices_per_client

class HFPACSDomainDataset(Dataset):
    def __init__(self, hf_ds, domain_name, transform=None):
        self.transform = transform
        self.indices = [i for i, ex in enumerate(hf_ds) if ex["domain"] == domain_name]
        self.hf_ds = hf_ds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.hf_ds[self.indices[idx]]
        img = ex["image"]           
        label = int(ex["label"])    
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def build_pacs_clients(held_out_domain="sketch", batch_size=32):
    ds = load_dataset("flwrlabs/pacs")
    hf_split = ds["train"]
    all_domains = sorted(list(set(hf_split["domain"])))
    
    # PACS standard transforms
    train_tf = T.Compose([
        T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    test_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    clients_list = []
    test_loader = None
    
    for dom in all_domains:
        if dom == held_out_domain:
            test_ds = HFPACSDomainDataset(hf_split, dom, transform=test_tf)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        else:
            client_ds = HFPACSDomainDataset(hf_split, dom, transform=train_tf)
            loader = DataLoader(client_ds, batch_size=batch_size, shuffle=True)
            clients_list.append((dom, loader))
            
    return clients_list, test_loader