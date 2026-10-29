from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

def evaluate_clustering(pred, labels):
    """Calculate ARI, NMI, AMI for clustering evaluation"""
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    ami = adjusted_mutual_info_score(labels, pred)
    return ari, nmi, ami