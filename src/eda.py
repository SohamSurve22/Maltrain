import os
import matplotlib.pyplot as plt

# Get project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "Malimg")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_PATH, exist_ok=True)


def analyze_dataset():
    families = sorted(os.listdir(DATASET_PATH))
    
    print(f"Total malware families: {len(families)}\n")
    
    family_counts = {}
    
    for family in families:
        family_path = os.path.join(DATASET_PATH, family)
        if os.path.isdir(family_path):
            count = len(os.listdir(family_path))
            family_counts[family] = count
            print(f"{family}: {count} samples")
    
    return family_counts


def plot_distribution(family_counts):
    plt.figure()
    plt.bar(family_counts.keys(), family_counts.values())
    plt.xticks(rotation=90)
    plt.title("Malware Family Distribution")
    plt.xlabel("Malware Families")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_PATH, "family_distribution.png")
    plt.savefig(save_path)
    print(f"\nSaved distribution plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    counts = analyze_dataset()
    plot_distribution(counts)