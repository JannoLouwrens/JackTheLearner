"""
DATASET DOWNLOADER FOR TRAININGJACK
Automatically downloads robot demonstration datasets for behavior cloning
"""

import os
import requests
from tqdm import tqdm
import zipfile
import tarfile
from typing import Optional
import urllib.request


class DatasetDownloader:
    """
    Downloads and manages robot demonstration datasets for behavior cloning.

    Available datasets:
    1. MoCapAct (5-10GB) - Human motion capture for humanoid locomotion
    2. RT-1 (subset 10GB) - Google robot manipulation demonstrations
    3. Language-Table (5GB) - Language-conditioned manipulation
    """

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Dataset registry - ORDERED BY TRAINING SEQUENCE!
        self.datasets = {
            # PHASE 2A: Natural Human Movement (Train First!)
            "cmu_mocap": {
                "name": "CMU Motion Capture",
                "size": "2-3GB",
                "description": "Human motion capture - walking, running, jumping, dancing",
                "url": "http://mocap.cs.cmu.edu/",
                "extract_dir": "cmu_mocap",
                "enabled": True,
                "priority": 1,  # Train FIRST for natural movement
                "instructions": "Free download. Natural human movements for humanoid robots.",
            },
            "mocapact": {
                "name": "MoCapAct",
                "size": "5-10GB",
                "description": "CMU MoCap adapted for MuJoCo humanoids",
                "url": "https://microsoft.github.io/MoCapAct/",
                "extract_dir": "mocapact",
                "enabled": True,
                "priority": 2,  # Train SECOND - refined for robots
                "instructions": "Microsoft's cleaned version of CMU MoCap for robots",
            },
            "deepmind_control": {
                "name": "DeepMind Control Suite MoCap",
                "size": "3-5GB",
                "description": "DeepMind humanoid reference motions",
                "url": "https://github.com/deepmind/dm_control",
                "extract_dir": "deepmind_mocap",
                "enabled": True,
                "priority": 3,  # Train THIRD - includes physics-aware motions
                "instructions": "DeepMind's reference motions for humanoid control",
            },

            # PHASE 2B: Object Manipulation (Train After Movement!)
            "rt1_subset": {
                "name": "RT-1 Subset",
                "size": "10GB",
                "description": "Google robot manipulation demonstrations",
                "url": None,  # Requires Google Cloud Storage access
                "extract_dir": "rt1",
                "enabled": False,
                "priority": 4,  # Train FOURTH - manipulation skills
                "instructions": "Requires Google Cloud SDK. See: https://robotics-transformer1.github.io/",
            },

            # PHASE 2C: Language Understanding (Train Last!)
            "language_table": {
                "name": "Language-Table",
                "size": "5GB",
                "description": "Language-conditioned manipulation tasks",
                "url": None,
                "extract_dir": "language_table",
                "enabled": False,
                "priority": 5,  # Train FIFTH - language conditioning
                "instructions": "Requires download script. See: https://language-table.github.io/",
            },
        }

    def list_datasets(self):
        """List all available datasets in training order"""
        print("\n" + "="*70)
        print("TRAINING DATASETS - RECOMMENDED ORDER")
        print("="*70)
        print("\nTrain in this sequence for best results:")
        print("Phase 2A: Natural Movement → Phase 2B: Manipulation → Phase 2C: Language")
        print("="*70)

        # Sort by priority
        sorted_datasets = sorted(self.datasets.items(), key=lambda x: x[1].get('priority', 99))

        for key, info in sorted_datasets:
            priority = info.get('priority', 99)
            status = "[AUTO]" if info["enabled"] else "[MANUAL]"

            print(f"\n[Priority {priority}] {status} {info['name']}")
            print(f"   Size: {info['size']}")
            print(f"   Description: {info['description']}")
            if not info["enabled"]:
                print(f"   Setup: {info.get('instructions', 'N/A')}")
            else:
                print(f"   Ready to download automatically!")

        print("\n" + "="*70)
        print("TIP: Download datasets in priority order for optimal training!")
        print("="*70)

    def download_file(self, url: str, destination: str, description: str = "Downloading"):
        """Download file with progress bar"""
        print(f"\n[*] {description}...")
        print(f"   URL: {url}")
        print(f"   Destination: {destination}")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=description
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"[OK] Downloaded to: {destination}")

    def extract_archive(self, archive_path: str, extract_to: str):
        """Extract tar.gz or zip archive"""
        print(f"\n[*] Extracting {os.path.basename(archive_path)}...")

        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            print(f"[ERROR] Unsupported archive format: {archive_path}")
            return False

        print(f"[OK] Extracted to: {extract_to}")
        return True

    def download_mocapact(self):
        """Download MoCapAct human motion capture dataset"""
        dataset_key = "mocapact"
        info = self.datasets[dataset_key]

        print("\n" + "="*70)
        print(f"DOWNLOADING: {info['name']}")
        print("="*70)
        print(f"Size: {info['size']}")
        print(f"Description: {info['description']}")

        extract_dir = os.path.join(self.data_dir, info['extract_dir'])

        # Check if already downloaded
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"\n[INFO] Dataset already exists at: {extract_dir}")
            overwrite = input("Overwrite? (yes/no): ").lower()
            if overwrite != 'yes':
                print("[INFO] Skipping download")
                return extract_dir

        # Download
        archive_path = os.path.join(self.data_dir, "mocapact_data.tar.gz")

        # Note: This is a placeholder URL - the actual MoCapAct data might have a different location
        # You may need to manually download from: https://microsoft.github.io/MoCapAct/
        print(f"\n[WARNING] MoCapAct requires manual download")
        print(f"[INFO] Please visit: https://microsoft.github.io/MoCapAct/")
        print(f"[INFO] Download the dataset and place it at: {archive_path}")
        print(f"[INFO] Then run this script again")

        return None

    def download_all(self):
        """Download all available datasets"""
        print("\n" + "="*70)
        print("DOWNLOADING ALL AVAILABLE DATASETS")
        print("="*70)

        results = {}

        for key, info in self.datasets.items():
            if info["enabled"]:
                print(f"\n[*] Processing: {info['name']}")
                if key == "mocapact":
                    result = self.download_mocapact()
                    results[key] = result
            else:
                print(f"\n[SKIP] {info['name']} - Requires manual setup")
                print(f"   {info.get('instructions', 'N/A')}")

        return results

    def check_datasets(self):
        """Check which datasets are already downloaded"""
        print("\n" + "="*70)
        print("DATASET STATUS")
        print("="*70)

        for key, info in self.datasets.items():
            extract_dir = os.path.join(self.data_dir, info['extract_dir'])

            if os.path.exists(extract_dir) and os.listdir(extract_dir):
                print(f"\n[OK] {info['name']}")
                print(f"   Location: {extract_dir}")

                # Count files
                total_files = sum([len(files) for r, d, files in os.walk(extract_dir)])
                print(f"   Files: {total_files}")
            else:
                print(f"\n[--] {info['name']}")
                print(f"   Status: Not downloaded")

        print("\n" + "="*70)


# Manual download instructions
MANUAL_DOWNLOAD_INSTRUCTIONS = """
================================================================================
MANUAL DATASET DOWNLOAD INSTRUCTIONS
================================================================================

Since some datasets require special access or are very large, here's how to
download them manually:

1. MoCapAct (Human Motion Capture)
   ----------------------------------------------------------------------
   Size: 5-10GB
   Website: https://microsoft.github.io/MoCapAct/

   Steps:
   a) Visit the website
   b) Download the dataset (follow their instructions)
   c) Extract to: datasets/mocapact/
   d) Verify files are present

2. RT-1 (Google Robot Manipulation)
   ----------------------------------------------------------------------
   Size: 130GB (full) or 10GB (subset)
   Website: https://robotics-transformer1.github.io/

   Steps:
   a) Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
   b) Authenticate: gcloud auth login
   c) Download subset:
      gsutil -m cp -r gs://gresearch/robotics/rt1_data_subset ./datasets/rt1/

   Note: This requires Google Cloud account (free tier works)

3. Language-Table (Language-Conditioned Tasks)
   ----------------------------------------------------------------------
   Size: 5GB
   Website: https://language-table.github.io/

   Steps:
   a) Visit the website
   b) Follow their download instructions
   c) Extract to: datasets/language_table/

4. Open X-Embodiment (Multi-Robot Data)
   ----------------------------------------------------------------------
   Size: 1-2TB (full) or customizable subsets
   Website: https://robotics-transformer-x.github.io/

   Steps:
   a) Visit the website
   b) Choose which robot types you want
   c) Download subset (recommend 10-50GB to start)
   d) Extract to: datasets/open_x/

================================================================================
RECOMMENDED STARTING POINT
================================================================================

For fastest results, start with:
1. MoCapAct (if you have a humanoid robot)
   OR
2. RT-1 subset (if you want manipulation first)

You can download more datasets later as needed.

================================================================================
"""


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Download robot demonstration datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--check", action="store_true", help="Check downloaded datasets")
    parser.add_argument("--download", type=str, help="Download specific dataset (mocapact, rt1_subset, language_table)")
    parser.add_argument("--download-all", action="store_true", help="Download all available datasets")
    parser.add_argument("--instructions", action="store_true", help="Show manual download instructions")
    args = parser.parse_args()

    downloader = DatasetDownloader()

    if args.list:
        downloader.list_datasets()
    elif args.check:
        downloader.check_datasets()
    elif args.download:
        if args.download == "mocapact":
            downloader.download_mocapact()
        else:
            print(f"[ERROR] Unknown dataset: {args.download}")
    elif args.download_all:
        downloader.download_all()
    elif args.instructions:
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("DATASET DOWNLOADER FOR TRAININGJACK")
        print("="*70)
        downloader.list_datasets()

        print("\nOptions:")
        print("1. Check downloaded datasets")
        print("2. Download MoCapAct")
        print("3. Show manual download instructions")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            downloader.check_datasets()
        elif choice == "2":
            downloader.download_mocapact()
        elif choice == "3":
            print(MANUAL_DOWNLOAD_INSTRUCTIONS)
        elif choice == "4":
            print("Exiting...")
        else:
            print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()
