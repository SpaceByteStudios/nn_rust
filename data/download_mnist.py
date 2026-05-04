import urllib.request
import os

def download_mnist():
    # Use a reliable mirror since the official site often fails
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    
    files = [
        "train-images-idx3-ubyte.gz", 
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", 
        "t10k-labels-idx1-ubyte.gz"
    ]

    print(f"Downloading MNIST data to: {os.getcwd()}")

    # Use a generic User-Agent to avoid being blocked
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for file in files:
        if not os.path.exists(file):
            print(f"Fetching {file}...")
            try:
                urllib.request.urlretrieve(base_url + file, file)
                print(f"Success: {file}")
            except Exception as e:
                print(f"Failed to download {file}: {e}")
        else:
            print(f"Skipped: {file} already exists.")

if __name__ == "__main__":
    download_mnist()
