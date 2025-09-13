import os

def main():
    small = os.environ.get("AE_SMALL") == "1"
    if small:
        print("Running representation experiment (small)...")
    else:
        print("Running representation experiment...")

if __name__ == "__main__":
    main()
