import os

def main():
    small = os.environ.get("AE_SMALL") == "1"
    if small:
        print("Running scaling smoke test...")
    else:
        print("Running scaling experiment...")

if __name__ == "__main__":
    main()
