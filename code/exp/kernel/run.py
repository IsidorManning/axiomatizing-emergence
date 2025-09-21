import os

def main():
    small = os.environ.get("AE_SMALL") == "1"
    if small:
        print("Running kernel experiment (small)...")
    else:
        print("Running kernel experiment...")

if __name__ == "__main__":
    main()
