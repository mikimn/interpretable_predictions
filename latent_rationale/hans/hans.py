from datasets import load_dataset


if __name__ == '__main__':
    ds = load_dataset('hans_dataset.py')
    for item in ds['train']:
        print(item)
        break
