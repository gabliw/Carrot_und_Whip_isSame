import os


# image file rename function
def rename(target_dir):
    files = [os.path.join(target_dir, file) for file in os.listdir(target_dir)]

    for idx, file in enumerate(sorted(files)):

        file_name = f"{idx + 1:03d}.png"
        print(file_name)
        os.rename(os.path.join(target_dir, file), os.path.join(target_dir, file_name))


rename('/Users/noble/workspace/NaNet/Dataset/target')