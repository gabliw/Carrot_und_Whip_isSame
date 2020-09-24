import os


# image file rename function
def rename(target_dir):
    # files = [os.path.join(target_dir, file) for file in os.listdir(target_dir)
    #          if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"))]

    files = [os.path.join(target_dir, file) for file in os.listdir(target_dir)]

    over_count = 0
    for idx, file in enumerate(sorted(files)):
        if file.split('/')[-1].split('.')[0] == f"{idx:03d}.png":
            continue
        elif os.path.exists(os.path.join(target_dir, f"{idx:03d}.png")) is True:
            over_count += 1

        file_name = f"{idx + over_count:03d}.png"
        print(file_name)
        os.rename(os.path.join(target_dir, file), os.path.join(target_dir, file_name))


rename('/Users/noble/workspace/NaNet/Dataset/target')