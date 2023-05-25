import os


def get_data_configs():
    all_files = os.listdir("config")
    data_files = list(filter(lambda x: ".json" not in x, all_files))
    return data_files


def get_data_file_contents(data_file):
    with open(f"config/{data_file}", "r") as f:
        lines = f.readlines()
        filenames = list(map(lambda x: x.split()[-1]+".gnu", lines))
    return filenames


def get_all_gnu_files():
    return os.listdir("gnufiles")


def get_unused_files():
    all_config_gnu_files = []
    gnu_files = get_all_gnu_files()
    datafiles = get_data_configs()
    for datafile in datafiles:
        data_gnufiles = get_data_file_contents(datafile)
        all_config_gnu_files += data_gnufiles

    unused = []
    for file in gnu_files:
        if not file in all_config_gnu_files:
            unused.append(file)

    return unused


def get_number_unused_files():
    unused = get_unused_files()
    return len(unused)


def remove_unused_files():
    Lunused = get_unused_files()
    for file_unused in Lunused:
        os.remove(f"gnufiles/{file_unused}")


if __name__ == "__main__":
    os.chdir("qefiles")
    print(get_number_unused_files())
    remove_unused_files()
    print(get_number_unused_files())
