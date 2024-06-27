import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def is_directory(path):
    # Check if the path is a directory
    return os.path.isdir(path), path

def init_process(_tqdm):
    # Without this init, tqdm will not work properly in multiprocessing
    global tqdm
    tqdm = _tqdm

def count_all_subdirectories(directory_path):
    # List all paths in the directory
    all_paths = [os.path.join(root, name) for root, dirs, files in os.walk(directory_path) for name in dirs + files]
    
    # Calculate the number of processes based on the available CPUs
    num_processes = cpu_count()

    # Create a pool of processes
    pool = Pool(processes=num_processes, initializer=init_process, initargs=(tqdm,))

    # Use pool.map to apply is_directory to all_paths, tqdm is used to display progress
    results = list(tqdm(pool.imap(is_directory, all_paths), total=len(all_paths), desc="Counting directories"))

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Count the number of True values returned by is_directory
    total_directories = sum(result[0] for result in results)
    return total_directories

# Specify the path to the directory
directory_path = "/mnt/xuanyuyi/data/gobjaverse_280k/"

# Get the total number of directories
total_directories = count_all_subdirectories(directory_path)

print(f"Total number of directories within {directory_path}: {total_directories}")
