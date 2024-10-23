import os
import hydra
from huggingface_hub import hf_hub_download

@hydra.main(config_name="download.yaml")
def main(cfg):
    # Define the path to the directory where you want to save the file
    save_directory = cfg.local_directory 
    repo_id = cfg.repo_id
    file_list = cfg.file_list

    # Download the file and save it to the specified directory
    for _file in file_list:
        path_train = hf_hub_download(
            repo_id=repo_id,  # Replace with your dataset repository ID
            filename=_file,     # Replace with the file you want to download
            repo_type="dataset",                        # This specifies that it is a dataset repo
            local_dir=save_directory                    # Specify your local directory here
        )
        print(f"File saved to: {path_train}")

if __name__=="__main__":
    main()
