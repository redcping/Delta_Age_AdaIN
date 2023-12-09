#Copy pasted from our colab notebook
#####################################################################################
# For the following models go through the test pipeline
def plot_test_l1_losses(trainer,l1_losses):
    interval = 10
    avg_l1_losses = [sum(l1_losses[i:i+interval])/interval for i in range(0, len(l1_losses), interval)]
    plt.figure(figsize=(10, 5))
    plt.plot(avg_l1_losses)
    plt.title('Average L1 Loss per Interval')
    plt.xlabel('Interval')
    plt.ylabel('Average L1 Loss')
    plt.savefig(f'{trainer.save_model_dir}/test_l1_loss_{datetime.now().strftime("%m-%d_%H")}.png')

def find_file_with_epoch(directory, search_string="epoch_50"):
    """
    Search for a file in the given directory that contains the search_string in its name.

    :param directory: Directory to search in
    :param search_string: String to look for in file names
    :return: The name of the first file that contains the search_string, or None if no such file is found
    """
    for filename in os.listdir(directory):
        if search_string in filename:
            return filename
    return None

def testing_pipeline(project_dir, models, epoch_list):
    # Iterating through the models dictionary
    for dataset, augmentation_types in models.items():
        print(f"\n\n\nDataset: {dataset}")
        for augmentation, model_name in augmentation_types.items():
            print(f"\n\n  Model: {augmentation}")
            print(f"    Model Name: {model_name}")


            model_path = os.path.join(project_dir, model_name)
            for epoch in epoch_list:
                print(f"\n\nEpochs: {epoch}\n\n")
                model_file = find_file_with_epoch(model_path, f"epoch_{epoch}")

                if model_file is None:
                  print(f"Could not find model with {augmentation} epoch: {epoch}, modelfile: {model_file}")
                  continue

                config = Config()
                config.pretrained_fn = os.path.join(model_path, model_file)
                config.datanames = dataset
                config.save_model_folder()

                # start testing with test data
                trainer = DAATrainer(config)
                mae, l1_losses = trainer.test()

                # plot graphs
                plot_test_l1_losses(trainer,l1_losses)
        print()  # Adding a newline for better readability

project_dir = "/content/gdrive/MyDrive/CV_DAA_PROJ/DAA_models/"
models = {
    # "megaage_asian": {
        # "No Augmentation":"no_augmentation_VF_50epoch_megaage_asian_resnet18_100_binary",
        # "Color Based Augmentation": "color_based_aug_VF_50epoch_megaage_asian_resnet18_100_binary",
        # "Distortion Based Augmentation": "distortion_based_aug_VF_TRIMMED_utkface_resnet18_100_binary",
        # "Distortion & Color Based Augmentation": "distortion_color_combined_aug_VF_50epoch_megaage_asian_resnet18_100_binary"
    # },
    "utkface": {
        # "No Augmentation":"no_augmentation_VF_TRIMMED_utkface_resnet18_100_binary",
        # "Color Based Augmentation": "color_based_aug_VF_TRIMMED_utkface_resnet18_100_binary",
        "Distortion Based Augmentation": "distortion_based_aug_TRIMMED_V2_utkface_resnet18_100_binary",
        # "Distortion & Color Based Augmentation": "distortion_color_combined_aug_VF_50epoch_utkface_resnet18_100_binary"
    }
}
epoch_list = [25,35,50]#[15,20,25,30,35,40,45,50]
testing_pipeline(project_dir, models, epoch_list)