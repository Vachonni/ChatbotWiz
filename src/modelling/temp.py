import hydra 

@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    print(config['parameters'])
    print(type(dict(config['parameters'])))


if __name__ == "__main__":
    main()