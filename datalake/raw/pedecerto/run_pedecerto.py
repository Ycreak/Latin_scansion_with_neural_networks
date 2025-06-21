from datalake.raw.pedecerto._pedecerto import Pedecerto

def run():
    source_name = "pedecerto"
    Pedecerto().run(source_path=f"lsnn/bucket/landing_zone/{source_name}", destination_path=f"lsnn/bucket/raw/{source_name}")


if __name__ == "__main__":
    run()
