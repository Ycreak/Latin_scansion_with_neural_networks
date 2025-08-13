from datalake.clean.clean_pedecerto import Pedecerto


def run():
    source_name = "pedecerto"

    Pedecerto().run(
        source_path=f"datalake/bucket/raw/{source_name}",
        destination_path=f"datalake/bucket/clean/{source_name}",
    )


if __name__ == "__main__":
    run()
