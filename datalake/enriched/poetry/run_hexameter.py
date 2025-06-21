from datalake.enriched.hexameter import Hexameter

def run():
    source_name = "hexameter"

    Hexameter().run(destination_path=f"enriched/{source_name}")


if __name__ == "__main__":
    run()
