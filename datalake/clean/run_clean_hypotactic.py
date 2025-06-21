from datalake.clean.clean_hypotactic import Hypotactic

def run():
    source_name = "hypotactic"
    
    Hypotactic().run(source_path=f"lsnn/bucket/raw/{source_name}", destination_path=f"lsnn/bucket/clean/{source_name}")


if __name__ == "__main__":
    run()
