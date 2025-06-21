from lsnn.raw.hypotactic._hypotactic import Hypotactic

def run():
    source_name = "hypotactic"
    
    Hypotactic().run(source_path=f"landing_zone/{source_name}", destination_path=f"raw/{source_name}")


if __name__ == "__main__":
    run()
