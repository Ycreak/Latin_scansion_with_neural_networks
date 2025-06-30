import datalake.utilities as util
import polars as pl

class Poetry:
    """
    In the Poetry class we collect all lines from the cleaned layer and put them into one Polars dataframe.
    """
    def run(self, destination_path: str) -> None:
        raw_bucket_path: str= "datalake/bucket/clean"
    
        pedecerto_files: list = util.create_files_list(f"{raw_bucket_path}/pedecerto", 'json')
        pedecerto_files = [f"{raw_bucket_path}/pedecerto/" + s for s in pedecerto_files]
        hypotactic_files: list = util.create_files_list(f"{raw_bucket_path}/hypotactic", 'json')
        hypotactic_files = [f"{raw_bucket_path}/hypotactic/" + s for s in hypotactic_files]

        all_files = pedecerto_files + hypotactic_files

        print('Building dataframe.')
        all_rows = []
        line_counter = 1  # global line number across all files
        for file_path in all_files:
            lines = util.read_json(file_path)
            
            for line_data in lines:
                author = line_data["author"]
                meter = line_data["meter"]
                
                for entry in line_data["line"]:
                    if "-" in entry:
                        # We encode spaces a bit differently.
                        all_rows.append({
                            "author": author,
                            "meter": meter,
                            "line_number": line_counter,
                            "syllable": "-",
                            "label": "space",
                            "word": "space"
                        })
                    else:
                        all_rows.append({
                            "author": author,
                            "meter": meter,
                            "line_number": line_counter,
                            "syllable": entry["syllable"],
                            "label": entry["length"],
                            "word": entry["word"]
                        })
                
                line_counter += 1  # increment after processing each line

        # Final Polars DataFrame
        df = pl.DataFrame(all_rows)

        # Optional: inspect
        with pl.Config(tbl_rows=20):
            print(df)

        print('STATS ABOUT THIS DATAFRAME')

        df_count_grouped_by_meter = df.group_by("meter").agg(
            pl.col("line_number").n_unique().alias("unique_line_count")
        ).sort("unique_line_count", descending=True)

        with pl.Config(tbl_rows=200):
            print(df_count_grouped_by_meter)
        
        df.write_parquet(f"{destination_path}/poetry_dataframe.parquet")
