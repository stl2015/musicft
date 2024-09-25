import json
import pandas as pd
from m2t.gcs_utils import list_files_with_extension


def read_ids_from_file(file_path):
    df = pd.read_json(file_path, lines=True)
    # df['keys'] = df.apply(lambda x: x['id']+'_'+str(x['start_secs'])+'_'+str(x['end_secs']))
    print('found {} records'.format(len(df)))
    return set(df['id'])


def find_unprocessed_records(source_file_path, processed_files, output_file_path):
    # Read IDs from all processed files
    processed_ids = set()
    for processed_file in processed_files:
        processed_ids.update(read_ids_from_file(processed_file))
    
     # Load the source file into a DataFrame
    source_df = pd.read_json(source_file_path, lines=True)
  
    # source_df['keys'] = source_df.apply(lambda x: x['id']+'_'+str(x['start_secs'])+'_'+str(x['end_secs']))  
    
    # Filter the DataFrame for unprocessed IDs
    unprocessed_df = source_df[~source_df['id'].isin(processed_ids)]
    
    # Save the unprocessed records to a JSONL file
    unprocessed_df.to_json(output_file_path, orient='records', lines=True)

    print(f"Unprocessed records saved to {output_file_path} with {len(unprocessed_df)} lines")


# Example usage
def gen_unprocessed():
    source_file_path = 'datasets/MAPS/MAPS-train-crop-annotation.jsonl'
    processed_files_path = 'datasets/MAPS/notes-caption'
    processed_files = list_files_with_extension(processed_files_path, 'jsonl')

    output_file_path = 'datasets/MAPS/todo/unprocessed_crop_annotations.jsonl'

    find_unprocessed_records(source_file_path, processed_files, output_file_path)


def combine_processed(output_file_path, 
                      processed_files_path = 'datasets/MAPS/notes-caption',
    ):

    processed_files = list_files_with_extension(processed_files_path, 'jsonl')

    processed = []
    for processed_file in processed_files:
        df = pd.read_json(processed_file, lines=True)
        print(f'processing {processed_file}, with {len(df)} lines')
        df = df[~df.start_secs.isna()]
        print(f'processed {processed_file}, with {len(df)} lines')

        processed.append(df)

    processed_df = pd.concat(processed)

    processed_df.to_json(output_file_path, orient='records', lines=True)

output_file_path = 'datasets/MAPS/MAPS_crop_notes_instruct.jsonl'
combine_processed(output_file_path)