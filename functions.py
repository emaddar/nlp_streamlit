import pandas as pd


def process_ner_results(ner_results):
    # create dataframe from ner_results
    df = pd.DataFrame(ner_results)

    # add new column with entity after '-'
    df['entity_type'] = df['entity'].apply(lambda x: x.split('-')[1])
    # delete original entity column
    df = df.drop(columns=['entity'])

    # rename entity_type column to entity
    df = df.rename(columns={'entity_type': 'entity'})


    # define desired column order
    new_column_order = ['entity', 'word', 'score', 'index', 'start', 'end']

    # reindex columns in desired order
    df = df.reindex(columns=new_column_order)

    # group by entity and get count
    count_df = df.groupby('entity').count().reset_index()[['entity', 'score']]
    count_df = count_df.rename(columns={'score': 'count'})

    return df, count_df